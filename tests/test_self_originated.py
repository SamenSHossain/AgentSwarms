"""Self-originated answers: an agent that posted the value before any other agent did keeps D = 1 (the value was public
before its report) but credits no source.  A planner summarises the researchers' findings before they report them, so
the planner's copy is a later repeat, not their source."""

import json

import pandas as pd
import pytest

from swarmprov import checks, exposure, graph, pipeline, query, report
from swarmprov.analysis import origins, structure
from swarmprov.schema import RunDir

CHAT = [  # (author, channel, text), a minute apart from 09:00
    ("planner", "general", "Kickoff: we need 2023 GDP growth for Poland, Hungary and Chile. Split up please."),
    ("researcher-1", "research", "Taking Poland. Searching the World Bank indicator page."),
    ("researcher-1", "research", "Found it: the World Bank page shows Poland 0.2 for 2023."),
    ("researcher-2", "research", "Hungary -0.9 per the World Bank page."),
    ("critic", "review", "@researcher-1 IMF says Poland 0.1, not 0.2. Which vintage?"),
    ("researcher-3", "research", "Chile 0.2 too, per the World Bank page."),
    ("planner", "general", "Summary so far: Poland 0.2, Hungary -0.9, Chile 0.2. Writer, please draft."),
    ("researcher-4", "research", "Late joiner: I pulled Poland from Eurostat: 0.1."),
    ("planner", "general", "Final answer submitted: Poland 0.2, Hungary -0.9, Chile 0.2."),
    ("researcher-2", "research", "I answered Hungary -0.9 in the form as well."),
    ("researcher-3", "research", "I answered Chile 0.2."),
    ("researcher-1", "research", "I answered Poland 0.2."),
    ("researcher-4", "research", "I answered Poland 0.1 (Eurostat)."),
]
T0 = pd.Timestamp("2026-05-04T09:00:00Z")


@pytest.fixture(scope="module")
def chat(tmp_path_factory):
    d = tmp_path_factory.mktemp("self_originated")
    path = d / "chat.jsonl"
    path.write_text("".join(json.dumps({"id": f"m{i}", "timestamp": (T0 + pd.Timedelta(minutes=i)).isoformat(),
                                        "author": a, "channel": c, "text": t}) + "\n"
                            for i, (a, c, t) in enumerate(CHAT)))
    cfg = d / "cfg.json"
    cfg.write_text(json.dumps({"bootstrap_reps": 0}))
    return pipeline.run_all(path, d / "run", adapter="chat", config=str(cfg))


def test_the_researchers_posted_first_and_credit_nobody(chat):
    ex = chat.read("exposures_merged").set_index("agent")
    own = ex.loc[["researcher-1", "researcher-2", "researcher-3"]]
    assert (own["D"] == 1).all() and (own["self_originated"] == 1).all() and (own["src_agent"] == "planner").all()
    assert (ex.loc[["planner", "researcher-4"], "self_originated"] == 0).all()   # Poland 0.2 / 0.1 posted by others first
    e = chat.read("edges_merged")
    assert sorted(zip(*e.loc[e["kind"] == "exposure", ["src", "dst"]].T.values)) == [("critic", "researcher-4"),
                                                                                      ("researcher-1", "planner")]
    prof = chat.read("agent_profiles_merged").set_index("label")
    assert prof.loc["planner", "answers_supplied"] == 0 and prof.loc["researcher-1", "answers_supplied"] == 1
    assert prof.loc["researcher-1", "n_exposed"] == 1   # D is kept


def test_report_counts_them_apart(chat):
    s = json.loads((chat.path / "summary.json").read_text())
    a1, a5 = s["A1"], s["A5"]
    assert (a1["n_independent"], a1["n_self_originated"], a1["n_independent_at_most"]) == (0, 3, 3)
    assert ("at most 3 were independent lookups (0 with nothing public before the report; 3 more were first posted "
            "by the agent itself before any other copy)") in s["headline"][0]
    first = {r["rule"]: r for r in a5["attribution"] if r["answers"] == "all exposed"}["first source"]
    assert first["n"] == a5["n_exposure_edges"] == 2 and a5["n_self_originated"] == 3
    assert a5["origins"]["self_originated"] == {"merged": 3, "strict": 3}   # the mappings group the authors alike
    # the planner's credited source posted the value first: its origin, not a relay of the planner's later copy
    assert a5["origins"]["by_mapping"]["merged"]["first source"] == {
        "computed": 0, "no answer round observed": 1, "dissent": 0, "relayed": 0, origins.POSTED_FIRST: 1}
    assert "(3 more, first posted by the answering agent itself, are credited to nobody)" in s["headline"][4]
    md = (chat.path / "report.md").read_text()
    assert "**5 answers given by 5 agents; at most 3 independent lookups**: 0 with nothing" in md
    assert f"| {origins.SELF} | 3 | 3 |" in md and "3 more exposed answers were self-originated" in md
    assert f"| {origins.POSTED_FIRST} | 1 (50%) | 1 (50%) |" in md and "| relayed | 0 (0%) | 0 (0%) |" in md
    assert "| researcher-1 | 1 | 1 | posted before its own source 1 |" in md
    assert "No rule credits the 3 self-originated answers" in md


def test_trace_names_the_planner_copy_a_later_repeat(chat):
    (a,) = query.answer_lineage(chat, agent="researcher-1", item="Poland")
    assert a["self_originated"] == 1 and a["first_public_copy"]["agent"] == "planner"
    text = "\n".join(query.text_answers([a]))
    assert "posted first by this agent, at 2026-05-04 09:02:00: post m2 on research (self-originated" in text
    assert "a later repeat by another agent, before the report: planner, post m6" in text
    assert "own earlier post" not in text and "first public copy" not in text
    v = query.agent_view(chat, "researcher-2")
    assert v["answers"][0]["first_public_source"] is None
    assert "but this agent posted it first (self-originated: no source credited)" in "\n".join(query.text_agent(v))
    assert "first public source for" not in "\n".join(query.text_agent(query.agent_view(chat, "planner")))


def test_an_edge_for_a_self_originated_round_is_an_error(chat, tmp_path):
    out = checks.check_run(chat).set_index(["check", "table"])
    assert out["n_violations"].sum() == 0 and out.loc[("self_originated_no_edge", "edges_merged"), "severity"] == "error"
    import shutil
    shutil.copytree(chat.path, tmp_path / "run")
    run = RunDir(tmp_path / "run")
    ex = run.read("exposures_merged")
    own = ex[ex["self_originated"] == 1]
    run.write("edges_merged", pd.concat([run.read("edges_merged"),
                                         graph.exposure_edges(own.assign(self_originated=0))], ignore_index=True))
    bad = checks.check_run(run).set_index(["check", "table"])
    assert bad.loc[("self_originated_no_edge", "edges_merged"), "example_ids"] == sorted(own["claim_id"])
    assert bad.loc[("exposure_edges", "edges_merged"), "n_violations"] == 0


def test_commands_on_exposure_tables_from_before_self_originated_answers(chat, tmp_path):
    """``report``, ``trace`` and ``agent`` stop alike, naming the stages to rerun, instead of failing on the column."""
    import shutil
    from swarmprov import cli
    shutil.copytree(chat.path, tmp_path / "run")
    old = RunDir(tmp_path / "run")
    for m in ("merged", "strict"):
        old.write(f"exposures_{m}", old.read(f"exposures_{m}").drop(columns=["self_originated"]))
    for argv in (["report"], ["trace", "--agent", "researcher-1", "--item", "Poland"], ["agent", "researcher-1"]):
        with pytest.raises(SystemExit, match=r"predate this version \(no self_originated\); rerun `swarmprov expose"):
            cli.main([argv[0], str(old.path), *argv[1:]])


def test_a5_with_only_self_originated_answers_says_so(tmp_path):
    """a posts the value, b repeats it, a reports: the one exposed answer credits nobody, and A5 says why."""
    path = tmp_path / "chat.jsonl"
    path.write_text("".join(json.dumps({"id": f"m{i}", "timestamp": (T0 + pd.Timedelta(minutes=i)).isoformat(),
                                        "author": a, "channel": "research", "text": t}) + "\n"
                            for i, (a, t) in enumerate((("a", "Found it: the World Bank page shows Poland 0.2 for 2023."),
                                                        ("b", "Poland 0.2, per the World Bank page."),
                                                        ("a", "I answered Poland 0.2.")))))
    (tmp_path / "cfg.json").write_text(json.dumps({"bootstrap_reps": 0}))
    run = pipeline.run_all(path, tmp_path / "run", adapter="chat", config=str(tmp_path / "cfg.json"))
    why = ("no answer had a first public source (the 1 exposed answer was self-originated: its agent posted the value "
           "before any other agent did, so no source is credited), so there is no hub concentration")
    assert why in json.loads((run.path / "summary.json").read_text())["headline"][4]
    assert "No answer had a first public source here (the 1 exposed answer was self-originated" in \
        (run.path / "report.md").read_text()


def test_a_synthetic_case_credits_only_the_later_poster():
    """A posts the value at 0 and B at 10; A reports at 20 and B at 30.  A's answer is self-originated (B's copy is a
    later repeat); B's is credited to A, whose own answer is self-originated: A posted the value before its own source, not relayed it."""
    rounds = pd.DataFrame([{"claim_id": c, "report_event_id": f"r{c}", "agent": a, "family": "f", "episode": 1,
                            "item": "x", "value_norm": "7", "t_report": T0 + pd.Timedelta(minutes=m),
                            "latency_class": None, "wrong_flag": False, "correct_flag": False}
                           for c, a, m in (("a", "A", 20), ("b", "B", 30))])
    mentions = pd.DataFrame([{"event_id": e, "ts": T0 + pd.Timedelta(minutes=m), "agent_merged": a, "channel": "p",
                              "family": "f", "item": "x", "value_norm": "7", "value_key": "7"}
                             for e, a, m in (("e0", "A", 0), ("e1", "B", 10))])
    ex = exposure.build(rounds, mentions, {("f", "x"): "7"}, "agent_merged")
    assert ex["D"].tolist() == [1, 1] and ex["self_originated"].tolist() == [1, 0]
    assert exposure.credited(ex).tolist() == [False, True]
    assert graph.exposure_edges(ex)[["src", "dst"]].values.tolist() == [["A", "B"]]
    edges = pd.DataFrame(columns=["src", "dst", "family", "item", "value_norm", "kind", "src_event_id"])
    att = structure.attribution(ex, edges, pd.DataFrame(columns=["event_id", "ts"])).set_index(["rule", "answers"])
    assert att.loc[("first source", "all exposed"), "n"] == att.loc[("latest source", "all exposed"), "n"] == 1
    root = origins.trace_root(ex)
    assert root[["agent", "source", "origin", "root_origin"]].values.tolist() == [["B", "A", origins.POSTED_FIRST,
                                                                                  origins.POSTED_FIRST]]
    a1 = {"n_answers": 2, "n_agents": 2, "share_exposed": 1.0, "n_independent": 0, "n_self_originated": 1,
          "n_independent_at_most": 1}
    assert report.key_findings({"profile": {"n_events": 4}, "A1": a1})[1].count(
        "(0 with nothing public before the report; 1 more was first posted by the agent itself before any other copy)") == 1
    # D_visible states its own bound, the self-originated answers it does not already count added
    a1 |= {"n_independent_visible": 1, "n_independent_at_most_visible": 1} | report._visibility_counts(ex)
    assert a1["n_independent_at_most_visible"] == 1   # A's copy by B is still visible: D_visible = 1, no lookup
    assert ("1 more was first posted by the agent itself before any other copy; at most 1 under `D_visible`, which "
            "counts a copy deleted before the report as never public)") in report.key_findings(
        {"profile": {"n_events": 4}, "A1": a1})[1]
    gone = ex.assign(D_visible=[0, 1])   # had B deleted its copy before A's report, D_visible would count A already
    assert report._visibility_counts(gone)["n_independent_at_most_visible"] == 1
