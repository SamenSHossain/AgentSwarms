"""Reply edges, agent profiles, the attributed GraphML/JSON export and the supply figure."""

import json
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from swarmprov import cli, graph, pipeline, query
from swarmprov.analysis import agents, structure
from swarmprov.schema import RunDir

T0 = pd.Timestamp("2026-07-07T10:00:00Z")


def _event(event_id, author, s, parent=None, kind="message"):
    return {"event_id": event_id, "ts": T0 + pd.Timedelta(seconds=s), "author_raw": author, "parent_id": parent,
            "source_kind": kind, "family": "f"}


def test_reply_edges_link_the_parent_agent_to_the_replier():
    ev = pd.DataFrame([_event("m1", "A", 0), _event("m2", "B", 30, "m1"), _event("m3", "C", 90, "m2"),
                       _event("m4", "C", 95, "m3"),                       # a self-reply
                       _event("m5", "B", 99, "gone"),                     # a parent outside the transcript
                       _event("t1", "task", 1, kind="task"), _event("m6", "B", 100, "t1"),   # the task is no agent's post
                       _event("r1", "A", 101, "m3", kind="tool_result")])  # a tool result replies to nobody
    ag = pd.DataFrame({"author_raw": ["A", "B", "C", "task"], "agent_merged": ["a", "b", "c", "task"]})
    e, unresolved = graph.reply_edges(ev, ag, "agent_merged")
    assert list(zip(e["src"], e["dst"], e["event_id"], e["src_event_id"], e["dt_s"])) == [
        ("a", "b", "m2", "m1", 30.0), ("b", "c", "m3", "m2", 60.0)]
    assert (e["kind"] == "reply").all() and unresolved == 2


def test_to_networkx_counts_kinds_and_keeps_cross_page_hops_out_of_the_weight():
    ev = pd.DataFrame({"event_id": ["e1", "e2", "e3", "e4"], "ts": [T0 + pd.Timedelta(minutes=m) for m in (0, 5, 9, 20)]})
    rows = [("A", "B", "relay", "x", "1", 10.0, "e1", True), ("A", "B", "relay", "y", "2", 30.0, "e2", False),
            ("A", "B", "relay_xchannel", "z", "3", 50.0, "e3", False), ("A", "B", "exposure", "x", "1", 20.0, "e2", None),
            ("A", "B", "reply", None, None, np.nan, "e4", None), ("C", "B", "relay_xchannel", "x", "1", 5.0, "e1", False)]
    edges = pd.DataFrame(rows, columns=["src", "dst", "kind", "item", "value_norm", "dt_s", "event_id", "within_clock_res"])
    nodes = pd.DataFrame({"n_answers": [3, 1], "median_head_h": [np.nan, 2.5], "families": ["f, g", None]},
                         index=["A", "B"])
    G = graph.to_networkx(edges.assign(family="f"), events=ev, nodes=nodes)
    ab = G.edges["A", "B"]
    assert (ab["weight"], ab["n_relay"], ab["n_xchannel"], ab["n_exposure"], ab["n_reply"], ab["n_citation"]) == (4, 2, 1, 1, 1, 0)
    assert ab["n_facts"] == 3 and ab["n_within_clock_res"] == 1 and ab["median_dt_s"] == 25.0
    assert (ab["first_ts"], ab["last_ts"]) == (T0.isoformat(), (T0 + pd.Timedelta(minutes=20)).isoformat())
    assert ab["kinds"] == "exposure,relay,relay_xchannel,reply"
    assert G.edges["C", "B"]["weight"] == 0 and G.edges["C", "B"]["n_xchannel"] == 1
    assert G.nodes["A"] == {"n_answers": 3, "median_head_h": -1.0, "families": "f, g"}
    assert G.nodes["C"] == {"n_answers": -1, "median_head_h": -1.0, "families": ""}   # no profile row
    assert G.nodes["B"]["families"] == ""


@pytest.fixture(scope="module")
def thread(tmp_path_factory):
    """Two rounds in which two agents reply to the first message, a self-reply and a reply to a missing message."""
    d = tmp_path_factory.mktemp("thread")
    msgs = [{"id": f"m{r}{k}", "author": a, "timestamp": (T0 + pd.Timedelta(minutes=30 * r + 5 * k)).isoformat(),
             "text": f"R{r} CONFIRMED: {item} arrived; answered {val} at +{3 + k}s."} | ({"reply_to": f"m{r}0"} if k else {})
            for r, (item, val) in enumerate([("Utah", "73.74"), ("Idaho", "41.20")], start=1)
            for k, a in enumerate(["Planner", "Coder", "Critic"])]
    msgs += [{"id": "m30", "author": "Critic", "timestamp": (T0 + pd.Timedelta(hours=2)).isoformat(), "text": "noted",
              "reply_to": "m22"},
             {"id": "m31", "author": "Coder", "timestamp": (T0 + pd.Timedelta(hours=3)).isoformat(), "text": "late",
              "reply_to": "m99"}]
    (d / "chat.jsonl").write_text("\n".join(json.dumps(m) for m in msgs))
    (d / "cfg.json").write_text(json.dumps({"families": {"states": "CONFIRMED"}}))
    return pipeline.run_all(d / "chat.jsonl", d / "run", adapter="chat", config=str(d / "cfg.json"))


def test_reply_edges_reach_the_report_and_the_graphml(thread):
    e = thread.read("edges_merged")
    rep = e[e["kind"] == "reply"]
    assert sorted(zip(rep["src"], rep["dst"])) == [("planner", "coder")] * 2 + [("planner", "critic")] * 2
    assert e["event_id"].isin(thread.read("events")["event_id"]).all()
    text = (thread.path / "report.md").read_text()
    assert ("and 4 replies; a reply is an interaction, not a knowledge transfer, so replies count toward the brokers "
            "below but never toward supply or relays (1 more reply names a parent that is not an agent's post here and "
            "is left out)") in text
    G = nx.read_graphml(thread.path / "provenance_merged.graphml")
    pc = G.edges["planner", "critic"]
    assert (pc["n_reply"], pc["n_exposure"], pc["n_relay"], pc["weight"]) == (2, 2, 0, 4)
    p = thread.read("agent_profiles_merged").set_index("label")
    assert p.loc["planner", "answers_supplied"] == 4 and p.loc["planner", "betweenness"] == 0.0


def test_reply_edges_need_threading(thread, tmp_path):
    """A source without threading keeps its parent ids (wiki revisions, Village summaries) out of the graph."""
    shutil.copytree(thread.path, tmp_path / "run")
    run = RunDir(tmp_path / "run")
    prof_path = run.path / "profile.json"
    prof = json.loads(prof_path.read_text())
    prof["capabilities"]["has_threading"] = False
    prof_path.write_text(json.dumps(prof))
    pipeline.build_graph(run)
    e = run.read("edges_merged")
    assert "reply" not in set(e["kind"]) and len(e) == (thread.read("edges_merged")["kind"] != "reply").sum()


def test_profiles_add_up_to_the_exposure_table(synth_run):
    run, truth = synth_run
    for mapping in ("merged", "strict"):
        p, ex = run.read(f"agent_profiles_{mapping}"), run.read(f"exposures_{mapping}")
        e = run.read(f"edges_{mapping}")
        assert len(p) == run.read("agents")[agents.AGENT_COLS[mapping]].nunique() == ex["agent"].nunique() == 60
        assert p["n_answers"].sum() == len(ex) and p["n_no_prior_copy"].sum() == (ex["D"] == 0).sum()
        assert p["n_exposed"].sum() == ex["D"].sum() and p["n_exposed_60m"].sum() == ex["D_w60m"].sum()
        assert p["n_self_prepared"].sum() == ex["self_prepared"].sum()
        assert p["answers_supplied"].sum() == (e["kind"] == "exposure").sum()
        assert p["relay_out"].sum() == p["relay_in"].sum() == (e["kind"] == "relay").sum()
        three = p["n_answers"] >= agents.MIN_ROUNDS
        assert p.loc[three, "exposed_k_of_n"].notna().all() and p.loc[~three, "exposed_k_of_n"].isna().all()
    p = run.read("agent_profiles_merged").set_index("label")
    r5 = structure.analyze(run.read("edges_merged"), run.read("chains_merged"))
    assert np.allclose(p.loc[r5["top_brokers"].index, "betweenness"], r5["top_brokers"].values)   # A5's brokers
    assert (p["family"] == [i.split("|")[1] for i in p.index]).all()
    # no earlier public copy is a ceiling on independence: copying needs a public copy, so D == 0 never hides a copy
    label = run.read("agents").set_index("author_raw")["agent_merged"]
    computed = truth[~truth["copied"]].groupby(truth["agent"].map(label)).size().reindex(p.index, fill_value=0)
    assert (p["n_no_prior_copy"] <= computed).all() and p["n_no_prior_copy"].sum() > 0


def test_profiles_leave_hours_empty_without_a_wall_clock(synth_run):
    run, _ = synth_run
    tables = [run.read(n) for n in ("agents", "events", "tags", "exposures_merged", "edges_merged", "chains_merged")]
    p = agents.profile(*tables, wall_clock=False)
    assert p["n_exposed_60m"].isna().all() and p["median_head_h"].isna().all() and p["n_exposed"].sum() > 0


def test_graphml_and_json_round_trip(synth_run):
    run, _ = synth_run
    for mapping in ("merged", "strict"):
        G = nx.read_graphml(run.path / f"provenance_{mapping}.graphml")
        data = json.loads((run.path / f"provenance_{mapping}.json").read_text())
        assert len(data["nodes"]) == len(G) == 60
        for items in ([d for _, d in G.nodes(data=True)], [d for *_, d in G.edges(data=True)]):
            assert all(v is not None for d in items for v in d.values())
            types = {}
            for d in items:
                for k, v in d.items():
                    types.setdefault(k, set()).add(type(v))
            assert all(len(t) == 1 for t in types.values()), types
        assert all({"family", "cohort", "n_answers", "first_seen", "answers_supplied", "betweenness"} <= set(d)
                   for _, d in G.nodes(data=True))
        e = run.read(f"edges_{mapping}")
        assert sum(d["answers_supplied"] for _, d in G.nodes(data=True)) == (e["kind"] == "exposure").sum()
        assert sum(d["n_exposure"] for *_, d in G.edges(data=True)) == (e["kind"] == "exposure").sum()
        assert sum(d["n_xchannel"] for *_, d in G.edges(data=True)) == (e["kind"] == "relay_xchannel").sum()


def test_agent_command(synth_run, tmp_path, capsys):
    run, _ = synth_run
    out = tmp_path / "agent.json"
    assert cli.main(["agent", str(run.path), "synthbetaagentoct07", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "agent Oct07|beta-task: 1 author string (SynthBetaAgentOct07)" in text and "answers (5):" in text
    v = json.loads(out.read_text())["result"]
    p = run.read("agent_profiles_merged").set_index("label").loc["Oct07|beta-task"]
    assert v["profile"]["answers_supplied"] == p["answers_supplied"] == sum(r["edges"] for r in v["supplied"]["exposure"])
    assert len(v["answers"]) == p["n_answers"] and sum(a["D"] == 0 for a in v["answers"]) == p["n_no_prior_copy"]
    assert cli.main(["agent", str(run.path), "beta"]) == 1 and "agents match 'beta'" in capsys.readouterr().out
    assert query.match_agents(run, "SYNTHBETAAGENTOCT07") == ["Oct07|beta-task"]


def test_report_agents_section_and_supply_figure(synth_run, monkeypatch, tmp_path):
    run, _ = synth_run
    s = json.loads((run.path / "summary.json").read_text())
    text = (run.path / "report.md").read_text()
    a, p = s["agents"], run.read("agent_profiles_merged")
    assert a["n_answering"] == 60 and a["n_answers"] == s["A1"]["n_answers"] and a["n_sources"] == s["A5"]["exposure_sources"]
    assert a["n_all_no_prior_copy"] == ((p["n_answers"] > 0) & (p["n_exposed"] == 0)).sum()
    assert f"**60 of the 60 agents with a post gave answers** ({s['A1']['n_answers']} agent-rounds" in text
    assert text.index("](figures/a5_structure.png)") < text.index("](figures/a7_supply_network.png)") < \
        text.index("## Agents") < text.index("## A6")
    assert a["top_suppliers"][0]["answers_supplied"] == p["answers_supplied"].max()
    titles = []
    monkeypatch.setattr(structure.plotting, "save", lambda fig, path: titles.append(fig.axes[0].get_title("left")) or path)
    r5 = structure.analyze(run.read("edges_merged"), run.read("chains_merged"))
    structure.supply_figure(run.read("edges_merged"), p.set_index("label"), r5, tmp_path / "a7.png")
    assert f"supplied {s['A5']['top10_share_exposure']:.0%} of" in titles[0] and "first-source rule" in titles[0]


def test_supply_layout_does_not_depend_on_the_hash_seed():
    """A subgraph view iterates a set of node names, so a seeded layout must fix the node order itself."""
    import subprocess
    import sys
    code = ("import networkx as nx; from swarmprov.analysis import structure as s; G = nx.Graph(); "
            "G.add_weighted_edges_from([(f'src{i % 3}', f'agent{i}', 1) for i in range(30)] + [('src0', 'src1', 1), "
            "('x', 'y', 1)]); p = s._packed_layout(G, {'src0', 'src1', 'src2', 'x'}); "
            "print(sorted((k, round(float(v[0]), 6), round(float(v[1]), 6)) for k, v in p.items()))")
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          env={**__import__("os").environ, "PYTHONHASHSEED": str(h)}).stdout for h in (1, 2)}
    assert len(out) == 1
