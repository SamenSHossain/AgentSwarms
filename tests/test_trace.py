"""swarmprov trace: stored source posts, edges that join to posts, and lineage checked against synthetic truth."""

import hashlib
import json

import pandas as pd

from swarmprov import cli, exposure, graph, query, synth
from swarmprov.claims import value_key

T0 = pd.Timestamp("2026-06-17T09:46:10Z")


def _mention(event_id, agent, s, channel="p", value="9.9"):
    return {"event_id": event_id, "ts": T0 + pd.Timedelta(seconds=s), "agent_merged": agent, "channel": channel,
            "family": "f", "item": "Hungary", "value_norm": value, "value_key": value_key(value)}


def _agreement(run, truth):
    """Share of copied rounds whose inferred relay parent / first public copy is the true source."""
    label = run.read("agents").set_index("author_raw")["agent_merged"]
    cp = truth[truth["copied"]]
    found = {(a["agent"], a["item"]): a for a in query.answer_lineage(run)}
    rows = [(found[(label[r.agent], r.item)], label[r.src_agent]) for r in cp.itertuples() if (label[r.agent], r.item) in found]
    assert len(rows) == len(cp)
    parent = sum(bool(a["relay_parent"]) and a["relay_parent"]["agent"] == src for a, src in rows) / len(rows)
    first = sum(bool(a["first_public_copy"]) and a["first_public_copy"]["agent"] == src for a, src in rows) / len(rows)
    return parent, first, sum(a["D"] for a, _ in rows) / len(rows)


def test_same_second_sources_are_chosen_by_event_id():
    rounds = pd.DataFrame([{"claim_id": "c", "report_event_id": "r", "agent": "B", "family": "f", "episode": 1,
                            "item": "Hungary", "value_norm": "9.9", "t_report": T0 + pd.Timedelta(hours=2),
                            "latency_class": None, "wrong_flag": False, "correct_flag": False}])
    two = [_mention("b2", "A", 0), _mention("a1", "A", 0, channel="q"), _mention("own", "B", 60)]
    for rows in (two, two[::-1]):
        e = exposure.build(rounds, pd.DataFrame(rows), {("f", "Hungary"): "9.9"}, "agent_merged").iloc[0]
        assert (e["src_event_id"], e["src_channel"], e["src_visible_event_id"]) == ("a1", "q", "a1")
        assert e["report_event_id"] == "r" and e["self_event_id"] == "own" and e["src_agent"] == "A"


def test_relay_edges_name_the_source_post():
    m = pd.DataFrame([_mention("e0", "A", 0), _mention("e1", "A", 10), _mention("e2", "B", 30),
                      _mention("e3", "C", 60, channel="q")])
    e = graph.relay_edges(m, "agent_merged")[0].set_index("dst")
    assert (e.loc["B", "kind"], e.loc["B", "event_id"], e.loc["B", "src_event_id"]) == ("relay", "e2", "e1")
    assert (e.loc["C", "kind"], e.loc["C", "src_event_id"]) == ("relay_xchannel", "e0")   # the originator's first post


def test_exposure_edges_carry_the_report_post_and_value_key():
    exp = pd.DataFrame([{"D": 1, "src_agent": "A", "agent": "B", "family": "f", "item": "x", "value_norm": "163139/166813",
                         "consensus_value": "163139", "gap_s": 60.0, "claim_id": "c1", "report_event_id": "r1",
                         "src_event_id": "s1"}])
    e = graph.exposure_edges(exp).iloc[0]
    assert (e["event_id"], e["claim_id"], e["src_event_id"], e["value_norm"]) == ("r1", "c1", "s1", "163139")


def test_every_source_post_resolves_to_its_mention(synth_run):
    run, _ = synth_run
    ex, mn = run.read("exposures_merged"), run.read("mentions")
    d1 = ex[ex["D"] == 1].assign(value_key=[value_key(v) or c for v, c in zip(ex.loc[ex["D"] == 1, "value_norm"],
                                                                               ex.loc[ex["D"] == 1, "consensus_value"])])
    j = d1.merge(mn, left_on=["src_event_id", "family", "item", "value_key"],
                 right_on=["event_id", "family", "item", "value_key"], suffixes=("", "_m"))
    j = j[(j["agent_merged"] == j["src_agent"]) & (j["ts"] == j["t_public"])]
    assert len(d1) > 100 and set(j["claim_id"]) == set(d1["claim_id"])
    assert ex.loc[ex["D"] == 0, "src_event_id"].isna().all()
    cl = run.read("claims").set_index("claim_id")
    assert (cl.loc[ex["claim_id"], "event_id"].values == ex["report_event_id"].values).all()


def test_every_edge_joins_to_a_post(synth_run):
    run, _ = synth_run
    e, ev = run.read("edges_merged"), run.read("events")
    assert {"claim_id", "src_event_id"} <= set(e.columns) and {"relay", "exposure"} <= set(e["kind"])
    assert e["event_id"].isin(ev["event_id"]).all()
    assert e["src_event_id"].dropna().isin(ev["event_id"]).all() and e["src_event_id"].notna().all()
    x = e[e["kind"] == "exposure"]
    assert x["claim_id"].isin(run.read("claims")["claim_id"]).all() and e.loc[e["kind"] != "exposure", "claim_id"].isna().all()


def test_fact_lineage_matches_the_chains(synth_run):
    run, _ = synth_run
    ch = run.read("chains_merged")
    for c in ch[ch["n_agents"] >= 2].itertuples():
        [f] = query.fact_lineage(run, c.family, c.item, c.value_key)
        assert len(f["hops"]) == c.n_agents - 1 and max(h["depth"] for h in f["hops"]) == c.max_depth
        assert len({h["dst"] for h in f["hops"]}) == len(f["hops"]) and f["origin"]["agent"] == c.originator


def test_relay_parent_finds_the_copied_post(synth_run):
    """Copying agents take the latest post on the board: the inferred relay parent should name its author
    (>= 98% on seeds 0-7), the first public copy should not (<= 28%)."""
    parent, first, exposed = _agreement(*synth_run)
    assert parent >= 0.95 and first < 0.5 and exposed == 1


def test_first_public_copy_finds_the_copied_post(synth_first):
    """Copying agents take the first post: now the first public copy names it (100% on seeds 0-7)."""
    parent, first, exposed = _agreement(*synth_first)
    assert first >= 0.97 and parent < 0.5 and exposed == 1


def test_copy_from_latest_keeps_the_transcript(synth_run):
    run, truth = synth_run
    path = run.path.parent / "transcript.jsonl"   # seed 2, 60 agents: the hash of the transcript before copy_from existed
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "52a3220d7225fe6d0760096e3c3da7e5cbf718f06e6b4aeb4bf3cccec9f24722"
    assert truth.loc[truth["copied"], "src_agent"].notna().all() and truth.loc[~truth["copied"], "src_agent"].isna().all()
    _, other = synth.simulate(n_agents=60, copy_rate=0.7, seed=2, copy_from="random")
    assert (other["copied"] == truth["copied"]).all() and (other["latency_s"] == truth["latency_s"]).all()


def test_excerpt_is_one_safe_line():
    text = "intro " * 40 + "\x1b[31mHungary\t\n 9.90%  <script>x</script>" + " tail" * 40
    ex = query.excerpt(text, "hungary", width=60)
    assert ex.startswith("…") and ex.endswith("…") and "\x1b" not in ex and "\n" not in ex and "Hungary 9.90%" in ex
    assert len(ex) <= 62


def test_hop_flags_are_printed():
    hop = {"src": "A", "dst": "B", "kind": "relay_xchannel", "ts": T0, "dt_s": 1.0, "within_clock_res": True,
           "channel": "q", "event_id": "e1", "excerpt": "", "depth": 1}
    fact = {"family": "f", "item": "Hungary", "value": "9.9", "carriers": 2, "max_depth": 1, "span_s": 1.0,
            "n_relay": 0, "n_xchannel": 1, "n_within_clock_res": 1, "origin": {"agent": "A", "ts": T0}, "hops": [hop]}
    text = "\n".join(query.text_facts([fact]))
    assert "source unknown, attributed to originator" in text and "direction unreliable" in text


def test_agent_names_match_case_insensitively(synth_run):
    run, truth = synth_run
    name = truth["agent"].iloc[0]
    [label] = query.match_agents(run, name.lower())
    assert label == query.match_agents(run, label.upper())[0] and label.startswith(truth["cohort"].iloc[0])
    assert len(query.match_agents(run, "alpha")) > 10


def test_trace_cli_modes(synth_run, tmp_path, capsys):
    run, truth = synth_run
    ex = run.read("exposures_merged")
    a = ex[ex["D"] == 1].iloc[0]
    fact = f"{a['family']}|{a['item']}|{a['value_norm']}"
    modes = {"answer": ["--agent", a["agent"], "--item", a["item"]], "list": ["--list"], "fact": ["--fact", fact],
             "event": ["--event", a["report_event_id"]]}
    for mode, args in {**modes, "claim": ["--answer", a["claim_id"]]}.items():
        out = tmp_path / f"{mode}.json"
        assert cli.main(["trace", str(run.path), *args, "--json", str(out)]) == 0, mode
        res = json.loads(out.read_text())
        assert res["mapping"] == "merged" and res["mode"] == {"claim": "answer"}.get(mode, mode)
    printed = capsys.readouterr().out
    assert "mapping: merged" in printed and "first public copy:" in printed and "inferred relay parent:" in printed
    assert json.loads((tmp_path / "answer.json").read_text())["result"][0]["first_public_copy"]["event_id"] == a["src_event_id"]
    assert cli.main(["trace", str(run.path), "--fact", fact, "--png", str(tmp_path / "t.png")]) == 0
    assert (tmp_path / "t.png").stat().st_size > 0
    assert cli.main(["trace", str(run.path), "--agent", "alpha"]) == 1 and "agents match 'alpha'" in capsys.readouterr().out
