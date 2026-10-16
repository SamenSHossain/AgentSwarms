"""A5 attribution: concentration under each attribution rule, checked against synthetic swarms whose agents copy the
latest post, the first post, or nothing; and the origin of each copied answer's source."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from swarmprov import cli, exposure, report
from swarmprov.analysis import origins, structure
from swarmprov.schema import TABLES, RunDir

T0 = pd.Timestamp("2026-06-17T09:00:00Z")


def _t(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def _accuracy(run, truth, col):
    """Share of truly copied rounds whose ``col`` (or ``relay_parent``) names the agent whose post was copied."""
    label = run.read("agents").set_index("author_raw")["agent_merged"]
    cp = truth[truth["copied"]].assign(agent=lambda d: d["agent"].map(label), src=lambda d: d["src_agent"].map(label))
    ex = run.read("exposures_merged")
    ex = ex.assign(relay_parent=structure.relay_parents(ex, run.read("edges_merged"), run.read("events")))
    j = cp.merge(ex, on=["agent", "item"], suffixes=("_true", ""))
    assert len(j) == len(cp)
    src = cp["src"].value_counts()
    return float((j[col] == j["src"]).mean()), float(src.head(10).sum() / src.sum())


def _shares(run):
    a = structure.analyze(run.read("edges_merged"), run.read("chains_merged"), ex=run.read("exposures_merged"),
                          events=run.read("events"))
    return a["attribution"].set_index(["rule", "answers"])


def test_latest_carrier_is_stored_with_its_post():
    rounds = pd.DataFrame([{"claim_id": "c", "report_event_id": "r", "agent": "B", "family": "f", "episode": 1,
                            "item": "Hungary", "value_norm": "9.9", "t_report": _t(60),
                            "latency_class": None, "wrong_flag": False, "correct_flag": False}])
    m = [{"event_id": e, "ts": _t(s), "agent_merged": a, "channel": "p", "family": "f", "item": "Hungary",
          "value_norm": "9.90", "value_key": "9.9"} for e, a, s in
         [("e0", "A", 0), ("z9", "C", 30), ("a1", "D", 30), ("own", "B", 40), ("late", "E", 90)]]
    for rows in (m, m[::-1]):
        e = exposure.build(rounds, pd.DataFrame(rows), {}, "agent_merged").iloc[0]
        assert (e["src_agent"], e["src_latest"], e["t_latest"], e["latest_event_id"]) == ("A", "C", _t(30), "z9")
    assert {"src_latest", "t_latest", "latest_event_id"} <= set(TABLES["exposures"])


def test_match_key_falls_back_to_the_consensus():
    assert exposure.match_key("163139/166813", "1") == "163139"
    assert exposure.match_key(None, "9.9") == "9.9" and exposure.match_key(float("nan"), "9.9") == "9.9"
    assert exposure.match_key(None, float("nan")) is None


def test_relay_parent_join_uses_the_exposure_key():
    """A multi-number value joins on its first number, an unrestated answer on the consensus; a round whose post
    arrived by a cross-page hop is unmatched, and so is one whose parent posted at or after the report (as ``trace``
    and exposure have it)."""
    ex = pd.DataFrame([
        {"agent": "B", "family": "f", "item": "x", "value_norm": "163139/166813", "consensus_value": "1", "D": 1,
         "src_agent": "A", "src_latest": "A", "y_instant": 1, "t_report": _t(10)},
        {"agent": "C", "family": "f", "item": "y", "value_norm": None, "consensus_value": "7", "D": 1,
         "src_agent": "A", "src_latest": "B", "y_instant": 0, "t_report": _t(10)},
        {"agent": "D", "family": "f", "item": "y", "value_norm": "7", "consensus_value": "7", "D": 1,
         "src_agent": "A", "src_latest": "C", "y_instant": 0, "t_report": _t(10)},
        {"agent": "E", "family": "f", "item": "y", "value_norm": "8", "consensus_value": "7", "D": 0,
         "src_agent": None, "src_latest": None, "y_instant": 0, "t_report": _t(10)},
        {"agent": "F", "family": "f", "item": "y", "value_norm": "7", "consensus_value": "7", "D": 1,
         "src_agent": "A", "src_latest": "C", "y_instant": 0, "t_report": _t(10)},
        {"agent": "G", "family": "f", "item": "y", "value_norm": "7", "consensus_value": "7", "D": 1,
         "src_agent": "A", "src_latest": "C", "y_instant": 0, "t_report": _t(10)}]).assign(self_originated=0)
    edges = pd.DataFrame([
        {"src": "A", "dst": "B", "family": "f", "item": "x", "value_norm": "163139", "kind": "relay", "src_event_id": "a"},
        {"src": "B", "dst": "C", "family": "f", "item": "y", "value_norm": "7", "kind": "relay", "src_event_id": "b"},
        {"src": "A", "dst": "D", "family": "f", "item": "y", "value_norm": "7", "kind": "relay_xchannel", "src_event_id": "a"},
        {"src": "C", "dst": "F", "family": "f", "item": "y", "value_norm": "7", "kind": "relay", "src_event_id": "now"},
        {"src": "C", "dst": "G", "family": "f", "item": "y", "value_norm": "7", "kind": "relay", "src_event_id": "late"},
        {"src": "A", "dst": "B", "family": "f", "item": "x", "value_norm": "163139", "kind": "exposure", "src_event_id": "a"}])
    events = pd.DataFrame({"event_id": ["a", "b", "now", "late"], "ts": [_t(1), _t(2), _t(10), _t(11)]})
    assert structure.relay_parents(ex, edges, events).tolist() == ["A", "B", None, None, None, None]
    t = structure.attribution(ex, edges, events, top_k=1).set_index(["rule", "answers"])
    assert t.loc[("same-page relay parent", "all exposed"), ["n", "sources", "unmatched"]].tolist() == [2, 2, 3]
    assert t.loc[("first source", "all exposed"), ["n", "sources", "top1_share"]].tolist() == [5, 1, 1.0]
    assert t.loc[("latest source", "all exposed"), "top1_share"] == pytest.approx(3 / 5)
    assert t.loc[("latest source", "instant only"), ["n", "sources"]].tolist() == [1, 1]


def test_concentration():
    c = structure.concentration(pd.Series(["A"] * 6 + ["B"] * 2 + ["C"] * 2 + [None]), top_k=1)
    assert (c["n"], c["sources"], c["top1_share"]) == (10, 3, 0.6)
    assert c["gini"] == pytest.approx((2 * np.arange(1, 4) - 4).dot([2, 2, 6]) / 30)
    empty = structure.concentration(pd.Series([], dtype=object))
    assert empty["n"] == 0 and np.isnan(empty["top10_share"]) and np.isnan(empty["gini"])


def test_item_null_and_single_source_items():
    ex = pd.DataFrame({"D": [1, 1, 1, 1, 0], "family": "f", "item": ["x", "x", "x", "y", "y"],
                       "src_agent": ["A", "A", "B", "C", None], "src_latest": ["A", "B", "B", "C", None],
                       "agent": list("PQRST"), "value_norm": "1", "consensus_value": "1", "y_instant": 0,
                       "self_originated": 0})
    edges = pd.DataFrame({"src": ["A", "A", "B", "C"], "dst": list("PQRS"), "kind": "exposure", "family": "f",
                          "item": ["x", "x", "x", "y"], "value_norm": "1", "src_event_id": None})
    r = structure.analyze(edges, pd.DataFrame(columns=["family", "n_agents", "max_depth"]), ex=ex.assign(t_report=T0),
                          events=pd.DataFrame(columns=["event_id", "ts"]))
    assert (r["items_exposed"], r["items_one_source"]) == (2, 1)
    assert r["gini_item_exposures"] == pytest.approx(structure.gini([3, 1]))


def test_sources_with_equal_answers_are_listed_by_name():
    """Most answers first, ties by name, however many there are (an unstable sort reorders long runs of ties)."""
    names = [f"s{i:02d}" for i in range(40)]
    og = pd.DataFrame({"source": [n for n in names for _ in range(2)] + ["top"] * 3, "family": "f", "item": "x",
                       "origin": "computed"})
    assert origins.by_source(og, pd.DataFrame({"agent": names})).index.tolist() == ["top", *names]


def _ex(rows):
    """Exposure rows, ``self_originated`` from their times as exposure sets it."""
    cols = {"claim_id": None, "family": "f", "item": "x", "value_norm": "9.9", "consensus_value": "9.9", "D": 1,
            "src_latest": None, "t_latest": None, "t_self_prepared": None}
    rows = [cols | {"claim_id": r["agent"]} | r for r in rows]
    return pd.DataFrame([r | {"self_originated": int(r["t_self_prepared"] is not None and r["t_public"] is not None
                                                     and r["t_self_prepared"] < r["t_public"])} for r in rows])


def test_trace_root_follows_a_three_hop_chain():
    """D copied from C, C from B, B from A, who computed it; E's source answered another value; F's source has
    no answer round; G's source H had posted the value before its own first copy appeared, so H's own answer is
    self-originated and credits nobody."""
    ex = _ex([
        {"agent": "A", "D": 0, "src_agent": None, "t_public": None},
        {"agent": "B", "src_agent": "A", "t_public": _t(1)},
        {"agent": "C", "src_agent": "B", "t_public": _t(2)},
        {"agent": "D", "src_agent": "C", "t_public": _t(3)},
        {"agent": "E", "src_agent": "Y", "t_public": _t(3)},
        {"agent": "Y", "D": 0, "src_agent": None, "t_public": None, "value_norm": "9.91"},
        {"agent": "F", "src_agent": "Z", "t_public": _t(3)},
        {"agent": "G", "src_agent": "H", "t_public": _t(5)},
        {"agent": "H", "src_agent": "G", "t_public": _t(4), "t_self_prepared": _t(2)}])
    r = origins.trace_root(ex).set_index("agent")
    assert r.loc["D", ["origin", "root", "root_origin", "depth"]].tolist() == ["relayed", "A", "computed", 3]
    assert r.loc["B", ["origin", "root", "depth"]].tolist() == ["computed", "A", 1]
    assert r.loc["E", "origin"] == "dissent" and r.loc["F", "origin"] == "no answer round observed"
    # H is G's origin itself, not a relay: A1 and trace call H's own answer self-originated
    assert r.loc["G", ["origin", "root", "root_origin", "depth"]].tolist() == [origins.POSTED_FIRST, "H",
                                                                               origins.POSTED_FIRST, 1]
    assert "H" not in r.index
    assert set(r["origin"]) <= set(origins.LABELS)


def test_trace_root_latest_rule_stops_where_the_source_posted():
    """Under the latest rule the walk stops at a source whose own post came before its latest carrier: POSTED_FIRST
    only if no other agent had posted the value before it (P), else CUT (B, exposed by A first); a chain leading back
    to the answering agent (H) is judged by the same test.  Under the first rule B's source A computed the value."""
    ex = _ex([
        {"agent": "A", "D": 0, "src_agent": None, "t_public": None},
        {"agent": "B", "src_agent": "A", "t_public": _t(1), "t_self_prepared": _t(2), "src_latest": "C", "t_latest": _t(3)},
        {"agent": "C", "src_agent": "A", "t_public": _t(1), "src_latest": "A", "t_latest": _t(1)},
        {"agent": "D", "src_agent": "A", "t_public": _t(1), "src_latest": "B", "t_latest": _t(4)},
        {"agent": "P", "src_agent": "C", "t_public": _t(2), "t_self_prepared": _t(2), "src_latest": "C", "t_latest": _t(3),
         "item": "y"},
        {"agent": "C", "src_agent": "Q", "t_public": _t(1), "src_latest": "Q", "t_latest": _t(1), "item": "y"},
        {"agent": "Q", "D": 0, "src_agent": None, "t_public": None, "item": "y"},
        {"agent": "E", "src_agent": "Q", "t_public": _t(1), "src_latest": "P", "t_latest": _t(5), "item": "y"},
        {"agent": "H", "src_agent": "A", "t_public": _t(1), "t_self_prepared": _t(2), "src_latest": "J", "t_latest": _t(4)},
        {"agent": "J", "src_agent": "A", "t_public": _t(1), "src_latest": "H", "t_latest": _t(2)}])
    latest = origins.trace_root(ex, "src_latest").set_index(["agent", "item"])
    assert latest.loc[("D", "x"), ["source", "root", "root_origin", "depth"]].tolist() == ["B", "B", origins.CUT, 1]
    assert latest.loc[("E", "y"), ["source", "root", "root_origin", "depth"]].tolist() == ["P", "P", origins.POSTED_FIRST, 1]
    assert latest.loc[("H", "x"), ["root", "root_origin", "depth"]].tolist() == ["H", origins.CUT, 2]
    first = origins.trace_root(ex).set_index(["agent", "item"])
    assert first.loc[("D", "x"), ["root", "root_origin"]].tolist() == ["A", "computed"]
    assert origins.CUT not in set(first["root_origin"])
    text = "\n".join(report._origins_block({"merged": ex}, "merged", {"A5": {}})[0])
    assert f"*{origins.POSTED_FIRST.capitalize()}* = the agent had posted the value no later than any other" in text
    assert f". *{origins.CUT.capitalize()}* = the agent was exposed before its own post" in text
    assert f"| {origins.CUT} | 0 | 3 |" in text   # D (at B), H (loop) and J (at H)


def test_latest_rule_finds_the_copied_post(synth_run):
    """Copiers take the latest post: the latest source names it (0.98-1.0 on seeds 0-7), the first source does not
    (0.22-0.28), and on instant answers the latest rule's top-10 share equals the true one (within 0.01) while
    first-source credit inflates it to 0.64-0.76 against a true 0.26-0.28."""
    run, truth = synth_run
    acc, true_top = _accuracy(run, truth, "src_latest")
    assert acc >= 0.95 and _accuracy(run, truth, "src_agent")[0] < 0.4
    assert _accuracy(run, truth, "relay_parent")[0] >= 0.95   # the same-page relay parent: 0.98-1.0 on seeds 0-7
    a = _shares(run)
    assert abs(a.loc[("latest source", "instant only"), "top10_share"] - true_top) < 0.05
    assert a.loc[("first source", "instant only"), "top10_share"] - true_top > 0.25


def test_first_rule_finds_the_copied_post(synth_first):
    """Copiers take the first post: now first-source credit is right (1.0 on seeds 0-7, the latest source 0.19-0.25)
    and matches the true top-10 share (0.71-0.81, exactly), while the latest rule gives 0.26-0.28."""
    run, truth = synth_first
    acc, true_top = _accuracy(run, truth, "src_agent")
    assert acc >= 0.95 and _accuracy(run, truth, "src_latest")[0] < 0.4
    assert _accuracy(run, truth, "relay_parent")[0] < 0.4   # 0.19-0.25 on seeds 0-7
    a = _shares(run)
    assert abs(a.loc[("first source", "instant only"), "top10_share"] - true_top) < 0.05
    assert true_top - a.loc[("latest source", "instant only"), "top10_share"] > 0.25


def test_no_copying_leaves_the_copy_signature_empty(synth_null):
    """Nobody copies, yet first-source credit still finds 'hubs' among the agents who answered first (top-10 share
    0.68-0.80 on seeds 0-7)."""
    no_copy = synth_null[0]
    a = _shares(no_copy)
    assert (a.xs("instant only", level="answers")["n"] == 0).all()
    assert a.loc[("first source", "all exposed"), "n"] > 100 and a.loc[("first source", "all exposed"), "top10_share"] > 0.5
    text = (no_copy.path / "report.md").read_text()
    a5 = text[text.index("## A5."):text.index("## A6.")]
    assert "copy-signature subset is empty" in a5 and "| instant only |" not in a5 and "on instant answers" not in a5


def test_every_synthetic_source_answered(synth_run):
    """Synthetic agents post only answer reports, so every source has an answer round; the first carrier of a value
    reported it first, so it computed it; latest carriers are often copiers themselves."""
    run, _ = synth_run
    ex = run.read("exposures_merged")
    first, latest = origins.trace_root(ex), origins.trace_root(ex, "src_latest")
    assert len(first) == int(ex["D"].sum()) and (first["origin"] == "computed").all() and (first["depth"] == 1).all()
    assert (latest["origin"] != "no answer round observed").all() and (latest["origin"] == "relayed").mean() > 0.5
    assert (latest["root_origin"] == "computed").all() and latest["depth"].max() > 2


def test_report_shows_the_attribution_and_origins(synth_run):
    run, _ = synth_run
    text = (run.path / "report.md").read_text()
    a5 = text[text.index("## A5."):text.index("## A6.")]
    for s in ("**Hub concentration by attribution rule.**", "**Where did copied answers come from?**",
              "| same-page relay parent | instant only |", "| src | answers | items | origin |",
              "That share depends on who is credited", "who first posted the value on an item",
              "| root | first source | latest source |", f"| {origins.CUT} | 0 | 0 |"):
        assert s in a5, s
    s = json.loads((run.path / "summary.json").read_text())["A5"]
    o = s["origins"]
    # the strict mapping groups the seed-2 authors the same way (test_identity): its counts are the merged ones
    assert len(s["attribution"]) == 6 and list(o["by_mapping"]) == ["merged", "strict"] and o["mapping"] == "merged"
    assert o["by_mapping"]["strict"] == o["by_mapping"]["merged"] and o["self_originated"] == {"merged": 0, "strict": 0}
    assert o["by_mapping"]["merged"]["first source"]["no answer round observed"] == 0
    assert sum(o["by_mapping"]["merged"]["first source"].values()) == s["n_exposure_edges"]
    assert set(o["roots"]["latest source"]) == set(origins.LABELS[:3]) | {origins.POSTED_FIRST, origins.CUT}
    assert 0 < s["gini_item_exposures"] < 1 and s["items_one_source"] <= s["items_exposed"]


def test_report_on_a_run_that_predates_the_latest_source(synth_run, tmp_path):
    run, _ = synth_run
    shutil.copytree(run.path, tmp_path / "run")
    old = RunDir(tmp_path / "run")
    old.write("exposures_merged", old.read("exposures_merged").drop(columns=["src_latest", "t_latest", "latest_event_id"]))
    with pytest.raises(SystemExit, match=r"\(no src_latest, t_latest, latest_event_id\); rerun `swarmprov expose"):
        cli.main(["report", str(old.path)])
