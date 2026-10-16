"""Sensitivity sweeps over the analyst-chosen windows: the exposure head start (A1), the relay window W (A6) and
the instant-answer threshold k (A1)."""

import json

import numpy as np
import pandas as pd
import pytest

from swarmprov import graph
from swarmprov.analysis import provenance, structure
from swarmprov.claims import public

T0 = pd.Timestamp("2026-06-16T08:00:00Z")


def _at(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def test_threshold_curve_on_a_hand_frame():
    """Head starts 30 min, 2 h, 25 h and one answer with nothing public; the second copy was deleted before its
    report, so under D_visible only the 30 min and 25 h ones count."""
    ex = pd.DataFrame({"family": ["f", "f", "g", "g"], "D": [1, 1, 1, 0], "gap_s": [1800, 7200, 90000, np.nan],
                       "t_report": [_at(600)] * 4, "D_visible": [1, 0, 1, 0],
                       "t_public_visible": [_at(570), pd.NaT, _at(600 - 1500), pd.NaT]})
    c = provenance.threshold_curve(ex, hours=(0, 1, 24, 48), min_n=2, visible=True)
    assert list(c.columns) == ["n", "0", "1 h", "24 h", "48 h"]
    assert list(c.loc["all answers"]) == [4, 0.75, 0.5, 0.25, 0]
    assert list(c.loc["all answers, still-visible copies"]) == [4, 0.5, 0.25, 0.25, 0]
    assert list(c.loc["f"]) == [2, 1, 0.5, 0, 0] and list(c.loc["g"]) == [2, 0.5, 0.5, 0.5, 0]
    assert list(provenance.threshold_curve(ex, min_n=3).index) == ["all answers"]   # families gated on n
    assert provenance.hour_label(1 / 6) == "10 min" and provenance.hour_label(0) == "0"


def test_threshold_curve_recovers_the_synthetic_head_starts(synth_run):
    """Each answer's true head start: its report time minus the first other agent's report of the same value."""
    run, truth = synth_run
    ex = run.read("exposures_merged")
    c = provenance.threshold_curve(ex)
    t = truth.assign(t_report=pd.to_datetime(truth["t_report"]))
    first = {k: g for k, g in t.groupby(["family", "item", "value"])}
    gap = []
    for r in t.itertuples():
        g = first[(r.family, r.item, r.value)]
        earlier = g.loc[(g["agent"] != r.agent) & (g["t_report"] < r.t_report), "t_report"]
        gap.append((r.t_report - earlier.min()).total_seconds() if len(earlier) else np.nan)
    gap = np.array(gap)
    want = [(gap >= h * 3600).mean() for h in provenance.HOURS]
    assert np.allclose(c.loc["all answers"].iloc[1:].to_numpy(dtype=float), want)
    assert c.loc["all answers", "1 h"] == ex["D_w60m"].mean() and c.loc["all answers", "10 min"] == ex["D_w10m"].mean()
    assert c.loc["all answers", "24 h"] < 0.1 < 0.5 < c.loc["all answers", "1 h"]   # boards a day apart: no 24 h head start
    assert set(c.index) == {"all answers", "alpha-task", "beta-task"}


def test_relay_window_none_leaves_the_graph_unchanged(synth_run):
    run, _ = synth_run
    pub = public(run.read("mentions"))
    e0, c0 = graph.relay_edges(pub, "agent_merged")
    for w in (None, 1e12):
        e, c = graph.relay_edges(pub, "agent_merged", max_gap_s=w)
        pd.testing.assert_frame_equal(e, e0)
        pd.testing.assert_frame_equal(c, c0)
    stored = run.read("edges_merged")
    stored = stored[stored["kind"].isin(["relay", "relay_xchannel"])]
    cols = ["src", "dst", "family", "item", "value_norm", "kind", "dt_s", "event_id", "src_event_id"]
    key = ["family", "item", "value_norm", "dst"]
    pd.testing.assert_frame_equal(e0[cols].sort_values(key).reset_index(drop=True),
                                  stored[cols].sort_values(key).reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(c0.sort_values(["family", "item", "value_key"]).reset_index(drop=True),
                                  run.read("chains_merged").sort_values(["family", "item", "value_key"])
                                  .reset_index(drop=True), check_dtype=False)


def test_relay_window_cuts_old_parents_into_roots():
    """A at 0, B at 10 min and C at 2 h on page p, D at 2 h 05 on p, E at 30 min on page q (cross-page)."""
    m = pd.DataFrame({"event_id": list("abcde"), "ts": [_at(0), _at(10), _at(120), _at(125), _at(30)],
                      "agent_merged": list("ABCDE"), "channel": ["p", "p", "p", "p", "q"],
                      "family": "f", "item": "Ohio", "value_key": "1.5"})
    e, c = graph.relay_edges(m, "agent_merged")
    assert list(zip(e["src"], e["dst"], e["kind"], e["dt_s"], e["event_id"], e["src_event_id"])) == [
        ("A", "B", "relay", 600, "b", "a"), ("A", "E", "relay_xchannel", 1800, "e", "a"),
        ("B", "C", "relay", 6600, "c", "b"), ("C", "D", "relay", 300, "d", "c")]   # the default, written out
    assert c["max_depth"].iloc[0] == 3 and c["n_agents"].iloc[0] == 5
    wide = graph.relay_edges(m, "agent_merged", max_gap_s=6600)          # every parent inside W: the same graph
    pd.testing.assert_frame_equal(wide[0], e)
    pd.testing.assert_frame_equal(wide[1], c)
    e, c = graph.relay_edges(m, "agent_merged", max_gap_s=3600)
    assert set(zip(e["src"], e["dst"], e["kind"])) == {("A", "B", "relay"), ("A", "E", "relay_xchannel"),
                                                       ("C", "D", "relay")}
    assert c["n_agents"].iloc[0] == 5 and c["max_depth"].iloc[0] == 1      # C is an unattributed root, D one hop below
    e, c = graph.relay_edges(m, "agent_merged", max_gap_s=0)
    assert e.empty and c["max_depth"].iloc[0] == 0 and c["n_agents"].iloc[0] == 5


def test_window_sensitivity_matches_reach_at_no_limit(synth_run):
    run, _ = synth_run
    ws = structure.window_sensitivity(public(run.read("mentions")), "agent_merged")
    r6 = structure.reach(run.read("chains_merged"))["task values"]
    none = ws[ws["W_s"].isna()].iloc[0]
    assert none["unattributed"] == 0 and none["mean_linked_repeaters"] == pytest.approx(r6["mean_repeaters"])
    assert none["mean_max_depth"] == pytest.approx(r6["mean_hops_repeated"])
    carriers = ws["relay"] + ws["relay_xchannel"] + ws["unattributed"]
    assert (carriers == carriers.iloc[-1]).all()                       # every fact keeps its carriers
    assert ws["unattributed"].is_monotonic_decreasing and ws["mean_max_depth"].is_monotonic_increasing
    assert ws["unattributed"].iloc[0] > 0


def test_instant_sweep_recovers_the_true_latencies(synth_run):
    """Copied answers say "instantly" (1 s, no number), computed ones "at +Ns" (5-40 s): the share at each k is
    the true share of latencies at most k."""
    run, truth = synth_run
    from swarmprov.claims import agent_rounds
    ex = run.read("exposures_merged")
    rounds = agent_rounds(run.read("claims"), "agent_merged")
    sw = provenance.instant_sweep(ex.assign(latency_s=ex["claim_id"].map(rounds.set_index("claim_id")["latency_s"])))
    assert sw["share_instant"] == pytest.approx([(truth["latency_s"] <= k).mean() for k in sw["ks"]])
    assert sw["share_instant"][1] == ex["y_instant"].mean()           # the configured 2 s gives the instant column
    assert sw["n_instant_by_wording"] == truth["copied"].sum() and sw["n_stated_latency"] == (~truth["copied"]).sum()
    words = pd.DataFrame({"latency_s": [np.nan, 1.0, 4.0, np.nan], "latency_class": ["instant", "instant", "delayed", None]})
    assert provenance.instant_sweep(words, ks=(1, 5))["share_instant"] == [0.5, 0.75]
    assert provenance.instant_sweep(words, ks=(1, 5), k_cfg=2)["ks"] == [1, 2, 5]   # the configured k is added


def test_instant_sweep_names_what_stays_fixed():
    """An LLM extractor's classes come without a number: "delayed" in words stays delayed.  "instantly at +6s" is
    instant by its wording but not by its number at k = 2 s, so the sweep there differs from the instant column."""
    from swarmprov.report import _instant_block
    llm = pd.DataFrame({"latency_s": [None] * 4, "latency_class": ["delayed", "delayed", "instant", None]})
    sw = provenance.instant_sweep(llm, k_cfg=2.0)
    assert (sw["n_instant_by_wording"], sw["n_delayed_by_wording"], sw["n_class_differs_at_configured"]) == (1, 2, 0)
    text = _instant_block(sw, {"A1": {}})[0]
    assert ("1 called instant in words with no number stay instant at every k, 2 called delayed in words with no "
            "number stay delayed, and the rest state no latency.") in text and "goes by the number" not in text
    rules = pd.DataFrame({"latency_s": [6.0, 1.0, 9.0], "latency_class": ["instant", "instant", "delayed"]})
    sw = provenance.instant_sweep(rules, k_cfg=2.0)
    assert sw["n_class_differs_at_configured"] == 1 and sw["share_instant"][sw["ks"].index(2)] == pytest.approx(1 / 3)
    summary = {"A1": {}}
    lines = _instant_block(sw, summary)
    assert "delayed in words" not in lines[0]
    assert "k = 2 s the sweep goes by the number and A1's *instant* column by the class, and 1 answer has" in lines[0]
    assert summary["A1"]["instant_sweep"]["k_configured"] == 2.0 and "| k = 2 s (configured) |" in lines[1]


def test_report_shows_the_sweeps(synth_run):
    run, _ = synth_run
    text, s = (run.path / "report.md").read_text(), json.loads((run.path / "summary.json").read_text())
    tc = s["A1"]["threshold_curve"]
    assert tc["label"] == "public at least t before the report" and tc["hours"] == list(provenance.HOURS)
    assert tc["curves"][0]["answers"] == "all answers" and tc["curves"][0]["share"][0] == s["A1"]["share_exposed"]
    assert "**Exposure vs required head start.**" in text and "| answers | n | 0 | 10 min |" in text
    assert (run.path / "figures" / "a1_threshold.png").exists() and "figures/a1_threshold.png" in text
    assert "still-visible copies" not in text                          # no lifecycle in a chat transcript
    assert "| answers with a stated latency | k = 1 s | k = 2 s (configured) | k = 5 s | k = 10 s |" in text
    assert s["A1"]["instant_sweep"]["k_configured"] == 2
    ws = s["A6"]["window_sensitivity"]
    assert [r["W_s"] for r in ws] == [600, 3600, 21600, 86400, None]
    assert "**Sensitivity to the relay window W.**" in text and "| none (the graph above) |" in text
    hour = next(r for r in ws if r["W_s"] == 3600)["mean_max_depth"]
    assert f"hops on average ({hour:.1f} if a parent must have posted within 1 h; A6 varies that window)" in text
