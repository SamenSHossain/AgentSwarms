"""A2's exploratory dose response among exposed rounds: the consensus head start exposure stores, the synthetic copy
ramp, and an estimator that finds a planted ramp and nothing without one."""

import json

import numpy as np
import pandas as pd
import pytest

from swarmprov import pipeline, synth
from swarmprov.analysis import causal
from swarmprov.schema import RunDir

RAMP_H = 4.0


def _head_at_arrival(truth: pd.DataFrame) -> pd.Series:
    """Hours the item had been on the board (another agent's first report) when each question arrived; NaN if not."""
    t = truth.assign(t_arrival=pd.to_datetime(truth["t_arrival"]), t_report=pd.to_datetime(truth["t_report"]))
    out = []
    for r in t.itertuples():
        o = t[(t["family"] == r.family) & (t["item"] == r.item) & (t["agent"] != r.agent) & (t["t_report"] <= r.t_arrival)]
        out.append((r.t_arrival - o["t_report"].min()).total_seconds() / 3600 if len(o) else np.nan)
    return pd.Series(out, index=truth.index)


@pytest.fixture(scope="module")
def ramp_run(tmp_path_factory):
    """60 agents whose copying ramps up over the answer's first 4 h on the board (seed 6: the slope is detected on
    6 of 8 seeds at 120 agents and 3 of 8 at 60; seed 6 is one of them, with both middle bins identified)."""
    d = tmp_path_factory.mktemp("ramp")
    path = synth.write(d / "transcript.jsonl", n_agents=60, copy_rate=0.7, seed=6, copy_ramp_h=RAMP_H)
    run = RunDir(d / "run")
    pipeline.ingest(path, run, "chat", str(d / "config.json"))
    pipeline.extract(run)
    pipeline.expose(run)
    return run.read("exposures_merged"), pd.read_csv(d / "truth.csv")


def test_copy_ramp_keeps_arrivals_and_lowers_early_copying():
    plain, truth = synth.simulate(n_agents=60, seed=6)
    same, truth_none = synth.simulate(n_agents=60, seed=6, copy_ramp_h=None)
    assert same == plain and truth_none.equals(truth)
    _, ramp = synth.simulate(n_agents=60, seed=6, copy_ramp_h=RAMP_H)
    assert (ramp["t_arrival"] == truth["t_arrival"]).all() and (ramp["item"] == truth["item"]).all()
    for t in (truth, ramp):
        t["head_h"] = _head_at_arrival(t)
    early, late = (lambda t: t[t["head_h"] < 1]), (lambda t: t[t["head_h"] >= RAMP_H])
    # copy probability 0.7 * head / 4 under 1 h (at most 0.175), the full 0.7 from 4 h on
    assert early(ramp)["copied"].mean() < 0.3 < 0.5 < early(truth)["copied"].mean()
    assert abs(late(ramp)["copied"].mean() - late(truth)["copied"].mean()) < 0.15
    with pytest.raises(ValueError, match="copy_ramp_h"):
        synth.simulate(n_agents=4, copy_ramp_h=0)


def test_exposure_stores_the_consensus_head_start(synth_run):
    ex = synth_run[0].read("exposures_merged")
    assert (ex["gap_cons_s"].notna() == (ex["D_cons"] == 1)).all() and (ex["t_cons"].notna() == (ex["D_cons"] == 1)).all()
    assert ((ex["gap_cons_s"] >= 3600) == (ex["D_cons_w60m"] == 1)).all()
    on = ex[ex["D_cons"] == 1]
    gap = (pd.to_datetime(on["t_report"]) - pd.to_datetime(on["t_cons"])).dt.total_seconds()
    assert np.allclose(gap, on["gap_cons_s"]) and (on["gap_cons_s"] > 0).all()


def test_dose_detects_a_planted_ramp(ramp_run):
    ex, truth = ramp_run
    dose = causal.estimate_dose(ex)
    known = dose[dose["outcome"] == "y_instant|known"].set_index("term")
    s = known.loc[causal.SLOPE]
    assert pd.isna(s["note"]) and s["within_sd"] >= causal.MIN_DOSE_SD and s["bin_agents"] >= causal.MIN_BIN_AGENTS
    assert s["beta"] > 0.15 and s["p"] < 0.05
    # the design's own estimand: the same regression on each round's true copy probability (copied => instant)
    truth = truth.assign(head_h=_head_at_arrival(truth))
    p_true = 0.7 * np.minimum(1, truth["head_h"] / RAMP_H).fillna(0)
    smp = causal._dose_sample(ex, "y_instant|known")
    key = dict(zip(zip(truth["cohort"], truth["item"]), p_true))
    smp["p_true"] = [key[(a.split("|")[0], it)] for a, it in zip(smp["agent"], smp["item"])]
    target = causal._fit(smp, "p_true ~ log_head").loc["log_head", "Estimate"]
    assert target > 0.2 and s["ci_lo"] <= target <= s["ci_hi"]
    assert known.loc["1-6h", "beta"] > 0 and known.loc["6-24h", "beta"] > 0
    assert pd.isna(known.loc[causal.DOSE_REF, "beta"]) and pd.isna(known.loc[causal.DOSE_REF, "note"])
    assert known.loc[">24h", "agents_spanning"] < causal.MIN_BIN_AGENTS and pd.isna(known.loc[">24h", "beta"])
    assert known.loc[">24h", "note"].startswith("not identified: only")
    # every synthetic round states its latency, so the two outcomes are the same rounds
    both = dose.pivot(index="term", columns="outcome", values="beta")
    assert np.allclose(both["y_instant"], both["y_instant|known"], equal_nan=True)
    assert causal.stated_latency_slope(ex) == {}


def test_dose_is_flat_without_a_ramp_and_stays_out_of_the_a2_headline(synth_run):
    run, _ = synth_run
    dose = causal.estimate_dose(run.read("exposures_merged")).set_index(["outcome", "term"])
    s = dose.loc[("y_instant|known", causal.SLOPE)]
    assert pd.notna(s["beta"]) and abs(s["beta"]) < 2 * s["se"]    # |z| <= 1.74 on seeds 0-7 at 120 agents
    text = (run.path / "report.md").read_text()
    a2 = text[text.index("## A2."):text.index("## A3.")]
    head, block = a2.split("**Exploratory: dose response among exposed rounds.**")
    assert "dose" not in head and "a2_dose.png" in block and (run.path / "figures" / "a2_dose.png").exists()
    assert "So it identifies only this intensive margin" in block
    assert "`y_instant` is shown but biased: a round with no stated latency scores 0" in block
    assert "here every exposed round states one, so the two rows agree" in block
    assert f"at least {causal.MIN_DOSE_SD} (a spread of about 28% in 1 + hours)" in block
    assert "| y_instant\\|known | <1h (reference) |" in block
    summary = json.loads((run.path / "summary.json").read_text())
    d = summary["A2_dose"]
    assert d["label"].startswith("exploratory") and d["min_bin_agents"] == causal.MIN_BIN_AGENTS
    assert len(d["rows"]) == len(causal.DOSE_OUTCOMES) * (1 + len(causal.HEAD_BINS)) and d["other_mapping"]
    assert {r["outcome"] for r in summary["A2"]} >= {"y_instant", "y_consensus"}    # the A2 table is unchanged


def _panel(n_agents: int, heads_h, prefix: str = "a", n_items: int = 5) -> pd.DataFrame:
    """Agent ``a`` answers item ``(a + k) % n_items`` at the k-th head start of ``heads_h`` (hours), so each item
    mixes head starts; every round is exposed, and the outcome is a fixed coin flip per round."""
    rows = [{"agent": f"{prefix}{a}", "item_fe": f"f|i{(a + k) % n_items}", "D_cons": 1, "gap_cons_s": h * 3600,
             "latency_known": 1} for a in range(n_agents) for k, h in enumerate(heads_h)]
    return pd.DataFrame(rows)


def _dose(*panels) -> pd.DataFrame:
    df = pd.concat(panels, ignore_index=True)
    df["y_instant"] = np.random.default_rng(0).integers(0, 2, len(df))
    return causal.estimate_dose(df).set_index(["outcome", "term"]).loc["y_instant"]


def test_dose_gates_say_why_a_row_is_not_identified():
    # every agent's head starts are 2.0 and 2.2 h: a log-scale SD of about 0.05, under 0.25
    flat = _dose(_panel(30, [2.0, 2.2]))
    sd = flat.loc[causal.SLOPE, "within_sd"]
    assert sd < 0.1 and flat["beta"].isna().all()
    assert (flat["note"] == f"not identified: within-agent SD of the log head start is {sd:.2f} (< 0.25)").all()
    # wide spreads, but only 8 agents have rounds in two bins (the rest stay inside 6-24 h)
    few = _dose(_panel(8, [0.5, 30]), _panel(30, [7, 20], "b"))
    assert (few["note"] == "not identified: only 8 agents have rounds in two or more head-start bins").all()
    # 12 agents span <1 h, 6-24 h and >24 h; only 3 reach 1-6 h, so that bin alone is not identified
    mixed = _dose(_panel(12, [0.5, 30, 10]), _panel(3, [3, 0.5], "c"))
    assert mixed.loc["1-6h", "note"] == "not identified: only 3 agents have a round here and in another bin"
    assert mixed.loc[["6-24h", ">24h", causal.SLOPE], "beta"].notna().all()
    assert mixed.loc[[causal.SLOPE, causal.DOSE_REF, "6-24h", ">24h"], "note"].isna().all()
    assert (mixed.loc[["6-24h", ">24h", causal.DOSE_REF], "agents_spanning"] == [12, 12, 15]).all()
    # a bin whose rounds are all of one item, which has no other rounds: the item effect absorbs it
    solo = _panel(12, [10], "a").assign(item_fe="f|solo")
    collinear = _dose(_panel(12, [0.5, 30]), solo)
    assert collinear.loc["6-24h", "note"] == "not estimated: collinear with the agent and item fixed effects"
    assert pd.notna(collinear.loc[">24h", "beta"])
    # a thin reference leaves every bin unidentified, but not the slope
    thin = _dose(_panel(30, [3, 30, 10]), _panel(4, [0.5, 30], "d"))
    assert thin.loc[causal.DOSE_REF, "note"] == "not identified: only 4 agents have a round here and in another bin"
    assert (thin.loc[["1-6h", "6-24h", ">24h"], "note"] == "not identified: only 4 agents span the <1h reference bin").all()
    assert pd.notna(thin.loc[causal.SLOPE, "beta"]) and thin.loc[["1-6h", "6-24h", ">24h"], "beta"].isna().all()
