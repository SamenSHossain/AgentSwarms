"""A2's exploratory dose response among exposed rounds: the consensus head start exposure stores, the synthetic copy
ramp, and an estimator that finds a planted ramp and nothing without one."""

import json

import numpy as np
import pandas as pd
import pytest

from swarmprov import cli, pipeline, report, synth
from swarmprov.analysis import causal
from swarmprov.schema import RunDir

RAMP_H, RAMP_RATE, RAMP_SEEDS = 4.0, 0.9, (0, 1, 2)


def _head_at_arrival(truth: pd.DataFrame) -> pd.Series:
    """Hours the item had been on the board (another agent's first report) when each question arrived; NaN if not."""
    t = truth.assign(t_arrival=pd.to_datetime(truth["t_arrival"]), t_report=pd.to_datetime(truth["t_report"]))
    out = []
    for r in t.itertuples():
        o = t[(t["family"] == r.family) & (t["item"] == r.item) & (t["agent"] != r.agent) & (t["t_report"] <= r.t_arrival)]
        out.append((r.t_arrival - o["t_report"].min()).total_seconds() / 3600 if len(o) else np.nan)
    return pd.Series(out, index=truth.index)


@pytest.fixture(scope="module")
def ramp_runs(tmp_path_factory):
    """60 agents copying at rate 0.9, ramped up over the answer's first 4 h on the board, on three seeds.  Over seeds
    0-7 the slope on `y_instant|known` is positive with p < 0.01 and its 95% CI covers the target on all eight."""
    out = []
    for seed in RAMP_SEEDS:
        d = tmp_path_factory.mktemp(f"ramp{seed}")
        path = synth.write(d / "transcript.jsonl", n_agents=60, copy_rate=RAMP_RATE, seed=seed, copy_ramp_h=RAMP_H)
        run = RunDir(d / "run")
        pipeline.ingest(path, run, "chat", str(d / "config.json"))
        pipeline.extract(run)
        pipeline.expose(run)
        out.append((run.read("exposures_merged"), pd.read_csv(d / "truth.csv")))
    return out


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


def test_exposure_stores_the_consensus_head_start(synth_run):
    ex = synth_run[0].read("exposures_merged")
    assert (ex["gap_cons_s"].notna() == (ex["D_cons"] == 1)).all() and (ex["t_cons"].notna() == (ex["D_cons"] == 1)).all()
    assert ((ex["gap_cons_s"] >= 3600) == (ex["D_cons_w60m"] == 1)).all()
    on = ex[ex["D_cons"] == 1]
    gap = (pd.to_datetime(on["t_report"]) - pd.to_datetime(on["t_cons"])).dt.total_seconds()
    assert np.allclose(gap, on["gap_cons_s"]) and (on["gap_cons_s"] > 0).all()


def _target(ex: pd.DataFrame, truth: pd.DataFrame) -> float:
    """The design's own estimand: the same regression on each round's true copy probability (copied => instant)."""
    truth = truth.assign(head_h=_head_at_arrival(truth))
    p_true = RAMP_RATE * np.minimum(1, truth["head_h"] / RAMP_H).fillna(0)
    smp = causal._dose_sample(ex, "y_instant|known")
    key = dict(zip(zip(truth["cohort"], truth["item"]), p_true))
    smp["p_true"] = [key[(a.split("|")[0], it)] for a, it in zip(smp["agent"], smp["item"])]
    return causal._fit(smp, "p_true ~ log_head").loc["log_head", "Estimate"]


def test_dose_detects_a_planted_ramp_on_every_seed(ramp_runs):
    slopes = []
    for ex, truth in ramp_runs:
        dose = causal.estimate_dose(ex)
        s = dose.set_index(["outcome", "term"]).loc[("y_instant|known", causal.SLOPE)]
        assert pd.isna(s["note"]) and s["agents_informative"] >= 2 * causal.MIN_INFORMATIVE
        target = _target(ex, truth)
        assert target > 0.2 and s["beta"] > 0 and s["ci_lo"] <= target <= s["ci_hi"]
        # no agent dominates a synthetic swarm: the slope survives leaving any one out
        assert 0 < s["loo_lo"] <= s["beta"] <= s["loo_hi"] and s["var_share_top2"] < 0.5
        slopes.append(s)
    assert sum(s["p"] < 0.05 for s in slopes) >= 2
    # every synthetic round states its latency, so the two outcomes are the same rounds
    ex = ramp_runs[0][0]
    both = causal.estimate_dose(ex).pivot(index="term", columns="outcome", values="beta")
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
    assert "It identifies only this intensive margin and says nothing about whether exposure itself matters (the " \
           "extensive margin), which this block leaves unidentified" in block
    assert "| agents spanning | agents informative | note |" in block and "_Leave one agent out_: dropping one" in block
    inf = int(dose.loc[("y_instant|known", causal.SLOPE), "agents_informative"])
    assert f"on `y_instant|known`, {inf} agents vary in both the head start and the outcome" in block
    assert f"{inf} informing the slope" in block and "after dropping single-round agents and items" in block
    assert "`y_instant` is shown but biased: a round with no stated latency scores 0" in block
    assert "here every exposed round states one, so the two rows agree" in block
    assert f"at least {causal.MIN_DOSE_SD} (a spread of about 28% in 1 + hours)" in block
    assert "| y_instant\\|known | <1h (reference) |" in block
    summary = json.loads((run.path / "summary.json").read_text())
    d = summary["A2_dose"]
    assert d["label"].startswith("exploratory") and d["min_bin_agents"] == causal.MIN_BIN_AGENTS
    assert d["min_informative"] == causal.MIN_INFORMATIVE and d["rows"][0]["loo_lo"] is not None
    assert len(d["rows"]) == len(causal.DOSE_OUTCOMES) * (1 + len(causal.HEAD_BINS)) and d["other_mapping"]
    assert {r["outcome"] for r in summary["A2"]} >= {"y_instant", "y_consensus"}    # the A2 table is unchanged


def _panel(n_agents: int, heads_h, prefix: str = "a", n_items: int = 5) -> pd.DataFrame:
    """Agent ``a`` answers item ``(a + k) % n_items`` at the k-th head start of ``heads_h`` (hours), so each item
    mixes head starts; every round is exposed and states its latency."""
    rows = [{"agent": f"{prefix}{a}", "item_fe": f"f|i{(a + k) % n_items}", "D_cons": 1, "gap_cons_s": h * 3600,
             "latency_known": 1} for a in range(n_agents) for k, h in enumerate(heads_h)]
    return pd.DataFrame(rows)


def _coins(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A fixed coin flip per round as ``y_instant``, an agent's first round flipped where all its flips agree, so every
    agent's outcome varies."""
    y = pd.Series(np.random.default_rng(seed).integers(0, 2, len(df)), index=df.index)
    flip = (y.groupby(df["agent"]).transform("nunique") == 1) & ~df["agent"].duplicated()
    return df.assign(y_instant=np.where(flip, 1 - y, y))


def _dose(*panels) -> pd.DataFrame:
    df = _coins(pd.concat(panels, ignore_index=True))
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
    # 12 agents span <1 h, 6-24 h and >24 h; only 3 reach 1-6 h, so that bin alone is not identified, and the rest
    # rest on fewer than 20 agents: estimated, with a note that says so
    mixed = _dose(_panel(12, [0.5, 30, 10]), _panel(3, [3, 0.5], "c"))
    assert mixed.loc["1-6h", "note"] == "not identified: only 3 agents have a round here and in another bin"
    assert mixed.loc[["6-24h", ">24h", causal.SLOPE], "beta"].notna().all() and pd.isna(mixed.loc[causal.DOSE_REF, "note"])
    assert (mixed.loc[["6-24h", ">24h"], "note"] == "shape only: 12 agents inform it (< 20)").all()
    assert mixed.loc[causal.SLOPE, "note"] == "weak: only 15 agents inform it (< 20)"
    assert (mixed.loc[["6-24h", ">24h", causal.DOSE_REF], "agents_spanning"] == [12, 12, 15]).all()
    assert (mixed.loc[["6-24h", ">24h", causal.DOSE_REF], "agents_informative"] == [12, 12, 15]).all()
    # a bin whose rounds are all of one item, which has no other rounds: the item effect absorbs it
    solo = _panel(12, [10], "a").assign(item_fe="f|solo")
    collinear = _dose(_panel(12, [0.5, 30]), solo)
    assert collinear.loc["6-24h", "note"] == "not estimated: collinear with the agent and item fixed effects"
    assert pd.notna(collinear.loc[">24h", "beta"])
    # a thin reference leaves every bin unidentified, but not the slope; 20 agents in a bin but 14 in the reference
    thin = _dose(_panel(30, [3, 30, 10]), _panel(4, [0.5, 30], "d"))
    assert thin.loc[causal.DOSE_REF, "note"] == "not identified: only 4 agents have a round here and in another bin"
    assert (thin.loc[["1-6h", "6-24h", ">24h"], "note"] == "not identified: the <1h reference bin is not").all()
    assert pd.notna(thin.loc[causal.SLOPE, "beta"]) and thin.loc[["1-6h", "6-24h", ">24h"], "beta"].isna().all()
    ref14 = _dose(_panel(30, [3, 30]), _panel(14, [0.5, 30], "e"))
    assert ref14.loc["1-6h", "note"] == "shape only: 14 agents inform the <1h reference (< 20)"
    assert pd.isna(ref14.loc[causal.SLOPE, "note"]) and pd.notna(ref14.loc["1-6h", "beta"])


def test_dose_counts_only_agents_whose_outcome_varies():
    df = _panel(30, [0.5, 3, 10, 30])
    # 8 agents alternate 0/1 across their rounds, 22 answer every round instantly
    varied = _coins(df).assign(y_instant=np.where(df["agent"].isin([f"a{i}" for i in range(8)]), df.index % 2, 1))
    dose = causal.estimate_dose(varied).set_index(["outcome", "term"]).loc["y_instant"]
    assert dose["beta"].isna().all() and (dose["agents_spanning"] == 30).all() and (dose["agents_informative"] == 8).all()
    assert dose.loc[causal.SLOPE, "note"] == ("not identified: only 8 of the 30 agents whose head start varies also "
                                              "vary in the outcome")
    assert (dose.loc[list(causal.HEAD_BINS), "note"] == ("not identified: only 8 of the 30 agents with a round here "
                                                         "and in another bin vary in the outcome")).all()
    # an outcome that never varies is not identified anywhere, rather than estimated at 0 with SE 0
    flat = causal.estimate_dose(df.assign(y_instant=1)).set_index(["outcome", "term"]).loc["y_instant"]
    assert flat["beta"].isna().all() and flat["se"].isna().all() and (flat["agents_informative"] == 0).all()
    assert flat.loc[causal.SLOPE, "note"].startswith("not identified: only 0 of the 30 agents")


def test_dose_notes_a_fit_that_fails_or_fits_exactly(monkeypatch):
    df = _coins(_panel(30, [0.5, 3, 10, 30]))
    # the outcome is the item's alone: it varies within every agent, but the fixed effects fit it exactly
    exact = causal.estimate_dose(df.assign(y_instant=df["item_fe"].isin(["f|i0", "f|i1"]).astype(int)))
    exact = exact.set_index(["outcome", "term"]).loc["y_instant"]
    assert exact.loc[causal.SLOPE, "note"] == "not estimated: the model fits the outcome exactly"
    assert exact["beta"].isna().all()

    def boom(*_):
        raise RuntimeError("singular design")
    monkeypatch.setattr(causal, "_fit", boom)
    failed = causal.estimate_dose(df).set_index(["outcome", "term"]).loc["y_instant"]
    assert failed["beta"].isna().all()
    assert (failed.loc[[causal.SLOPE, "1-6h", "6-24h", ">24h"], "note"] == "not estimated: singular design").all()


def test_dose_leave_one_out_is_the_exact_refit_and_finds_a_dominant_agent():
    import pyfixest as pf
    df = _panel(20, [0.5, 3, 10, 30], n_items=6)
    # two agents answer instantly only with a long head start; every other agent answers one item slowly and the rest
    # instantly, which the item effects fit exactly, so the slope's clustered variance comes from the two
    long = (df["gap_cons_s"] > 5 * 3600).astype(int)
    df["y_instant"] = np.where(df["agent"].isin(["a0", "a1"]), long, (df["item_fe"] != "f|i0").astype(int))
    s = causal.estimate_dose(df).set_index(["outcome", "term"]).loc[("y_instant", causal.SLOPE)]
    smp = causal._dose_sample(df, "y_instant")
    refits = [pf.feols("y_instant ~ log_head | agent + item_fe", data=smp[smp["agent"] != a]).coef()["log_head"]
              for a in smp["agent"].unique()]
    assert s["loo_lo"] == pytest.approx(min(refits), abs=1e-9) and s["loo_hi"] == pytest.approx(max(refits), abs=1e-9)
    # the share of the CRV1 variance each agent carries, from pyfixest's own demeaned design and residuals
    fit = pf.feols("y_instant ~ log_head | agent + item_fe", data=smp, vcov={"CRV1": "agent"})
    score = pd.Series(fit._X[:, 0] * fit._u_hat).groupby(fit._data["agent"].to_numpy()).sum() ** 2
    assert s["var_share_top2"] == pytest.approx(score.nlargest(2).sum() / score.sum())
    assert score.idxmax() in ("a0", "a1") and s["var_share_top2"] > 0.5


def test_stated_latency_that_falls_with_the_head_start_is_reported(tmp_path):
    df = _panel(30, [0.5, 3, 10, 30])
    rng = np.random.default_rng(1)
    known = (rng.random(len(df)) < 1 - np.log1p(df["gap_cons_s"] / 3600) / 4).astype(int)
    df = _coins(df).assign(latency_known=known)
    df["y_instant"] *= df["latency_known"]                # an unstated latency scores 0
    lat = causal.stated_latency_slope(df)
    assert lat["beta"] < 0 and lat["p"] < 0.05 and lat["n"] == 120
    est = pd.DataFrame([{"outcome": "y_instant|known", "treatment": "D_cons", "agents_with_D_variation": 0}])
    summary = {}
    block = "\n".join(report._dose_block(tmp_path / "a2_dose.png", df, pd.DataFrame(), est, summary))
    assert (f"the share of rounds stating one changes by {lat['beta']:+.3f} per unit of log1p(hours) "
            f"(SE {lat['se']:.3f}), which pushes its slope down.") in block
    assert ("`y_instant|known` avoids that scoring but keeps only the rounds that state a latency, and whether a round "
            "states one comes after the head start and moves with it, as just measured, so its sample is selected on "
            "the dose: it is the less biased read, not an unbiased one.") in block
    assert "Both outcomes measure the head start to the report" in block and "where exposure barely does" in block
    assert summary["A2_dose"]["stated_latency_slope"] == lat


def test_synth_rejects_a_ramp_that_is_not_a_positive_finite_number_of_hours(tmp_path, capsys):
    for bad in ("0", "-2", "inf", "nan", "abc"):
        with pytest.raises(SystemExit) as e:
            cli.main(["synth", "-o", str(tmp_path / "t.jsonl"), "--copy-ramp-h", bad])
        assert e.value.code == 2 and "expected a positive, finite number of hours" in capsys.readouterr().err
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="copy_ramp_h must be a positive, finite number of hours"):
            synth.simulate(n_agents=4, copy_ramp_h=bad)
    assert not (tmp_path / "t.jsonl").exists()
