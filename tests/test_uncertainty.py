"""Intervals: the cluster bootstrap (A1 by item, A6 by fact), Greenwood's interval on the A3 median, and the A5
leave-one-item-out range."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from swarmprov.analysis import diffusion, provenance, structure, uncertainty
from swarmprov.schema import RunDir


def _mean(col):
    return lambda d, w: uncertainty.wmean(d[col], w)


def test_identical_clusters_give_a_zero_width_interval():
    df = pd.DataFrame({"c": np.repeat(np.arange(10), 3), "x": np.tile([1.0, 0.0, 1.0], 10)})
    lo, hi = uncertainty.cluster_bootstrap(df, "c", _mean("x"), B=200)
    assert lo == pytest.approx(2 / 3) and hi == pytest.approx(2 / 3)


def test_clusters_are_resampled_whole_and_seeded():
    """One cluster all ones, the rest zeros: a resample's mean moves in steps of whole clusters, never of rows."""
    df = pd.DataFrame({"c": np.repeat(np.arange(8), 5), "x": [1.0] * 5 + [0.0] * 35})
    stat = lambda d, w: np.column_stack([uncertainty.wmean(d["x"], w), w.sum(axis=1)])
    w_rows = []
    lo, hi = uncertainty.cluster_bootstrap(df, "c", lambda d, w: w_rows.append(w) or stat(d, w), B=300, seed=3)
    w = np.concatenate(w_rows)
    assert (w.sum(axis=1) == 40).all() and (w.reshape(len(w), 8, 5).std(axis=2) == 0).all()   # rows of a cluster move together
    assert lo[0] == 0 and 0.125 < hi[0] <= 0.375
    again = uncertainty.cluster_bootstrap(df, "c", stat, B=300, seed=3)
    assert np.array_equal(again[0], lo) and np.array_equal(again[1], hi)


def test_too_few_clusters_or_no_reps_give_no_interval():
    df = pd.DataFrame({"c": [1, 1, 2, 3, 4], "x": [1.0, 0.0, 1.0, 0.0, 1.0]})
    for kw in ({"B": 100}, {"B": 0, "min_clusters": 1}):
        lo, hi = uncertainty.cluster_bootstrap(df, "c", _mean("x"), **kw)
        assert np.isnan(lo) and np.isnan(hi)
    lo, hi = uncertainty.cluster_bootstrap(df, None, _mean("x"), B=100)   # every row its own cluster: five
    assert 0 <= lo < hi <= 1


def test_weighted_median_matches_pandas_on_the_expanded_sample():
    rng = np.random.default_rng(0)
    x = np.where(rng.random(30) < 0.2, np.nan, rng.normal(size=30))
    w = rng.integers(0, 3, size=(50, 30))
    got = uncertainty.wmedian(x, w)
    want = [pd.Series(np.repeat(x, row)).median() for row in w]
    assert np.allclose(got, want, equal_nan=True)


def test_greenwood_matches_a_hand_computed_example():
    """Five subjects, events at 1, 2 and 4, censored at 3 and 5: S = 0.8, 0.6, 0.3 with Greenwood sums 1/20,
    1/20 + 1/12, 1/20 + 1/12 + 1/2; the log-log band is S ** exp(-+1.96 sqrt(sum) / |log S|)."""
    km = uncertainty.km_greenwood([1, 2, 3, 4, 5], [1, 1, 0, 1, 0])
    assert list(km["t"]) == [0, 1, 2, 4]
    assert np.allclose(km["survival"], [1, 0.8, 0.6, 0.3])
    assert np.allclose(km["se"], [0, 0.8 * np.sqrt(1 / 20), 0.6 * np.sqrt(1 / 20 + 1 / 12), 0.3 * np.sqrt(19 / 30)])
    assert np.allclose(km["lo"], [1, 0.203809, 0.125730, 0.012302], atol=1e-6)
    assert np.allclose(km["hi"], [1, 0.969180, 0.881756, 0.719218], atol=1e-6)
    assert uncertainty.km_median(km)[0] == 4 and uncertainty.km_median(km)[1] == 1
    assert np.isnan(uncertainty.km_median(km)[2])          # the upper band never falls to half: open upper bound
    everyone = uncertainty.km_greenwood([1, 2], [1, 1])     # the last subject's event takes S to 0, band and all
    assert list(everyone["survival"]) == [1, 0.5, 0] and list(everyone["lo"])[-1] == 0 == list(everyone["hi"])[-1]


def test_a3_median_carries_greenwood_only_where_it_is_reached():
    from swarmprov.adapters.base import Technique
    t0 = pd.Timestamp("2026-06-20T05:00:00Z")
    agents = [f"a{i}" for i in range(10)]
    rows = [{"ts": t0, "agent_merged": "a0", "family": "f", "techniques": "blob"}]
    for i, a in enumerate(agents[1:], start=1):   # a1-a6 adopt after i hours, a7-a9 post at 20 h without it
        rows.append({"ts": t0 + pd.Timedelta(hours=i if i <= 6 else 20), "agent_merged": a, "family": "f",
                     "techniques": "blob" if i <= 6 else ""})
    tags = pd.DataFrame(rows)
    r = diffusion.analyze(tags, [Technique("blob", "x")], "agent_merged")["blob"]
    km = uncertainty.km_greenwood(*zip(*[(i, 1) for i in range(1, 7)] + [(20, 0)] * 3))
    assert (r["median_hours_to_adopt"], r["median_hours_to_adopt_lo"]) == uncertainty.km_median(km)[:2] == (5.0, 1.0)
    assert np.isnan(r["median_hours_to_adopt_hi"])          # the upper band is still 0.62 at the last adoption
    from swarmprov.report import _median_cell
    assert _median_cell(r["median_hours_to_adopt"], r["median_hours_to_adopt_lo"], r["median_hours_to_adopt_hi"]) \
        == "5.0 (1.0–not reached)"
    rare = diffusion.analyze(tags.assign(techniques=np.where(tags["agent_merged"].isin(["a0", "a1"]), "blob", "")),
                             [Technique("blob", "x")], "agent_merged")["blob"]
    assert np.isnan(rare["median_hours_to_adopt"]) and np.isnan(rare["median_hours_to_adopt_lo"])
    assert _median_cell(rare["median_hours_to_adopt"], rare["median_hours_to_adopt_lo"], 9.0) == ""


def test_a1_intervals_cover_the_synthetic_truth(synth_run):
    """The value each agent answered had been posted by another agent before its report: the truth behind D."""
    run, truth = synth_run
    ex = run.read("exposures_merged")
    t = truth.assign(t_report=pd.to_datetime(truth["t_report"]))
    first = t.groupby(["family", "item", "value"])
    public = [((g := first.get_group((r.family, r.item, r.value)))["agent"].ne(r.agent) & (g["t_report"] < r.t_report)).any()
              for r in t.itertuples()]
    r = provenance.summarize(ex, boot=500)
    lo, hi = r["share_exposed_ci"]
    assert r["share_exposed"] == pytest.approx(np.mean(public)) and lo < np.mean(public) < hi
    assert hi - lo > 0.01 and all(r[f"{k}_ci"][0] <= r[k] <= r[f"{k}_ci"][1] for k in provenance.HEADLINE)
    assert provenance.summarize(ex, boot=500)["share_exposed_ci"] == r["share_exposed_ci"]   # seeded
    fam, ep = r["by_family"], r["by_episode"]
    assert {"items", "exposed_lo", "exposed_hi", "exposed_60m_lo", "exposed_60m_hi"} <= set(fam) and len(ep) == 5
    assert ((fam["exposed_lo"] <= fam["exposed"]) & (fam["exposed"] <= fam["exposed_hi"])).all()
    plain = provenance.summarize(ex)
    assert not any(k.endswith("_ci") for k in plain) and "exposed_lo" not in plain["by_family"]
    seq = provenance.summarize(ex, boot=100, wall_clock=False)
    assert seq["share_exposed_60m_ci"] is None and seq["median_gap_h_ci"] is None and seq["share_exposed_ci"]
    assert "exposed_60m_lo" not in seq["by_family"] and "exposed_lo" in seq["by_family"]


def test_family_rows_with_few_items_get_no_interval():
    ex = pd.DataFrame({"family": ["f"] * 6 + ["g"] * 6, "item_fe": ["f|a", "f|b", "f|c"] * 2 + [f"g|{i}" for i in range(6)],
                       "D": [1, 0] * 6, "D_w10m": 0, "D_w60m": 0, "self_prepared": 0, "y_instant": 0,
                       "gap_s": np.nan, "agent": [f"x{i}" for i in range(12)], "episode": 1})
    fam = provenance.summarize(ex, min_n=1, boot=100)["by_family"]
    assert np.isnan(fam.loc["f", "exposed_lo"]) and fam.loc["f", "items"] == 3
    assert fam.loc["g", "exposed_lo"] <= 0.5 <= fam.loc["g", "exposed_hi"]


def test_leave_one_item_out_range():
    """Credits A, A (item 1), B (item 2), C (item 3): the top-1 share is 1/2, 1/2 without item 1, 2/3 without 2 or 3."""
    src, items = pd.Series(["A", "A", "B", "C"]), pd.Series(["i1", "i1", "i2", "i3"])
    r = structure.leave_one_item_out(src, items, top_k=1)
    assert (r["top1_share_loo_lo"], r["top1_share_loo_hi"]) == pytest.approx((0.5, 2 / 3))
    assert r["gini_loo_lo"] <= structure.concentration(src, 1)["gini"] <= r["gini_loo_hi"]
    assert np.isnan(structure.leave_one_item_out(src, pd.Series(["i1"] * 4))["gini_loo_lo"])   # one item: no range


def test_reach_interval_resamples_facts(synth_run):
    run, _ = synth_run
    chains = run.read("chains_merged")
    r = structure.reach(chains, boot=400)["task values"]
    assert r["mean_repeaters_lo"] < r["mean_repeaters"] < r["mean_repeaters_hi"]
    assert "mean_repeaters_lo" not in structure.reach(chains)["task values"]
    same = chains.assign(n_agents=3)
    r = structure.reach(same, boot=100)["task values"]
    assert r["mean_repeaters_lo"] == r["mean_repeaters_hi"] == 2


def test_report_states_the_intervals(synth_run):
    run, _ = synth_run
    text, s = (run.path / "report.md").read_text(), json.loads((run.path / "summary.json").read_text())
    a1 = s["A1"]
    assert a1["bootstrap"]["reps"] == 1000 and a1["bootstrap"]["cluster"] == "item_fe"
    lo, hi = a1["share_exposed_ci"]
    assert f"{a1['share_exposed']:.0%} (95% CI {lo * 100:.0f}–{hi:.0%}) of answers were already public" in text
    assert f"Median head start: {a1['median_gap_h']:.1f} h ({a1['median_gap_h_ci'][0]:.1f}–" in text
    assert text.count("95% CI ") == 1 and "resamples of whole **items**" in text   # the cluster unit, named once
    assert "not identity-mapping or extraction error" in text and "| exposed (95% CI) |" in text
    a5 = s["A5"]
    assert "Leaving out any one item moves the share between " in text and len(a5["top10_share_exposure_loo"]) == 2
    assert "| top-10 share, one item left out | Gini, one item left out |" in text
    assert {"top10_share_loo_lo", "gini_loo_hi"} <= set(a5["attribution"][0])
    assert s["A6"]["task values"]["mean_repeaters_lo"] < s["A6"]["task values"]["mean_repeaters"]
    assert "mean_repeaters (95% CI)" in text and "resamples of the facts" in text


def test_report_boot_zero_drops_the_bootstrap_but_keeps_greenwood(synth_run, tmp_path):
    """``swarmprov report --boot 0`` on a copy of the synthetic run given a technique most agents use."""
    from swarmprov import cli
    src, _ = synth_run
    run = RunDir(shutil.copytree(src.path, tmp_path / "run"))
    prof = json.loads((run.path / "profile.json").read_text())
    cfg = json.loads(open(prof["config_path"]).read()) | {"techniques": [{"name": "instant-copy", "pattern": "instantly"}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (run.path / "profile.json").write_text(json.dumps(prof | {"config_path": str(tmp_path / "cfg.json")}))
    ev, tags = run.read("events"), run.read("tags")
    hit = tags["event_id"].map(ev.set_index("event_id")["text"]).str.contains("instantly")
    run.write("tags", tags.assign(techniques=np.where(hit, "instant-copy", "")))
    assert cli.main(["report", str(run.path), "--boot", "0"]) == 0
    text, s = (run.path / "report.md").read_text(), json.loads((run.path / "summary.json").read_text())
    assert "95% CI" not in text.split("## A2")[0] and "bootstrap" not in s["A1"] and "share_exposed_ci" not in s["A1"]
    assert "| exposed |" in text and "mean_repeaters_lo" not in s["A6"]["task values"]
    a3 = s["A3"][0]
    assert a3["median_hours_to_adopt_lo"] < a3["median_hours_to_adopt"] < a3["median_hours_to_adopt_hi"]
    cell = f"{a3['median_hours_to_adopt']:.1f} ({a3['median_hours_to_adopt_lo']:.1f}–{a3['median_hours_to_adopt_hi']:.1f})"
    assert f"| {cell} |" in text and "median_hours_to_adopt (95% CI)" in text and "Greenwood's 95% interval" in text
    assert "Leaving out any one item" in text                # not a bootstrap: kept
    with pytest.raises(SystemExit):
        cli.main(["report", str(run.path), "--boot", "-1"])
