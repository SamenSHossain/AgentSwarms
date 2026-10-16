"""Figures say what their tables say: asserted on the drawn data (line ends, tick labels, texts), not on pixels."""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest

from swarmprov import plotting
from swarmprov.analysis import causal, diffusion, errors, structure

T0 = pd.Timestamp("2026-06-20", tz="UTC")


@pytest.fixture
def drawn(monkeypatch):
    """The figures a call draws, kept open instead of saved."""
    figs = []
    monkeypatch.setattr(plotting, "save", lambda fig, path: figs.append(fig) or str(path))
    yield figs
    for f in figs:
        plotting.plt.close(f)


def _labels(fig, ax, axis="y"):
    fig.canvas.draw()
    ticks = ax.get_yticks() if axis == "y" else ax.get_xticks()
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    texts = [t.get_text() for t in (ax.get_yticklabels() if axis == "y" else ax.get_xticklabels())]
    return [(v, t) for v, t in zip(ticks, texts) if lo <= v <= hi]


def test_pct_axis_prints_the_decimals_the_range_needs():
    fig, ax = plotting.plt.subplots()
    ax.set_ylim(0, 0.19)
    plotting.pct_axis(ax)
    ax.set_yticks(np.arange(0, 0.19, 0.025))
    shown = _labels(fig, ax)
    plotting.plt.close(fig)
    assert ("2.5%" in dict(shown).values()) and all(abs(float(t.rstrip("%")) - v * 100) < 1e-9 for v, t in shown)


def test_a3_percent_ticks_are_not_rounded(drawn):
    km = pd.DataFrame({"t": [0.0, 1.0, 2.0], "survival": [1.0, 0.9, 0.825]})
    res = {"wide": {"adopters": 3, "at_risk": 20, "km": km}}
    diffusion.figure(res, "a3.png")
    fig = drawn[0]
    ax = fig.axes[0]
    ax.set_yticks(np.arange(0, 0.18, 0.025))       # the formatter the figure installed, on 2.5% steps
    shown = _labels(fig, ax)
    assert {"2.5%", "17.5%"} <= set(dict(shown).values())      # a rounding formatter prints 2% and 18%
    assert all(abs(float(t.rstrip("%")) - v * 100) < 1e-9 for v, t in shown)


def _slot(spec):
    """spec: (hour, variant) per carrier, one agent each; the slot dict the report passes to the figure."""
    car = pd.DataFrame({"ts": [T0 + pd.Timedelta(hours=h) for h, _ in spec],
                        "agent": [f"a{i}" for i in range(len(spec))], "variant": [v for _, v in spec]})
    s = errors._slot_stats(car, "fam | item")
    return s | {"agents_per_variant": car.groupby("variant")["agent"].nunique().to_dict()}


def _a4(drawn, spec):
    """(slot, each variant line's last y by variant, legend texts, top axes, share axes)."""
    s = _slot(spec)
    errors.figure([s], "a4.png")
    fig = drawn[-1]
    ax, axb = fig.axes[0], fig.axes[1]
    ends = {ln.get_label().split(" (")[0]: ln.get_ydata()[-1] for ln in ax.lines if " agents)" in ln.get_label()}
    legend = [t.get_text() for t in axb.get_legend().get_texts()]
    return s, ends, legend, ax, axb


# both variants first seen at 0 h: a 3% trim drops both, so each line would end one short of its agents
SHARED_START = [(0, "A"), (0, "B")] + [(0.25 + i / 6, "A") for i in range(15)] + [(3.5 + i / 6, "B") for i in range(20)]
# one stray A carrier three days early, then B takes every bin from 3 h on (one carrier in its last bin)
STRAY = [(-72, "A")] + [(i / 6, "A") for i in range(6)] + [(3.5 + i / 6, "B") for i in range(12)] + [(9.5, "B")]


def test_a4_lines_end_at_every_variants_agents(drawn):
    s, ends, legend, ax, _ = _a4(drawn, SHARED_START)
    assert ends == s["agents_per_variant"] == {"A": 16, "B": 21}
    assert ax.get_xlim()[0] <= mdates.date2num(T0)                   # nothing clipped
    assert not any("earlier carrier" in t for t in legend)
    assert s["status"] == "contested" and "B (21 agents), winner" in legend


def test_a4_clipped_axis_names_the_earlier_carriers_and_marks_the_overturn(drawn):
    s, ends, legend, ax, axb = _a4(drawn, STRAY)
    assert ends == s["agents_per_variant"] == {"A": 7, "B": 13}
    left = mdates.num2date(ax.get_xlim()[0])
    assert T0 - pd.Timedelta(hours=72) < left < T0
    assert f"1 earlier carrier before {left:%m-%d %H:%M}" in legend
    assert s["status"] == "overturned" and "B (13 agents), winner" in legend and "A (7 agents)" in legend
    t50 = s["t_winner_majority"]
    assert [ln.get_xdata()[0] for a in (ax, axb) for ln in a.lines if ln.get_linestyle() == "--"] == [t50, t50]
    assert any(t.startswith("winner's lasting majority of new carriers from") for t in legend)
    # the share row: the winner's share in every non-empty bin, the one-carrier bin faint
    new = s["new_by_bin"].sum(axis=1)
    segs = [(seg[0][1], c.get_alpha()) for c in axb.collections for seg in c.get_segments()]
    want = s["share_by_bin"]["B"][new > 0]
    assert sorted(y for y, _ in segs) == sorted(want)
    assert sorted(a for _, a in segs) == sorted(0.3 if n < errors.SPARSE_BIN else 1.0 for n in new[new > 0])
    assert any(t.startswith("share in a bin with fewer than") for t in legend)


def test_a4_notes_a_winner_without_a_lasting_majority_and_a_tie(drawn):
    hungary = [(i / 6, "A") for i in range(14)] + [(3.5 + i / 6, "B") for i in range(4)] + \
              [(6.5, "A"), (6.6, "A"), (7, "B")]
    s, _, legend, ax, _ = _a4(drawn, hungary)
    assert s["status"] == "contested" and "B never held a lasting majority of new carriers" in legend
    assert not [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    tie = [(i / 6, "A") for i in range(8)] + [(3.5, "B"), (4, "A"), (4.5, "B"), (5, "A")]
    s, _, legend, _, _ = _a4(drawn, tie)
    assert s["winner"] is None and "no winner: the late carriers tie" in legend
    assert not any(t.endswith(", winner") for t in legend)


def test_a2_draws_every_row_and_says_which_are_not_identified(drawn):
    est = pd.DataFrame([
        {"outcome_label": "Instant", "treatment": "D_cons", "treatment_label": "public", "beta": -0.07, "ci_lo": -0.2,
         "ci_hi": 0.06, "n": 787, "agents_with_D_variation": 37, "note": np.nan},
        {"outcome_label": "Instant, known", "treatment": "D_cons", "treatment_label": "public", "beta": np.nan,
         "ci_lo": np.nan, "ci_hi": np.nan, "n": 328, "agents_with_D_variation": 4,
         "note": "not identified: only 4 agents switch exposure status"},
        {"outcome_label": "Instant, known", "treatment": "D_cons_w60m", "treatment_label": ">= 1 h", "beta": 0.001,
         "ci_lo": -0.007, "ci_hi": 0.009, "n": 328, "agents_with_D_variation": 10, "note": np.nan}])
    causal.figure(est, "a2.png")
    fig = drawn[0]
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "not identified (4 agents switch)" in texts
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "Instant\n[public], 37 switchers", "Instant, known\n[public], 4 switchers",
        "Instant, known\n[>= 1 h], 10 switchers"]
    bars = {round(c.lines[0].get_xdata()[0], 3): c.lines[0].get_color() for c in ax.containers}
    # no bar for the unidentified row, and 10 switchers (under 20) is grey
    assert bars == {-0.07: plotting.SERIES[0], 0.001: plotting.NEUTRAL}
    assert f"under {causal.MIN_SWITCHERS} switchers" in ax.get_title("left")


def test_a5_lorenz_marks_where_the_top_sources_begin(drawn):
    outdeg = pd.Series([40, 30, 20, 10] + [2] * 8 + [1] * 8, index=[f"s{i}" for i in range(20)])
    share = outdeg.head(10).sum() / outdeg.sum()
    res = {"outdeg": outdeg, "gini_exposure_outdegree": 0.6, "top10_share_exposure": share,
           "chain_depth": pd.Series(dtype=int)}
    structure.figure(res, pd.DataFrame(), "a5.png")
    ax = drawn[0].axes[0]
    curve, dot = ax.lines[0], next(ln for ln in ax.lines if ln.get_label().startswith("top 10 of 20"))
    x, y = dot.get_xdata()[0], dot.get_ydata()[0]
    assert (x, y) == (0.5, pytest.approx(1 - share))
    assert y == pytest.approx(np.interp(x, curve.get_xdata(), curve.get_ydata()))   # on the curve
    assert f"{share:.0%} of exposed" in dot.get_label() and "first to post the value" in dot.get_label()


def test_a6_pools_twenty_or_more_into_one_labelled_marker(drawn):
    chains = pd.DataFrame({"family": ["f"] * 9 + ["url"] * 3,
                           "n_agents": [1, 1, 2, 3, 5, 20, 21, 30, 69, 1, 2, 2]})
    structure.reach_figure(chains, "a6.png")
    fig = drawn[0]
    ax = fig.axes[0]
    shown = _labels(fig, ax, "x")
    assert shown[-1] == (20, "20+") and all(t == f"{v:.0f}" for v, t in shown[:-1])
    task = ax.lines[0]
    assert list(task.get_xdata()) == [0, 1, 2, 4, 19] and list(task.get_ydata()) == [2, 1, 1, 1, 1]
    pooled = [ln for ln in ax.lines if list(ln.get_xdata()) == [20]]
    assert len(pooled) == 1 and pooled[0].get_ydata()[0] == 3        # 20, 29 and 68 repeaters
    assert "3 facts, 20–68" in [t.get_text() for t in ax.texts]
    structure.reach_figure(chains[chains["n_agents"] < 6], "a6.png")
    fig = drawn[1]
    assert all(float(t).is_integer() for t, _ in _labels(fig, fig.axes[0], "x"))
