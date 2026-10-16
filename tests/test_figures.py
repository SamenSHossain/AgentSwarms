"""Figures say what their tables say: asserted on the drawn data (line ends, tick labels, texts), not on pixels."""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest

from swarmprov import plotting
from swarmprov.analysis import causal, diffusion, errors, provenance, structure

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


def _exact(shown):
    return all(abs(float(t.rstrip("%")) - v * 100) < 1e-9 for v, t in shown)


def test_pct_axis_places_and_prints_its_own_ticks():
    plotting.setup()
    fig, ax = plotting.plt.subplots(figsize=(6.2, 4))
    ax.set_ylim(0, 0.2)
    plotting.pct_axis(ax)                                  # its own locator: no set_yticks
    shown = _labels(fig, ax)
    assert 0.025 in [round(v, 6) for v, _ in shown] and "2.5%" in dict(shown).values() and _exact(shown)
    for hi in (0.12, 0.25):                                # MaxNLocator's default steps give 1.5% and 3% here
        ax.set_ylim(0, hi)
        steps = np.diff([v for v, _ in _labels(fig, ax)])
        assert {round(d / 10 ** np.floor(np.log10(d)), 6) for d in steps} <= {1, 2, 2.5, 5}
    plotting.plt.close(fig)


def test_a3_percent_ticks_are_not_rounded(drawn):
    km = pd.DataFrame({"t": [0.0, 1.0, 2.0], "survival": [1.0, 0.9, 0.81]})
    res = {"wide": {"adopters": 3, "at_risk": 20, "km": km}}
    diffusion.figure(res, "a3.png")
    fig = drawn[0]
    shown = _labels(fig, fig.axes[0])                      # the ticks the figure placed
    assert {"2.5%", "17.5%"} <= set(dict(shown).values()) and _exact(shown)   # a rounding formatter prints 2%, 18%


def test_a1_by_round_axis_is_in_percent(drawn):
    ex = pd.DataFrame({"family": ["f"] * 4, "D": [0, 1, 1, 0], "D_w60m": [0, 0, 1, 0]})
    res = {"by_family": pd.DataFrame({"n": [4]}, index=["f"]), "n_answers": 4, "n_independent": 2,
           "by_episode": pd.DataFrame({"exposed": [0.2, 0.5], "exposed_60m": [0.1, 0.25]}, index=[1, 2])}
    provenance.figure(ex, res, "a1.png")
    fig = drawn[0]
    shown = _labels(fig, fig.axes[1])
    assert len(shown) >= 3 and all(t.endswith("%") for _, t in shown) and _exact(shown)


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
    ends = {ln.get_label().split(" (")[0]: ln.get_ydata()[-1] for ln in ax.lines if ln.get_drawstyle() == "steps-post"}
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
    assert f"1 earlier carrier before {left:%m-%d %H:%M}, at {T0 - pd.Timedelta(hours=72):%m-%d %H:%M}" in legend
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
    # 0% segments clear the bottom spine, and every segment is drawn over the 50% line
    half = next(ln for ln in axb.lines if list(ln.get_ydata()) == [0.5, 0.5])
    assert axb.get_ylim()[0] < 0 and all(c.get_zorder() > half.get_zorder() for c in axb.collections)
    # no one-carrier bin on the axis: the only sparse bin is the stray's, left of it
    stray_only = [(-72, "A")] + [(0.1 + i / 6, "A") for i in range(6)] + [(3.5 + i / 6, "B") for i in range(12)]
    *_, legend, _, axb = _a4(drawn, stray_only)
    assert any(c.get_alpha() == 0.3 and len(c.get_segments()) for c in axb.collections)
    assert not any(t.startswith("share in a bin with fewer than") for t in legend)


def test_a4_draws_a_winner_that_appeared_fourth(drawn):
    spec = [(0, "A"), (0.1, "B"), (0.2, "C")] + [(0.5 + i / 6, "A") for i in range(6)] + \
           [(1, "B"), (1.2, "C")] + [(3.5 + i / 6, "D") for i in range(12)]
    s, ends, legend, ax, _ = _a4(drawn, spec)
    assert s["variants"] == ["A", "B", "C", "D"] and s["winner"] == "D"
    assert [ln.get_label() for ln in ax.lines if ln.get_drawstyle() == "steps-post"] == [
        "A (7 agents)", "B (2 agents)", "D (12 agents), winner"]
    assert ends == {"A": 7, "B": 2, "D": 12}


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


def test_a4_tie_draws_the_tied_variants_and_their_shares(drawn):
    # A, B and C come first with one agent each; D and E (7 each) tie 3-3 among the late carriers
    tie = [(0, "A"), (0.1, "B"), (0.2, "C")] + [(1 + i / 6, "D") for i in range(4)] + \
          [(1.05 + i / 6, "E") for i in range(4)] + [(3 + i / 3, v) for i in range(3) for v in "DE"]
    s, ends, legend, _, axb = _a4(drawn, tie)
    assert s["winner"] is None and s["tail_counts"] == {"D": 3, "E": 3}
    assert errors.plotted_variants(s) == ["A", "D", "E"] and ends == {"A": 1, "D": 7, "E": 7}
    assert "A (1 agent)" in legend and "no winner: the late carriers tie" in legend
    new = s["new_by_bin"].sum(axis=1)
    rgb = plotting.matplotlib.colors.to_rgb
    for i, v in enumerate(["A", "D", "E"]):                 # each plotted variant's share, in its colour
        ys = [seg[0][1] for c in axb.collections if len(c.get_segments()) and tuple(c.get_colors()[0][:3]) ==
              rgb(plotting.SERIES[i]) for seg in c.get_segments()]
        assert sorted(ys) == sorted(s["share_by_bin"][v][new > 0])


def test_a2_draws_every_row_and_says_which_are_not_identified(drawn):
    est = pd.DataFrame([
        {"outcome_label": "Instant", "treatment": "D_cons", "treatment_label": "public", "beta": -0.07, "ci_lo": -0.2,
         "ci_hi": 0.06, "n": 787, "agents_with_D_variation": 37, "note": np.nan},
        {"outcome_label": "Instant, known", "treatment": "D_cons", "treatment_label": "public", "beta": np.nan,
         "ci_lo": np.nan, "ci_hi": np.nan, "n": 328, "agents_with_D_variation": 4,
         "note": "not identified: only 4 agents switch exposure status"},
        {"outcome_label": "Instant, known", "treatment": "D_cons_w60m", "treatment_label": ">= 1 h", "beta": 0.001,
         "ci_lo": -0.007, "ci_hi": 0.009, "n": 328, "agents_with_D_variation": 10, "note": np.nan},
        {"outcome_label": "Matches consensus", "treatment": "D_cons", "treatment_label": "public", "beta": np.nan,
         "ci_lo": np.nan, "ci_hi": np.nan, "n": 900, "agents_with_D_variation": 40, "note": np.nan}])   # fit failed
    causal.figure(est, "a2.png")
    fig = drawn[0]
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "not identified (4 agents switch)" in texts and "not estimated" in texts
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "Instant\n[public], 37 switchers", "Instant, known\n[public], 4 switchers",
        "Instant, known\n[>= 1 h], 10 switchers", "Matches consensus\n[public], 40 switchers"]
    bars = {round(c.lines[0].get_xdata()[0], 3): c.lines[0].get_color() for c in ax.containers}
    # no bar for the unidentified row, and 10 switchers (under 20) is grey
    assert bars == {-0.07: plotting.SERIES[0], 0.001: plotting.NEUTRAL}
    assert f"under {causal.MIN_SWITCHERS} switchers" in ax.get_title("left")


def test_a5_lorenz_marks_where_the_top_sources_begin(drawn):
    outdeg = pd.Series([40, 30, 20, 10] + [2] * 8 + [1] * 8, index=[f"s{i}" for i in range(20)])
    share = outdeg.head(10).sum() / outdeg.sum()
    res = {"outdeg": outdeg, "gini_exposure_outdegree": 0.6, "chain_depth": pd.Series(dtype=int)}
    structure.figure(res, pd.DataFrame(), "a5.png")
    ax = drawn[0].axes[0]
    curve, dot = ax.lines[0], next(ln for ln in ax.lines if ln.get_label().startswith("top 10 of 20"))
    x, y = dot.get_xdata()[0], dot.get_ydata()[0]
    assert (x, y) == (0.5, pytest.approx(1 - share))
    assert y == pytest.approx(np.interp(x, curve.get_xdata(), curve.get_ydata()))   # on the curve
    assert f"{share:.0%} of exposed" in dot.get_label() and "first to post the value" in dot.get_label()
    structure.figure(res, pd.DataFrame(), "a5.png", top=5)          # any top, from the out-degrees
    dot = next(ln for ln in drawn[1].axes[0].lines if ln.get_label().startswith("top 5 of 20"))
    assert (dot.get_xdata()[0], dot.get_ydata()[0]) == (0.75, pytest.approx(1 - 102 / 124))


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
    for cap, ticks in ((12, ["0", "5", "10", "12+"]), (22, ["0", "5", "10", "15", "20", "22+"])):
        structure.reach_figure(chains, "a6.png", cap=cap)
        assert [t for _, t in _labels(drawn[-1], drawn[-1].axes[0], "x")] == ticks
    structure.reach_figure(chains[chains["n_agents"] <= 21], "a6.png")       # one fact with exactly 20
    assert "1 fact, 20" in [t.get_text() for t in drawn[-1].axes[0].texts]
