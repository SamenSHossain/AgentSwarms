"""A1 - provenance split: how many answers were already public before they were given?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..plotting import plt
from .uncertainty import cluster_bootstrap, wmean, wmedian

# exposure by required head start: thresholds in hours, and the answers a family needs for a row
HOURS = (0, 1 / 6, 0.5, 1, 3, 6, 12, 24, 48)
CURVE_MIN_N = 40
# instant-answer thresholds k (seconds) for the sweep
KS = (1, 2, 5, 10)
# the headline statistics with a bootstrap interval, and the exposure column each table share is the mean of
HEADLINE = ("share_exposed", "share_exposed_10m", "share_exposed_60m", "share_self_prepared", "median_gap_h")
TABLE_SHARES = {"exposed": "D", "exposed_60m": "D_w60m"}
CLUSTER = "item_fe"


def _headline(d: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    return np.column_stack([wmean(d["D"], w), wmean(d["D_w10m"], w), wmean(d["D_w60m"], w),
                            wmean(d["self_prepared"], w), wmedian(d["gap_s"] / 3600, w)])


def _table_ci(ex: pd.DataFrame, by: pd.Series, groups, cols: list[str], boot: int, seed: int) -> pd.DataFrame:
    """``<col>_lo`` / ``<col>_hi`` for each table share in ``cols``, per group, its items resampled within it."""
    def stat(d, w):
        return np.column_stack([wmean(d[TABLE_SHARES[c]], w) for c in cols])
    rows = {}
    for key in groups:
        lo, hi = cluster_bootstrap(ex[by == key], CLUSTER, stat, boot, seed)
        rows[key] = {f"{c}_{end}": v[i] for i, c in enumerate(cols) for end, v in (("lo", lo), ("hi", hi))}
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize(ex: pd.DataFrame, min_n: int = 10, wall_clock: bool = True, boot: int = 0, seed: int = 0) -> dict:
    """Without a wall clock (posts ordered by message position) the 10 min / 1 h shares and the head start
    in hours are None and the tables drop ``exposed_60m``: positions are not minutes.

    ``boot`` > 0 adds 95% percentile intervals from that many resamples of whole items (``item_fe``): ``<stat>_ci``
    = [lo, hi] for each ``HEADLINE`` statistic, and ``exposed_lo`` / ``exposed_hi`` (and the same for
    ``exposed_60m``) in the family and round tables, whose items are resampled within each row (none for a row
    with fewer than ``uncertainty.MIN_CLUSTERS`` items).  No interval is measured in hours without a wall clock."""
    n = len(ex)
    out = {
        "n_answers": n,
        "n_independent": int((ex["D"] == 0).sum()),
        "share_exposed": float(ex["D"].mean()) if n else float("nan"),
        "share_exposed_10m": float(ex["D_w10m"].mean()) if n else float("nan"),
        "share_exposed_60m": float(ex["D_w60m"].mean()) if n else float("nan"),
        "share_self_prepared": float(ex["self_prepared"].mean()) if n else float("nan"),
        "median_gap_h": float(ex["gap_s"].median() / 3600) if ex["gap_s"].notna().any() else float("nan"),
        "n_agents": int(ex["agent"].nunique()),
    }
    by_fam = (ex.groupby("family")
                .agg(n=("D", "size"), items=(CLUSTER, "nunique"), agents=("agent", "nunique"), exposed=("D", "mean"),
                     exposed_60m=("D_w60m", "mean"), self_prepared=("self_prepared", "mean"),
                     instant=("y_instant", "mean"))
                .query("n >= @min_n").sort_values("n", ascending=False))
    timed = ex.dropna(subset=["episode"]).assign(episode=lambda d: d["episode"].astype(int))
    by_ep = timed.groupby("episode").agg(n=("D", "size"), items=(CLUSTER, "nunique"), exposed=("D", "mean"),
                                         exposed_60m=("D_w60m", "mean"), instant=("y_instant", "mean"))
    if boot and n:
        lo, hi = cluster_bootstrap(ex, CLUSTER, _headline, boot, seed)
        out |= {f"{k}_ci": [float(a), float(b)] if a == a else None for k, a, b in zip(HEADLINE, lo, hi)}
        cols = list(TABLE_SHARES) if wall_clock else ["exposed"]
        by_fam = by_fam.join(_table_ci(ex, ex["family"], by_fam.index, cols, boot, seed))
        by_ep = by_ep.join(_table_ci(timed, timed["episode"], by_ep.index, cols, boot, seed))
    if not wall_clock:
        hours = ("share_exposed_10m", "share_exposed_60m", "median_gap_h")
        out.update(dict.fromkeys(hours) | (dict.fromkeys(f"{k}_ci" for k in hours) if boot and n else {}))
        by_fam, by_ep = by_fam.drop(columns="exposed_60m"), by_ep.drop(columns="exposed_60m")
    out["by_family"] = by_fam
    out["by_episode"] = by_ep
    return out


def hour_label(h: float) -> str:
    return "0" if h == 0 else f"{h * 60:g} min" if h < 1 else f"{h:g} h"


VISIBLE = "all answers, still-visible copies"


def _head_starts(ex: pd.DataFrame, min_n: int, visible: bool) -> list[tuple[str, pd.Series]]:
    """Each answer's head start in seconds (NaN when nothing was public before the report): every answer, the
    D_visible variant when ``visible``, then each family with ``min_n`` answers or more, largest first."""
    gap = ex["gap_s"].where(ex["D"] == 1)
    out = [("all answers", gap)]
    if visible:
        vis = pd.to_datetime(ex["t_report"], utc=True) - pd.to_datetime(ex["t_public_visible"], utc=True)
        out.append((VISIBLE, vis.dt.total_seconds().where(ex["D_visible"] == 1)))
    size = ex.groupby("family").size()
    return out + [(f, gap[ex["family"] == f]) for f in size[size >= min_n].sort_values(ascending=False, kind="stable").index]


def _share_at(g: pd.Series, hours) -> np.ndarray:
    return (g.to_numpy(dtype=float)[:, None] >= np.asarray(hours, dtype=float) * 3600).mean(axis=0)


def threshold_curve(ex: pd.DataFrame, hours=HOURS, min_n: int = CURVE_MIN_N, visible: bool = False) -> pd.DataFrame:
    """Exposure against the head start it requires: for each t in ``hours``, the share of answers whose value was
    public at least t before the report (D = 1 and ``gap_s`` >= t; t = 0 is D, 10 min D_w10m, 1 h D_w60m).  Rows:
    every answer, then each family with ``min_n`` answers or more, largest first; ``visible`` adds every answer
    under D_visible (only copies still visible at the report, from ``t_public_visible``).  t_report is an upper
    bound on when the question arrived, so this is a head start before the report, not before the question."""
    rows = {name: [len(g), *_share_at(g, hours)] for name, g in _head_starts(ex, min_n, visible) if len(g)}
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["n", *map(hour_label, hours)])
    return out.astype({"n": int}).rename_axis("answers")


def instant_sweep(ex: pd.DataFrame, ks=KS) -> dict:
    """The share of answers counted instant under each threshold k in ``ks`` (seconds).  Only an answer with a
    stated latency (``latency_s``, a number) moves: it is instant when the number is at most k.  The rest keep their
    class, instant from wording alone ("instantly", no number) or no latency stated."""
    lat = pd.to_numeric(ex["latency_s"], errors="coerce")
    fixed = int(((ex["latency_class"] == "instant") & lat.isna()).sum())
    n = len(ex)
    return {"n_answers": n, "n_stated_latency": int(lat.notna().sum()), "n_instant_by_wording": fixed, "ks": list(ks),
            "share_instant": [(fixed + int((lat <= k).sum())) / n if n else float("nan") for k in ks]}


def threshold_figure(ex: pd.DataFrame, path, hours=HOURS, min_n: int = CURVE_MIN_N, visible: bool = False,
                     cols: int = 4, max_panels: int = 12) -> str | None:
    """Small multiples of ``threshold_curve``: the share public at least t before the report as t grows, drawn
    exactly from the answers' head starts (a step at each one) and dotted at the table's thresholds; each family
    panel repeats every answer's curve in grey.  At most ``max_panels`` panels, families in table order."""
    if not len(ex):
        return None
    groups = _head_starts(ex, min_n, visible)
    t_max = max(hours) * 3600

    def exact(g: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        s = np.sort(g.dropna().to_numpy())
        s = s[s <= t_max]
        k = int(g.notna().sum())
        return (np.concatenate([[0], s, [t_max]]) / 3600,
                np.concatenate([[k], k - np.arange(1, len(s) + 1), [k - len(s)]]) / len(g))
    ref = groups[0][1]
    vis = dict(groups).get(VISIBLE)
    panels = [(n, g) for n, g in groups if n != VISIBLE][:max_panels]
    plotting.setup()
    cols = min(cols, len(panels))
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.5 * rows + 0.7), sharex=True, sharey=True,
                             squeeze=False)
    ticks = [h for h in (0, 0.5, 1, 3, 6, 12, 24, 48) if h <= max(hours)]
    for i, (ax, (name, g)) in enumerate(zip(axes.flat, panels)):
        if i:
            ax.step(*exact(ref), where="post", color=plotting.NEUTRAL, lw=1.2)
        ax.step(*exact(g), where="post", color=plotting.SERIES[0], lw=2, label=name)
        ax.plot(hours, _share_at(g, hours), "o", color=plotting.SERIES[0], markersize=4)
        if not i and vis is not None:
            ax.step(*exact(vis), where="post", color=plotting.SERIES[1], lw=1.5, ls="--", label="still-visible copies only")
            ax.plot(hours, _share_at(vis, hours), "o", color=plotting.SERIES[1], markersize=3)
            ax.legend(loc="lower left", fontsize=7.5)
        ax.set_title(f"{name} (n={len(g):,})", fontsize=9.5, loc="left")
        ax.set_xscale("symlog", linthresh=1, linscale=0.8)
        ax.set_xlim(0, max(hours))
        ax.set_ylim(0, 1.02)
        plotting.pct_axis(ax)
        if i + cols >= len(panels):   # no panel below: this one carries the axis
            ax.set_xticks(ticks, [hour_label(h).replace(" min", "m").replace(" h", "h") for h in ticks])
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("required head start t")
        if i % cols == 0:
            ax.set_ylabel("public ≥ t before report")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Exposure vs required head start: share of answers public at least t before the report"
                 + (" (grey: all answers)" if len(panels) > 1 else ""), x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=plotting.INK)
    fig.tight_layout()
    return plotting.save(fig, path)


def framework_split(ex: pd.DataFrame, tools: bool, task: bool) -> dict:
    """On a framework trace: answers split by a public copy first (D) and the agent's own tool output first
    (``n_own_lookup`` = ``own_lookup_evidence``, a lookup and no public copy), and answers whose value a task
    message gave first."""
    out = {}
    if tools:
        look, exposed = ex["t_own_lookup"].notna(), ex["D"] == 1
        out.update(n_own_lookup=int(ex["own_lookup_evidence"].sum()), n_exposed_with_own_lookup=int((exposed & look).sum()),
                   n_exposed_no_lookup=int((exposed & ~look).sum()), n_neither=int((~exposed & ~look).sum()))
    if task:
        given = ex["t_task"].notna()
        out.update(n_given_in_task=int(given.sum()), n_given_in_task_no_public_copy=int((given & (ex["D"] == 0)).sum()))
    return out


def figure(ex: pd.DataFrame, res: dict, path, wall_clock: bool = True) -> str:
    plotting.setup()
    fam = res["by_family"].head(14).iloc[::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 0.32 * len(fam) + 2.2),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    sub = ex[ex["family"].isin(fam.index)]
    parts = pd.DataFrame({
        "Independent (nothing public yet)": sub["D"].eq(0),
        "Public < 1 h before": sub["D"].eq(1) & sub["D_w60m"].eq(0),
        "Public >= 1 h before": sub["D_w60m"].eq(1),
    } if wall_clock else {
        "Independent (nothing public yet)": sub["D"].eq(0),
        "Public before the report": sub["D"].eq(1),
    }).groupby(sub["family"]).mean().reindex(fam.index)
    left = pd.Series(0.0, index=parts.index)
    for i, col in enumerate(parts.columns):
        ax.barh(parts.index, parts[col], left=left, color=plotting.SERIES[i], height=0.7,
                edgecolor=plotting.SURFACE, linewidth=2, label=col)
        left += parts[col]
    for y, (f, row) in enumerate(fam.iterrows()):
        ax.text(1.01, y, f"n={int(row['n'])}", va="center", fontsize=8, color=plotting.INK_2)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_title("Was the answer already public?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    ep = res["by_episode"]
    if len(ep):
        ax2.plot(ep.index, ep["exposed"], marker="o", color=plotting.SERIES[0], label="public before report")
        if wall_clock:
            ax2.plot(ep.index, ep["exposed_60m"], marker="o", color=plotting.SERIES[1], label="public >= 1 h before")
        ax2.set_ylim(0, 1.02)
        ax2.set_xticks(ep.index)
        ax2.set_xticklabels([f"R{e}" for e in ep.index])
        plotting.pct_axis(ax2)
        ax2.set_title("By round")
        ax2.legend(loc="lower right")
    fig.suptitle(f"{res['n_answers']} answers given, {res['n_independent']} with nothing public before the report",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=plotting.INK)
    fig.tight_layout()
    return plotting.save(fig, path)
