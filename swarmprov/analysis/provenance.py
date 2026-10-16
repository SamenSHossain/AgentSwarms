"""A1 - provenance split: how many answers were already public before they were given?"""

from __future__ import annotations

import pandas as pd

from .. import plotting
from ..plotting import plt


def summarize(ex: pd.DataFrame, min_n: int = 10, wall_clock: bool = True) -> dict:
    """Without a wall clock (posts ordered by message position) the 10 min / 1 h shares and the head start
    in hours are None and the tables drop ``exposed_60m``: positions are not minutes."""
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
                .agg(n=("D", "size"), agents=("agent", "nunique"), exposed=("D", "mean"),
                     exposed_60m=("D_w60m", "mean"), self_prepared=("self_prepared", "mean"),
                     instant=("y_instant", "mean"))
                .query("n >= @min_n").sort_values("n", ascending=False))
    by_ep = (ex.dropna(subset=["episode"]).assign(episode=lambda d: d["episode"].astype(int))
               .groupby("episode").agg(n=("D", "size"), exposed=("D", "mean"),
                                       exposed_60m=("D_w60m", "mean"), instant=("y_instant", "mean")))
    if not wall_clock:
        out.update(share_exposed_10m=None, share_exposed_60m=None, median_gap_h=None)
        by_fam, by_ep = by_fam.drop(columns="exposed_60m"), by_ep.drop(columns="exposed_60m")
    out["by_family"] = by_fam
    out["by_episode"] = by_ep
    return out


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
