"""A1 - provenance split: how many answers were already public before they were given?"""

from __future__ import annotations

import pandas as pd

from .. import plotting
from ..plotting import plt


def summarize(ex: pd.DataFrame, min_n: int = 10) -> dict:
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
    out["by_family"] = by_fam
    out["by_episode"] = by_ep
    return out


def figure(ex: pd.DataFrame, res: dict, path) -> str:
    plotting.setup()
    fam = res["by_family"].head(14).iloc[::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 0.32 * len(fam) + 2.2),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    sub = ex[ex["family"].isin(fam.index)]
    parts = pd.DataFrame({
        "Independent (nothing public yet)": sub["D"].eq(0),
        "Public < 1 h before": sub["D"].eq(1) & sub["D_w60m"].eq(0),
        "Public >= 1 h before": sub["D_w60m"].eq(1),
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
    ax.set_title("Was the answer already on the wiki?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    ep = res["by_episode"]
    if len(ep):
        ax2.plot(ep.index, ep["exposed"], marker="o", color=plotting.SERIES[0], label="public before report")
        ax2.plot(ep.index, ep["exposed_60m"], marker="o", color=plotting.SERIES[1], label="public >= 1 h before")
        ax2.set_ylim(0, 1.02)
        ax2.set_xticks(ep.index)
        ax2.set_xticklabels([f"R{e}" for e in ep.index])
        ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax2.set_title("By round")
        ax2.legend(loc="lower right")
    fig.suptitle(f"{res['n_answers']} answers given, {res['n_independent']} with nothing public before the report",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=plotting.INK)
    fig.tight_layout()
    return plotting.save(fig, path)
