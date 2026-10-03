"""A2 - does availability cause copying?  Two-way fixed effects on agent-rounds.

    Y_ir = beta * D_ir + alpha_agent + gamma_item + e,   SEs clustered by agent

Item FE absorb "same question, same answer"; agent FE absorb ability.  D
varies within agent (early rounds rarely exposed) and within item (first
mover vs follower).  Identifying assumption: an agent's position in the run
order (its assigned date) is unrelated to its ability.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import plotting
from ..plotting import plt

OUTCOMES = {
    "y_instant": ("Answered instantly (<= 2 s claimed)", None),
    "y_instant|known": ("Answered instantly, rounds with stated latency", "latency_known"),
    "y_consensus": ("Answer matches the consensus value", None),
}
# Treatment = consensus value public before the report (independent of the agent's own answer).
TREATMENTS = {"D_cons": "Consensus public before report", "D_cons_w60m": "Consensus public >= 1 h before"}


def _within_variation(df: pd.DataFrame, d: str) -> int:
    return int((df.groupby("agent")[d].nunique() > 1).sum())


MIN_SWITCHERS = 10  # agents whose exposure status varies across their rounds


def estimate(ex: pd.DataFrame, min_switchers: int = MIN_SWITCHERS) -> pd.DataFrame:
    try:
        import pyfixest as pf
    except ImportError:  # pragma: no cover
        pf = None
    rows = []
    for key, (ylab, subset) in OUTCOMES.items():
        y = key.split("|")[0]
        for d, dlab in TREATMENTS.items():
            df = ex.dropna(subset=[y]).copy()
            if subset:
                df = df[df[subset] == 1]
            df = df[df.groupby("agent")["agent"].transform("size") > 1]
            df = df[df.groupby("item_fe")["item_fe"].transform("size") > 1]
            row = {"outcome": key, "outcome_label": ylab, "treatment": d, "treatment_label": dlab,
                   "n": len(df), "agents": df["agent"].nunique(), "items": df["item_fe"].nunique(),
                   "agents_with_D_variation": _within_variation(df, d),
                   "mean_y_D0": df.loc[df[d] == 0, y].mean(), "mean_y_D1": df.loc[df[d] == 1, y].mean()}
            if row["agents_with_D_variation"] < min_switchers:
                row["note"] = f"not identified: only {row['agents_with_D_variation']} agents switch exposure status"
            elif pf is not None and len(df) > 20 and df[d].nunique() > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        fit = pf.feols(f"{y} ~ {d} | agent + item_fe", data=df, vcov={"CRV1": "agent"})
                        t = fit.tidy().loc[d]
                        row.update(beta=t["Estimate"], se=t["Std. Error"], p=t["Pr(>|t|)"],
                                   ci_lo=t["2.5%"], ci_hi=t["97.5%"])
                    except Exception as e:  # singular designs etc.
                        row["error"] = str(e)[:120]
            rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("beta", "se", "p", "ci_lo", "ci_hi", "note"):
        if c not in out:
            out[c] = np.nan
    return out


def figure(est: pd.DataFrame, path) -> str:
    plotting.setup()
    e = est.dropna(subset=["beta"]) if "beta" in est else est.iloc[0:0]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    labels = []
    for i, r in enumerate(e.itertuples()):
        c = plotting.SERIES[0] if r.treatment == "D_cons" else plotting.SERIES[1]
        ax.errorbar(r.beta, i, xerr=[[r.beta - r.ci_lo], [r.ci_hi - r.beta]], fmt="o", color=c,
                    capsize=3, lw=2, markersize=7)
        ax.text(r.ci_hi + 0.01, i, f"{r.beta:+.2f}  (n={r.n})", va="center", fontsize=8, color=plotting.INK_2)
        labels.append(f"{r.outcome_label}\n[{r.treatment_label}]")
    ax.axvline(0, color=plotting.NEUTRAL, lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect of exposure on the outcome (share, -1..1); agent + item FE; 95% CI clustered by agent")
    ax.set_title("Does availability cause copying?")
    fig.tight_layout()
    return plotting.save(fig, path)
