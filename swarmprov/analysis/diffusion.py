"""A3 - technique diffusion: time from first share to each agent's adoption.

At-risk set: agents active (posting) in the technique's families after its
first appearance (any task family for techniques without a family).  Adoption = the agent's first post mentioning/using it.
Agents who never adopt are censored at their last post.  Survival is a
plain Kaplan-Meier estimator (no lifelines dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..plotting import plt


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    times = np.unique(d[e == 1])
    s, rows = 1.0, [(0.0, 1.0)]
    for t in times:
        at_risk = (d >= t).sum()
        n_ev = ((d == t) & (e == 1)).sum()
        s *= 1 - n_ev / at_risk
        rows.append((float(t), s))
    return pd.DataFrame(rows, columns=["t", "survival"])


def analyze(tags: pd.DataFrame, techniques, agent_col: str) -> dict:
    tags = tags.sort_values("ts")
    out = {}
    for tech in techniques:
        hit = tags[tags["techniques"].fillna("").str.split(",").map(lambda l: tech.name in l)]
        if hit.empty:
            continue
        t0 = hit["ts"].iloc[0]
        origin = hit[agent_col].iloc[0]
        # at-risk: agents working the technique's families, or any task family for global ones
        pool = tags[tags["family"].isin(tech.families)] if tech.families else tags[tags["family"].fillna("") != ""]
        active = pool[pool["ts"] >= t0].groupby(agent_col)["ts"].max()
        active = active.drop(index=origin, errors="ignore")
        first_use = hit.groupby(agent_col)["ts"].min()
        dur, ev = [], []
        for a, last in active.items():
            if a in first_use.index:
                dur.append((first_use[a] - t0).total_seconds() / 3600)
                ev.append(1)
            else:
                dur.append((last - t0).total_seconds() / 3600)
                ev.append(0)
        dur, ev = np.array(dur), np.array(ev)
        km = kaplan_meier(dur, ev) if len(dur) else pd.DataFrame(columns=["t", "survival"])
        adopted = km[km["survival"] <= 0.5]
        out[tech.name] = {
            "description": tech.description, "first_seen": t0, "originator": origin,
            "at_risk": int(len(dur)), "adopters": int(ev.sum()),
            "adoption_share": float(ev.mean()) if len(ev) else float("nan"),
            "median_hours_to_adopt": float(adopted["t"].iloc[0]) if len(adopted) else float("nan"),
            "median_hours_among_adopters": float(np.median(dur[ev == 1])) if ev.sum() else float("nan"),
            "n_posts": int(len(hit)), "km": km,
        }
    return out


def figure(res: dict, path, families_of: dict | None = None, top: int = 3) -> str | None:
    """Two panels: techniques tied to a task family (small at-risk sets, fast
    clocks) and swarm-wide ones; at most ``top`` curves each, legend-labelled."""
    families_of = families_of or {}
    adopted = {n: r for n, r in res.items() if r["adopters"] > 0}
    if not adopted:
        return None
    groups = [("Task-specific techniques", [n for n in adopted if families_of.get(n)]),
              ("Swarm-wide techniques", [n for n in adopted if not families_of.get(n)])]
    groups = [(t, sorted(ns, key=lambda n: -adopted[n]["adopters"])[:top]) for t, ns in groups if ns]
    plotting.setup()
    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 4), squeeze=False)
    for ax, (title, names) in zip(axes[0], groups):
        for i, name in enumerate(names):
            r, km = adopted[name], adopted[name]["km"]
            ax.step(km["t"], 1 - km["survival"], where="post", color=plotting.SERIES[i],
                    label=f"{name} ({r['adopters']}/{r['at_risk']} agents)")
        ax.set_xlabel("Hours since the technique was first posted")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylim(0, None)
        ax.set_title(title)
        ax.legend(loc="upper left")
    axes[0][0].set_ylabel("Share of at-risk agents who adopted")
    fig.suptitle("Technique diffusion (Kaplan-Meier, non-adopters censored at last post)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return plotting.save(fig, path)
