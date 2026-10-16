"""A3 - technique diffusion: time from first share to each agent's adoption.

At-risk set: agents active (posting) in the technique's families after its
first appearance (any task family for techniques without a family; every agent
when no post has a family at all).  Adoption = the agent's first post mentioning/using it.
Agents who never adopt are censored at their last post.  Survival is a
plain Kaplan-Meier estimator (no lifelines dependency), and the median time to
adopt carries Greenwood's 95% interval where it is reached
(``uncertainty.km_greenwood``).  Posts without a timestamp take no part.
Each technique's result keeps one row per agent (``ADOPTION_COLS``): the KM
inputs are exactly its at-risk rows.

``contagion`` asks whether adopters adopted where the technique was already
posted: the share of adoptions on a page another agent had already posted it
on, against each adopter's own posts in the adoption's family, with a
within-agent permutation p-value.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import plotting
from ..plotting import plt
from .uncertainty import km_greenwood, km_median


ADOPTION_COLS = ["technique", "agent", "in_at_risk", "is_originator", "t_first_use", "hours_from_first_seen", "adopted",
                 "censored_at_h"]
MIN_ADOPTERS = 10   # fewer adopters in the contagion test get a small-n warning
ALPHA = 0.05        # the p-value under which a share above its baseline reads "consistent with contagion"


def familyless(tags: pd.DataFrame) -> bool:
    """No post has a task family, so the at-risk set cannot be restricted to one."""
    return not tags["family"].fillna("").astype(bool).any()


def _timed(tags: pd.DataFrame) -> pd.DataFrame:
    return tags[tags["ts"].notna()].sort_values(["ts", "event_id"], kind="stable")


def at_risk(tags: pd.DataFrame, tech, agent_col: str, pooled: bool) -> tuple[pd.DataFrame, pd.Series] | None:
    """The technique's posts (``tags``: timed, in time order) and each at-risk agent's last post: agents posting in its
    families from its first post on (any task family for a technique without families; everyone when ``pooled``),
    the originator left out.  None when no post uses it."""
    hit = tags[tags["techniques"].fillna("").str.split(",").map(lambda l: tech.name in l)]
    if hit.empty:
        return None
    pool = tags if pooled else (tags[tags["family"].isin(tech.families)] if tech.families
                                else tags[tags["family"].fillna("") != ""])
    last = pool[pool["ts"] >= hit["ts"].iloc[0]].groupby(agent_col)["ts"].max()
    return hit, last.drop(index=hit[agent_col].iloc[0], errors="ignore")


def analyze(tags: pd.DataFrame, techniques, agent_col: str) -> dict:
    """Per technique: the A3 counts, the Kaplan-Meier curve (``km``) and ``adoptions``, one row per at-risk agent and
    per other user of the technique (``in_at_risk`` False: the originator, or an agent with no post in the pool after
    the first appearance).  ``hours_from_first_seen`` is a user's first use, ``censored_at_h`` a non-adopter's last
    post in the pool; the curve is drawn from exactly the at-risk rows."""
    tags = _timed(tags)
    pooled = familyless(tags)
    out = {}
    for tech in techniques:
        risk = at_risk(tags, tech, agent_col, pooled)
        if risk is None:
            continue
        hit, active = risk
        t0, origin = hit["ts"].iloc[0], hit[agent_col].iloc[0]
        first_use = hit.groupby(agent_col)["ts"].min()

        def hours(t):
            return (t - t0).total_seconds() / 3600
        rows = []
        for a, last in active.items():
            used = a in first_use.index
            rows.append({"agent": a, "in_at_risk": True, "is_originator": False, "t_first_use": first_use.get(a),
                         "hours_from_first_seen": hours(first_use[a]) if used else np.nan, "adopted": used,
                         "censored_at_h": np.nan if used else hours(last)})
        rows += [{"agent": a, "in_at_risk": False, "is_originator": a == origin, "t_first_use": t,
                  "hours_from_first_seen": hours(t), "adopted": True, "censored_at_h": np.nan}
                 for a, t in first_use.items() if a not in active.index]
        ad = (pd.DataFrame(rows).assign(technique=tech.name)[ADOPTION_COLS]
              .sort_values(["t_first_use", "agent"], na_position="last", kind="stable").reset_index(drop=True))
        risk = ad[ad["in_at_risk"]]
        dur = np.where(risk["adopted"], risk["hours_from_first_seen"], risk["censored_at_h"]).astype(float)
        ev = risk["adopted"].to_numpy(dtype=int)
        km = km_greenwood(dur, ev)
        median, lo, hi = km_median(km)
        out[tech.name] = {
            "description": tech.description, "first_seen": t0, "originator": origin,
            "at_risk": int(len(dur)), "adopters": int(ev.sum()),
            "adoption_share": float(ev.mean()) if len(ev) else float("nan"),
            "median_hours_to_adopt": median,
            # Greenwood's interval, only where the median is reached; an upper bound the band never reaches is NaN
            "median_hours_to_adopt_lo": lo if median == median else float("nan"),
            "median_hours_to_adopt_hi": hi if median == median else float("nan"),
            "median_hours_among_adopters": float(np.median(dur[ev == 1])) if ev.sum() else float("nan"),
            "n_posts": int(len(hit)), "km": km, "adoptions": ad,
        }
    return out


def permutation_p(probs, k: int) -> float:
    """P(K >= k) when each adopter's adoption is a uniform draw from its eligible posts, a share ``probs[i]`` of them
    on a carrying page: the exact within-agent permutation distribution (a sum of independent Bernoullis)."""
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return float(min(1.0, dist[k:].sum()))


def _carrying(p: pd.DataFrame, first: pd.DataFrame, agent_col: str) -> np.ndarray:
    """Whether another agent had already used the technique on each post's page, strictly before the post (``first``:
    per page, its first user and that time, ``ts``, and the time a second agent first used it there, ``ts2``)."""
    f = first.reindex(p["channel"]).reset_index(drop=True)
    other = f["ts"].where(f[agent_col].to_numpy() != p[agent_col].to_numpy(), f["ts2"])
    return (p["ts"].reset_index(drop=True) > other).to_numpy(dtype=bool)


def _names_only(texts: pd.Series, pattern: str, channels) -> pd.Series:
    """Posts whose text matches ``pattern`` only through the name of a page that matches it too (a link to
    ``AgentLinksZZZ2`` under the pattern ``ZZZ``): with every such name taken out, nothing matches."""
    rx = re.compile(pattern, re.I)
    names = {n for c in channels if rx.search(str(c)) for n in (str(c), str(c).rsplit("/", 1)[-1])}
    if not names:
        return pd.Series(False, index=texts.index)
    strip = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)), re.I)
    return texts.fillna("").map(lambda t: bool(rx.search(t)) and not rx.search(strip.sub(" ", t)))


def contagion(tags: pd.DataFrame, events: pd.DataFrame, techniques, agent_col: str) -> dict:
    """Per technique, over A3's at-risk adopters (``at_risk``): did they adopt on a page that already carried it?

    A page carries the technique for an agent once another agent has posted it there; the adoption post (the
    agent's first use) never counts toward its own page.  ``share_on_carrying`` is the share of adoptions on a
    carrying page; ``baseline`` the mean, over adopters, of the share of each one's posts from the first appearance
    on, in the adoption post's family, that landed on a page carrying it at the time (the adoption post included:
    within family, so moving onto a task's pages is not read as contagion).  ``p`` is the exact within-agent
    permutation p-value of that many adoptions or more on carrying pages, were the adoption a uniform draw from each
    agent's eligible posts.  ``agents`` holds the per-adopter rows, with ``t_exposed``, the adopter's first post on
    a carrying page strictly before adopting.  A technique with a post that matches its pattern only by naming a page
    (``_names_only``) is not tested (``excluded``): being on that page can produce the match."""
    ev = events.drop_duplicates("event_id").set_index("event_id")
    text, channels = ev["text"], ev["channel"].dropna().unique()
    tags = _timed(tags)
    tags = tags.assign(channel=tags["event_id"].map(ev["channel"]), family=tags["family"].fillna(""))
    pooled = familyless(tags)
    out = {}
    for tech in techniques:
        risk = at_risk(tags, tech, agent_col, pooled)
        if risk is None:
            continue
        hit, active = risk
        n_named = int(_names_only(hit["event_id"].map(text), tech.pattern, channels).sum())
        res = {"n_carrying_pages": int(hit["channel"].nunique()), "n_posts_page_name_only": n_named, "excluded": None}
        if n_named:
            out[tech.name] = res | {"excluded": f"{n_named} of its {len(hit)} posts match its pattern only by naming a "
                                                "page, so a post on that page can match because of where it is"}
            continue
        firsts = hit[hit["channel"].notna()].drop_duplicates(["channel", agent_col])
        k = firsts.groupby("channel").cumcount()
        first = firsts[k == 0].set_index("channel")[[agent_col, "ts"]]
        first["ts2"] = firsts[k == 1].set_index("channel")["ts"].reindex(first.index)
        adopt = hit.drop_duplicates(agent_col)
        adopt = adopt[adopt[agent_col].isin(active.index)].set_index(agent_col)
        mine = tags[tags[agent_col].isin(adopt.index)]
        own = adopt.reindex(mine[agent_col]).set_index(mine.index)   # each post beside its agent's adoption post
        el = mine[(mine["ts"] >= hit["ts"].iloc[0]) & (mine["family"] == own["family"])]
        base = pd.Series(_carrying(el, first, agent_col), index=el.index).groupby(el[agent_col]).agg(["mean", "size"])
        on = pd.Series(_carrying(adopt.reset_index(), first, agent_col), index=adopt.index)
        pre = mine[mine["ts"] < own["ts"]]
        pre = pre[_carrying(pre, first, agent_col)]
        agents = pd.DataFrame({"technique": tech.name, "agent": adopt.index, "adoption_channel": adopt["channel"],
                               "adopted_on_carrying_page": on, "n_eligible_posts": base["size"].reindex(adopt.index),
                               "baseline_share": base["mean"].reindex(adopt.index),
                               "t_exposed": pre.groupby(agent_col)["ts"].min().reindex(adopt.index)}).reset_index(drop=True)
        n, k_on = len(agents), int(on.sum())
        out[tech.name] = res | {
            "n_adopters": n, "n_on_carrying": k_on, "share_on_carrying": k_on / n if n else float("nan"),
            "baseline": float(agents["baseline_share"].mean()) if n else float("nan"),
            "p": permutation_p(agents["baseline_share"], k_on) if n else float("nan"),
            "n_exposed_before": int(agents["t_exposed"].notna().sum()), "agents": agents}
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
        ax.set_ylim(0, None)
        plotting.pct_axis(ax)
        ax.set_title(title)
        ax.legend(loc="upper left")
    axes[0][0].set_ylabel("Share of at-risk agents who adopted")
    fig.suptitle("Technique diffusion (Kaplan-Meier, non-adopters censored at last post)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return plotting.save(fig, path)
