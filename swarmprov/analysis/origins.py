"""A5 - origins: what did the source of each copied answer do on the item itself?

The source's own answer round on the same item (from the same exposure table) says
whether it **computed** the value (answered it with nothing matching public, D == 0),
was itself exposed and **relayed** it (D == 1), or was matched on a different value
(**dissent**: its own, or the consensus when it stated none).  A source with no answer
round on the item reads **no answer round observed**: an agent that only published values (an answer list, a sequence scout),
or an answer the extractor missed, or an identity split.  Being the first public
source is not proof of being read.  A self-originated answer (its agent posted the value
before any other agent did) credits no source, so it has no row here.
"""

from __future__ import annotations

import pandas as pd

from ..exposure import match_key

LABELS = ("computed", "no answer round observed", "dissent", "relayed")
# the exposed answers no source is credited with, a row of their own in A5's origins table
SELF = "self-originated (the answering agent posted it first)"
# attribution rule: the source column and the time of the source's post
SOURCE_TIME = {"src_agent": "t_public", "src_latest": "t_latest"}
# roots where the walk stops because the source had posted the value no later than the post it is credited to.
# POSTED_FIRST: no later than any other carrier, so it counts as exposed only because another agent posted the
# value between its own post and its report.  CUT: an earlier carrier existed, but the latest rule credits it to
# one that posted after it (the first rule cannot).
POSTED_FIRST = "posted before its own source"
CUT = "chain cut (its latest source posted after it)"


def _rounds(ex: pd.DataFrame) -> dict:
    keys = [match_key(v, c) for v, c in zip(ex["value_norm"], ex["consensus_value"])]
    return {(r.agent, r.family, r.item): (r, k) for r, k in zip(ex.itertuples(index=False), keys)}


def origin_status(ex: pd.DataFrame, src_col: str = "src_agent") -> pd.DataFrame:
    """One row per credited exposed answer (``exposure.credited``): its source under ``src_col`` and that source's
    origin."""
    rounds = _rounds(ex)
    rows = []
    for r, key in rounds.values():
        if r.D != 1 or r.self_originated:
            continue
        src = getattr(r, src_col)
        own = rounds.get((src, r.family, r.item))
        if own is None:
            origin = "no answer round observed"
        elif own[1] != key:
            origin = "dissent"
        else:
            origin = "relayed" if own[0].D == 1 else "computed"
        rows.append({"claim_id": r.claim_id, "agent": r.agent, "family": r.family, "item": r.item,
                     "source": src, "origin": origin})
    return pd.DataFrame(rows, columns=["claim_id", "agent", "family", "item", "source", "origin"])


def trace_root(ex: pd.DataFrame, src_col: str = "src_agent") -> pd.DataFrame:
    """``origin_status`` plus the root: a relayed source is followed through its own source (same
    rule) until one is not relayed.  ``depth`` counts the hops (1 = the source itself).  The walk
    also stops at a source that had posted the value no later than the post it is credited to,
    or when the sources lead back to an agent already on the path (the answering agent itself,
    which had posted the value before them); that root reads POSTED_FIRST if it posted no later
    than any other carrier, else CUT.  Each step moves to an earlier first post of the value, so
    the walk ends."""
    og, t_col = origin_status(ex, src_col), SOURCE_TIME[src_col]
    rounds, label = _rounds(ex), dict(zip(zip(og["agent"], og["family"], og["item"]), og["origin"]))
    roots = []
    for o in og.itertuples(index=False):
        src, lab, depth, seen = o.source, o.origin, 1, {o.agent}
        while lab == "relayed":
            s = rounds[(src, o.family, o.item)][0]
            own = s.t_self_prepared if pd.notna(s.t_self_prepared) else None
            if src in seen or (own is not None and own <= getattr(s, t_col)):
                lab = POSTED_FIRST if own is not None and own <= s.t_public else CUT
                break
            seen.add(src)
            src, depth = getattr(s, src_col), depth + 1
            lab = label[(s.agent, o.family, o.item)]
        roots.append((src, lab, depth))
    return og.assign(root=[r[0] for r in roots], root_origin=[r[1] for r in roots], depth=[r[2] for r in roots])


def by_source(og: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    """Per source: answers supplied, items, the origins of those answers and the answer rounds it
    has in the run on any item (none suggests a publisher; some suggests a miss on this item)."""
    g = og.groupby("source")
    out = pd.DataFrame({"answers": g.size(),
                        "items": og.drop_duplicates(["source", "family", "item"]).groupby("source").size(),
                        "origin": g["origin"].agg(lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items())),
                        "answer rounds (any item)": ex.groupby("agent").size().reindex(g.size().index).fillna(0).astype(int)})
    return out.sort_values("answers", ascending=False)
