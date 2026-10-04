"""Provenance graph: agents as nodes, relays and citations as edges.

Edge kinds
* ``relay``     inferred: for each (family, item, value), a new carrier's parent
                is the latest earlier carrier on the same channel.  Gives relay
                trees whose depth is the number of hops from the originator.
* ``relay_xchannel``  the carrier's channel had no earlier carrier, so the hop
                is attributed to the originator (source unknown).
* ``exposure``  the agent-round's first public source -> the answering agent.
* ``citation``  explicit references: @Name, "Name's report", "per Mar16".
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np
import pandas as pd

from .identity import cohort_token

# two 1 s clocks: a hop shorter than this cannot be ordered reliably
CLOCK_RES_S = 2.0


def relay_edges(mentions: pd.DataFrame, agent_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relay trees per fact.  A new carrier's parent is the latest earlier
    carrier *on the same channel* (agents copy from the page they then write
    on); if nobody carried it there before, the parent is the originator and
    the hop is marked cross-channel."""
    edges, chains = [], []
    has_channel = "channel" in mentions.columns
    mentions = mentions.sort_values("ts")
    keys = ["family", "item", "value_key"]
    n_ag = mentions.groupby(keys)[agent_col].transform("nunique")
    # facts only one agent ever carried: no edges, a single chain row each
    solo = mentions[n_ag == 1].groupby(keys, as_index=False).agg(
        originator=(agent_col, "first"), t0=("ts", "min"), t1=("ts", "max"))
    if len(solo):
        chains.append(pd.DataFrame({"family": solo["family"], "item": solo["item"], "value_key": solo["value_key"],
                                    "originator": solo["originator"], "t0": solo["t0"], "n_agents": 1,
                                    "max_depth": 0, "span_s": (solo["t1"] - solo["t0"]).dt.total_seconds()}))
    multi_rows = []
    for (fam, item, vk), g in mentions[n_ag > 1].groupby(keys, sort=False):
        origin = getattr(g.iloc[0], agent_col)
        depth = {origin: 0}
        latest_on: dict[str, tuple[str, pd.Timestamp]] = {}
        t_origin = g["ts"].iloc[0]
        for r in g.itertuples(index=False):
            a, ch = getattr(r, agent_col), (r.channel if has_channel else "")
            if a not in depth:
                prev = latest_on.get(ch)
                if prev and prev[0] != a:
                    src, src_ts, kind = prev[0], prev[1], "relay"
                else:
                    src, src_ts, kind = origin, t_origin, "relay_xchannel"
                depth[a] = depth[src] + 1
                dt = (r.ts - src_ts).total_seconds()
                edges.append({"src": src, "dst": a, "family": fam, "item": item, "value_norm": vk,
                              "kind": kind, "dt_s": dt, "event_id": r.event_id,
                              "within_clock_res": dt <= CLOCK_RES_S})  # direction not reliable inside the clock resolution
            latest_on[ch] = (a, r.ts)
        multi_rows.append({"family": fam, "item": item, "value_key": vk, "originator": origin,
                           "t0": t_origin, "n_agents": len(depth), "max_depth": max(depth.values()),
                           "span_s": (g["ts"].iloc[-1] - t_origin).total_seconds()})
    chains.append(pd.DataFrame(multi_rows))
    chains = [c for c in chains if len(c)]
    return pd.DataFrame(edges), (pd.concat(chains, ignore_index=True) if chains else pd.DataFrame())


def exposure_edges(exp: pd.DataFrame) -> pd.DataFrame:
    e = exp[exp["D"] == 1]
    return pd.DataFrame({
        "src": e["src_agent"], "dst": e["agent"], "family": e["family"], "item": e["item"],
        "value_norm": e["value_norm"].fillna(e["consensus_value"]), "kind": "exposure",
        "dt_s": e["gap_s"], "event_id": e["claim_id"],
    })


def citation_edges(tags: pd.DataFrame, agents: pd.DataFrame, agent_col: str, cohort_rx: str) -> pd.DataFrame:
    by_name = {str(a).lower(): row[agent_col] for a, row in agents.set_index("author_raw").iterrows()}
    by_cohort_fam: dict[tuple[str, str], str] = {}
    for m in agents.itertuples(index=False):
        fam = m.agent_merged.split("|", 1)[1] if "|" in str(m.agent_merged) else ""
        if m.cohort and fam:
            by_cohort_fam.setdefault((m.cohort, fam), getattr(m, agent_col))
    rx = re.compile(cohort_rx)
    rows = []
    for t in tags[tags["cites"].fillna("") != ""].itertuples(index=False):
        for tok in str(t.cites).split(","):
            dst_agent = getattr(t, agent_col)
            src = by_name.get(tok.lower())
            if src is None:
                c = cohort_token(tok, rx)
                if c:
                    src = by_cohort_fam.get((c, t.family))
            if src is not None and src != dst_agent:
                rows.append({"src": src, "dst": dst_agent, "family": t.family, "item": None,
                             "value_norm": None, "kind": "citation", "dt_s": np.nan, "event_id": t.event_id})
    return pd.DataFrame(rows)


def to_networkx(edges: pd.DataFrame, kinds=("relay", "relay_xchannel", "citation")) -> nx.DiGraph:
    G = nx.DiGraph()
    e = edges[edges["kind"].isin(kinds)]
    for (s, d), g in e.groupby(["src", "dst"]):
        G.add_edge(s, d, weight=len(g), kinds=",".join(sorted(set(g["kind"]))))
    return G


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
