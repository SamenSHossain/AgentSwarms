"""Provenance graph: agents as nodes, relays and citations as edges.

Edge kinds
* ``relay``     inferred: for each (family, item, value), a new carrier's parent
                is the latest earlier carrier on the same channel, at any lag
                (A6's sensitivity table limits it to a window W).  Gives relay
                trees whose depth is the number of hops from the originator.
* ``relay_xchannel``  the carrier's channel had no earlier carrier, so the hop
                is attributed to the originator (source unknown).
* ``exposure``  the agent-round's first public source -> the answering agent.
* ``citation``  explicit references: @Name, "Name's report", "per Mar16".
* ``reply``     an explicit reply link (a post's ``parent_id`` is another agent's
                post), built only when the source has threading.  An interaction,
                not a knowledge transfer: A5 counts it for brokers, never for supply.

``event_id`` is the destination's post and ``src_event_id`` the source's (see
schema.TABLES['edges']); ``value_norm`` is the fact's value_key for every kind.
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np
import pandas as pd

from .claims import PRIVATE_KINDS
from .exposure import match_key
from .identity import cohort_token

# two 1 s clocks: a hop shorter than this cannot be ordered reliably
CLOCK_RES_S = 2.0
KINDS = ("relay", "relay_xchannel", "exposure", "citation", "reply")
# what an edge's weight counts: a cross-page hop's source is unknown, and a reply is an interaction
TRANSFER_KINDS = ("relay", "exposure", "citation")
SHORT = {"relay": "n_relay", "relay_xchannel": "n_xchannel", "exposure": "n_exposure", "citation": "n_citation",
         "reply": "n_reply"}


def relay_edges(mentions: pd.DataFrame, agent_col: str, clock_res_s: float | None = CLOCK_RES_S,
                max_gap_s: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relay trees per fact.  A new carrier's parent is the latest earlier
    carrier *on the same channel* (agents copy from the page they then write
    on); if nobody carried it there before, the parent is the originator and
    the hop is marked cross-channel.  Mentions without a timestamp have no place
    in the order and are left out; ``clock_res_s=None`` (a sequence clock, where
    there is no clock error) leaves ``within_clock_res`` empty.

    ``max_gap_s`` (the relay window W; None = no limit, the graph's own edges): a parent that posted the fact more
    than W before the carrier gives no edge, and the carrier is left unattributed, a root of depth 0 that later
    carriers can still descend from.  A chain's ``n_agents`` does not change, so the unattributed carriers are its
    ``n_agents - 1`` minus its edges."""
    edges, chains = [], []
    has_channel = "channel" in mentions.columns
    by = ["ts", "event_id"] if "event_id" in mentions.columns else ["ts"]
    mentions = mentions[mentions["ts"].notna()].sort_values(by, kind="stable")   # same-second mentions: deterministic parent
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
        latest_on: dict[str, tuple[str, pd.Timestamp, str]] = {}
        t_origin, origin_ev = g["ts"].iloc[0], g["event_id"].iloc[0]
        for r in g.itertuples(index=False):
            a, ch = getattr(r, agent_col), (r.channel if has_channel else "")
            if a not in depth:
                prev = latest_on.get(ch)
                if prev and prev[0] != a:
                    (src, src_ts, src_ev), kind = prev, "relay"
                else:
                    src, src_ts, src_ev, kind = origin, t_origin, origin_ev, "relay_xchannel"
                dt = (r.ts - src_ts).total_seconds()
                if max_gap_s is not None and dt > max_gap_s:
                    depth[a] = 0   # the parent is outside the window: an unattributed root
                else:
                    depth[a] = depth[src] + 1
                    edges.append({"src": src, "dst": a, "family": fam, "item": item, "value_norm": vk,
                                  "kind": kind, "dt_s": dt, "event_id": r.event_id, "src_event_id": src_ev,
                                  # direction not reliable inside the clock resolution
                                  "within_clock_res": None if clock_res_s is None else dt <= clock_res_s})
            latest_on[ch] = (a, r.ts, r.event_id)
        multi_rows.append({"family": fam, "item": item, "value_key": vk, "originator": origin,
                           "t0": t_origin, "n_agents": len(depth), "max_depth": max(depth.values()),
                           "span_s": (g["ts"].iloc[-1] - t_origin).total_seconds()})
    chains.append(pd.DataFrame(multi_rows))
    chains = [c for c in chains if len(c)]
    return pd.DataFrame(edges), (pd.concat(chains, ignore_index=True) if chains else pd.DataFrame())


def exposure_edges(exp: pd.DataFrame) -> pd.DataFrame:
    """First public copy -> answering agent.  ``event_id`` is the report post (the claim
    is ``claim_id``), and the value is the key exposure matched on."""
    e = exp[exp["D"] == 1]
    return pd.DataFrame({
        "src": e["src_agent"], "dst": e["agent"], "family": e["family"], "item": e["item"],
        "value_norm": [match_key(v, c) for v, c in zip(e["value_norm"], e["consensus_value"])], "kind": "exposure",
        "dt_s": e["gap_s"], "event_id": e["report_event_id"], "claim_id": e["claim_id"],
        "src_event_id": e["src_event_id"],
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


def reply_edges(events: pd.DataFrame, agents: pd.DataFrame, agent_col: str) -> tuple[pd.DataFrame, int]:
    """The parent's agent -> the replier, for each post whose ``parent_id`` is another agent's post.  Self-replies
    are dropped, and task messages and tool results neither reply nor are replied to (they are no agent's post).
    Also returns how many replies name a parent that is not an agent's post in the run."""
    ev = events[~events["source_kind"].isin(PRIVATE_KINDS)] if "source_kind" in events else events
    label = agents.drop_duplicates("author_raw").set_index("author_raw")[agent_col]
    ev = ev.assign(_agent=ev["author_raw"].map(label).fillna(ev["author_raw"]))
    parent = ev.drop_duplicates("event_id").set_index("event_id")
    r = ev[ev["parent_id"].notna()]
    known = r["parent_id"].isin(parent.index)
    r = r[known]
    p = parent.loc[r["parent_id"]]
    out = pd.DataFrame({"src": p["_agent"].values, "dst": r["_agent"].values,
                        "family": r["family"].fillna("").values if "family" in r else "", "item": None,
                        "value_norm": None, "kind": "reply",
                        "dt_s": (r["ts"].values - p["ts"].values) / np.timedelta64(1, "s"),
                        "event_id": r["event_id"].values, "src_event_id": r["parent_id"].values})
    return out[out["src"] != out["dst"]].reset_index(drop=True), int((~known).sum())


def edge_table(edges: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per (src, dst): its edges by kind, the distinct facts they carried, the first and last destination post's
    time, the median delay and the hops inside the clock resolution."""
    ts = events.drop_duplicates("event_id").set_index("event_id")["ts"]
    e = edges.assign(_ts=edges["event_id"].map(ts), _res=edges["within_clock_res"].fillna(False).astype(bool))
    g = e.groupby(["src", "dst"])
    out = pd.crosstab([e["src"], e["dst"]], e["kind"]).reindex(columns=KINDS, fill_value=0).rename(columns=SHORT)
    facts = e[e["item"].notna()].drop_duplicates(["src", "dst", "family", "item", "value_norm"])
    return out.assign(n_facts=facts.groupby(["src", "dst"]).size().reindex(out.index, fill_value=0),
                      first_ts=g["_ts"].min(), last_ts=g["_ts"].max(), median_dt_s=g["dt_s"].median(),
                      n_within_clock_res=g["_res"].sum())


def scalar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One scalar type per column, as GraphML needs: a count stays an integer, other numbers floats, times ISO
    text; a missing number becomes -1 and a missing time or text ''."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(bool).tolist()
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[c] = [t.isoformat() if pd.notna(t) else "" for t in s]
        elif pd.api.types.is_integer_dtype(s):
            out[c] = [int(v) for v in s.fillna(-1)]
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = [float(v) for v in s.fillna(-1.0)]
        else:
            out[c] = ["" if v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v)) else str(v) for v in s]
    return pd.DataFrame(out, index=df.index)


def to_networkx(edges: pd.DataFrame, kinds=KINDS, events: pd.DataFrame | None = None,
                nodes: pd.DataFrame | None = None) -> nx.DiGraph:
    """One edge per (src, dst) over ``kinds``.  ``weight`` counts its ``TRANSFER_KINDS`` edges (with ``events``
    every kind is counted apart, ``relay_xchannel`` as ``n_xchannel``); ``kinds`` lists the kinds.
    ``events`` adds ``edge_table``'s attributes, ``nodes`` (indexed by agent label, e.g. ``agents.profile``) node
    attributes, both through ``scalar_frame``; a node missing from ``nodes`` gets -1 and ''."""
    G = nx.DiGraph()
    e = edges[edges["kind"].isin(kinds)]
    for (s, d), g in e.groupby(["src", "dst"]):
        G.add_edge(s, d, weight=int(g["kind"].isin(TRANSFER_KINDS).sum()), kinds=",".join(sorted(set(g["kind"]))))
    if events is not None and len(e):
        for (s, d), row in scalar_frame(edge_table(e, events)).to_dict("index").items():
            G.edges[s, d].update(row)
    if nodes is not None and len(G):
        attrs = scalar_frame(nodes)
        blank = {c: False if pd.api.types.is_bool_dtype(v) else -1 if pd.api.types.is_integer_dtype(v)
                 else -1.0 if pd.api.types.is_float_dtype(v) else "" for c, v in attrs.items()}
        known = attrs.to_dict("index")
        nx.set_node_attributes(G, {n: known.get(n, blank) for n in G})
    return G


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
