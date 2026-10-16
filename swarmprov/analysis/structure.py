"""A5 - structure: hub concentration, brokers, relay chain lengths, the supply network.  A6 - reach.

Concentration depends on who is credited with an exposed answer, so it is given under
three attribution rules (and on instant answers only, the copy signature), none of them
privileged: the *first* other carrier before the report (``src_agent``), the *latest*
(``src_latest``) and the same-page relay parent from the graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .. import plotting
from ..exposure import match_key
from ..graph import gini, to_networkx
from ..plotting import plt


# exact betweenness up to this many nodes (seconds on the wiki's ~1,400); sampled from 300 sources above it
EXACT_BETWEENNESS_NODES = 5000
# the brokers' graph: replies are interactions, so they count here but never toward supply (exposure) or relays
BROKER_KINDS = ("relay", "citation", "reply")


def betweenness(G: nx.DiGraph) -> tuple[pd.Series, int | None]:
    """Betweenness of every node, exact up to ``EXACT_BETWEENNESS_NODES``, else from 300 sampled sources (seed 0);
    also returns the sample size (None when exact)."""
    k = 300 if len(G) > EXACT_BETWEENNESS_NODES else None
    return pd.Series(nx.betweenness_centrality(G, k=k, seed=0) if len(G) else {}, dtype=float), k


def relay_parents(ex: pd.DataFrame, edges: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Each round's same-page relay parent: the ``relay`` edge into the agent for the value the
    round was matched on (``exposure.match_key``); None for a cross-page hop, no edge at all, or a
    parent whose post is not before the report (exposure's rule for a public copy; ``trace`` agrees)."""
    rel = edges[edges["kind"] == "relay"]
    ts = events.drop_duplicates("event_id").set_index("event_id")["ts"]
    parent = dict(zip(zip(rel["dst"], rel["family"], rel["item"], rel["value_norm"]),
                      zip(rel["src"], rel["src_event_id"].map(ts))))
    out = []
    for a, f, i, v, c, t in zip(ex["agent"], ex["family"], ex["item"], ex["value_norm"], ex["consensus_value"],
                                ex["t_report"]):
        src, t_src = parent.get((a, f, i, match_key(v, c)), (None, None))
        out.append(None if pd.notna(t_src) and t_src >= t else src)
    return pd.Series(out, index=ex.index, dtype=object)


def concentration(sources: pd.Series, top_k: int = 10) -> dict:
    """Answers credited, distinct sources, the top-k sources' share and the Gini of their counts."""
    deg = sources.dropna().value_counts()
    tot = deg.sum()
    return {"n": int(tot), "sources": int(len(deg)),
            f"top{top_k}_share": float(deg.head(top_k).sum() / tot) if tot else float("nan"),
            "gini": gini(deg.values) if tot else float("nan")}


def attribution(ex: pd.DataFrame, edges: pd.DataFrame, events: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Concentration of exposed answers (D == 1) under each attribution rule, on every exposed
    answer and on instant ones only.  ``unmatched`` counts the answers a rule credits to nobody."""
    x = ex[ex["D"] == 1]
    rules = {"first source": x["src_agent"], "latest source": x["src_latest"],
             "same-page relay parent": relay_parents(x, edges, events)}
    rows = []
    for subset, idx in (("all exposed", x.index), ("instant only", x.index[x["y_instant"] == 1])):
        for rule, src in rules.items():
            s = src.loc[idx]
            rows.append({"rule": rule, "answers": subset, **concentration(s, top_k), "unmatched": int(s.isna().sum())})
    return pd.DataFrame(rows)


def analyze(edges: pd.DataFrame, chains: pd.DataFrame, top_k: int = 10, ex: pd.DataFrame | None = None,
            events: pd.DataFrame | None = None) -> dict:
    """``ex`` (the exposure table) adds the attribution-rule table and the per-item null; it needs
    ``events`` for the relay parents' post times."""
    outdeg = edges[edges["kind"] == "exposure"].groupby("src").size().sort_values(ascending=False)
    relay_out = edges[edges["kind"] == "relay"].groupby("src").size().sort_values(ascending=False)
    G = to_networkx(edges, kinds=BROKER_KINDS)
    btw, k = betweenness(G)
    task_chains = chains[chains["family"] != "url"] if len(chains) else chains
    multi = task_chains[task_chains["n_agents"] >= 2] if len(task_chains) else task_chains
    res = {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "n_exposure_edges": int((edges["kind"] == "exposure").sum()),
        "n_relay_edges": int((edges["kind"] == "relay").sum()),
        "n_xchannel_edges": int((edges["kind"] == "relay_xchannel").sum()),
        "n_citation_edges": int((edges["kind"] == "citation").sum()),
        "n_reply_edges": int((edges["kind"] == "reply").sum()),
        "exposure_sources": int(len(outdeg)),
        "gini_exposure_outdegree": gini(outdeg.values) if len(outdeg) else float("nan"),
        f"top{top_k}_share_exposure": float(outdeg.head(top_k).sum() / outdeg.sum()) if len(outdeg) else float("nan"),
        "gini_relay_outdegree": gini(relay_out.values) if len(relay_out) else float("nan"),
        "top_sources": outdeg.head(top_k),
        "top_brokers": btw.sort_values(ascending=False).head(top_k), "betweenness": btw,
        "betweenness_sampled_sources": k,
        "chain_depth": multi["max_depth"].value_counts().sort_index() if len(multi) else pd.Series(dtype=int),
        "mean_chain_depth": float(multi["max_depth"].mean()) if len(multi) else float("nan"),
        "mean_agents_per_fact": float(multi["n_agents"].mean()) if len(multi) else float("nan"),
        "outdeg": outdeg,
    }
    if ex is not None and len(ex):
        x = ex[ex["D"] == 1]
        per_item = x.groupby(["family", "item"])
        # the Gini a first-to-post rule gives if every item's answers went to its own single source
        res |= {"item_exposures": per_item.size(),
                "gini_item_exposures": gini(per_item.size().values) if len(x) else float("nan"),
                "items_exposed": int(per_item.ngroups),
                "items_one_source": int((per_item["src_agent"].nunique() == 1).sum()),
                "attribution": attribution(ex, edges, events, top_k)}
    return res


def reach(chains: pd.DataFrame) -> dict:
    """A6: for every fact (item+value, or URL): how many agents repeated it, how many hops."""
    if chains.empty:
        return {}
    out = {}
    for kind, g in [("task values", chains[chains["family"] != "url"]), ("urls", chains[chains["family"] == "url"])]:
        if g.empty:
            continue
        out[kind] = {
            "facts": int(len(g)), "repeated": int((g["n_agents"] > 1).sum()),
            "mean_repeaters": float((g["n_agents"] - 1).mean()),
            "p90_repeaters": float((g["n_agents"] - 1).quantile(0.9)),
            "max_hops": int(g["max_depth"].max()), "mean_hops_repeated": float(g.loc[g["n_agents"] > 1, "max_depth"].mean()),
        }
    return out


def figure(res: dict, chains: pd.DataFrame, path) -> str | None:
    plotting.setup()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    def lorenz(counts, **kw):
        c = np.sort(np.asarray(counts, dtype=float))
        cum = np.concatenate([[0], np.cumsum(c) / c.sum()])
        ax.plot(np.linspace(0, 1, len(cum)), cum, **kw)
    if len(res["outdeg"]):
        lorenz(res["outdeg"].values, color=plotting.SERIES[0],
               label=f"first sources (Gini {res['gini_exposure_outdegree']:.2f})")
        if "item_exposures" in res:
            lorenz(res["item_exposures"].values, color=plotting.SERIES[1], lw=1.5, ls=":",
                   label=f"items (Gini {res['gini_item_exposures']:.2f}): one source each")
        ax.plot([0, 1], [0, 1], color=plotting.NEUTRAL, lw=1, ls="--", label="perfect equality")
        ax.set_xlabel("Share of source agents (or items), ascending")
        ax.set_ylabel("Share of answers they supplied")
        ax.legend(loc="upper left")
        ax.set_title("Hub concentration (Lorenz curve)")
    dep = res["chain_depth"]
    if len(dep):
        ax2.bar(dep.index.astype(int), dep.values, color=plotting.SERIES[0], width=0.7)
        ax2.set_xlabel("Relay hops from originator (max per fact)")
        ax2.set_ylabel("Facts")
        ax2.set_title("Relay chain length")
        ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return plotting.save(fig, path)


def reach_figure(chains: pd.DataFrame, path) -> str | None:
    if chains.empty:
        return None
    plotting.setup()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (lab, g) in enumerate([("task values", chains[chains["family"] != "url"]),
                                  ("URLs", chains[chains["family"] == "url"])]):
        if g.empty:
            continue
        vc = (g["n_agents"] - 1).clip(upper=20).value_counts().sort_index()
        ax.plot(vc.index, vc.values, marker="o", color=plotting.SERIES[i], label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Other agents who repeated the fact (capped at 20)")
    ax.set_ylabel("Facts (log)")
    ax.set_title("Reach: how far does an announced fact travel?")
    ax.legend()
    fig.tight_layout()
    return plotting.save(fig, path)


def _induced(G: nx.Graph, nodes) -> nx.Graph:
    """The subgraph on ``nodes`` with its nodes in sorted order: a subgraph view iterates a set, whose order changes
    with the hash seed, and a seeded layout depends on the node order."""
    nodes = sorted(nodes)
    H = nx.Graph()
    H.add_nodes_from(nodes)
    keep = set(nodes)
    H.add_weighted_edges_from((u, v, w) for u, v, w in G.edges(nodes, data="weight") if v in keep)
    return H


def _starburst(G: nx.Graph, sources, seed: int = 0) -> dict:
    """Positions in units of one recipient spacing.  Sources and the recipients with two or more sources go in a
    seeded spring layout, scaled to the room their recipients need; a recipient with one source sits in a sunflower
    disc around it, so a hub's recipients spread out instead of piling up on it."""
    leaf = {n for n in G if n not in sources and G.degree(n) == 1}
    core = sorted(n for n in G if n not in leaf)
    kids = {h: sorted(m for m in G[h] if m in leaf) for h in core}
    room = np.sqrt(sum((2 * np.sqrt(len(k)) + 2) ** 2 for k in kids.values()))
    pos = ({n: p * room for n, p in nx.spring_layout(_induced(G, core), seed=seed).items()} if len(core) > 1
           else {core[0]: np.zeros(2)})
    golden = np.pi * (3 - np.sqrt(5))
    for h, k in kids.items():
        for i, n in enumerate(k, start=1):
            pos[n] = pos[h] + np.sqrt(i + 1) * np.array([np.cos(i * golden), np.sin(i * golden)])
    return pos


def _packed_layout(G: nx.Graph, sources, seed: int = 0) -> dict:
    """``_starburst`` per connected component, the components packed in rows, largest first (one layout of the
    whole graph would push them far apart)."""
    comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: (-len(c), c[0]))
    parts = [_starburst(_induced(G, c), sources, seed) for c in comps]
    boxes = [(np.min(list(q.values()), axis=0), np.max(list(q.values()), axis=0)) for q in parts]
    width = 1.3 * np.sqrt(sum(np.prod(hi - lo + 2) for lo, hi in boxes))
    pos, x, y, row = {}, 0.0, 0.0, 0.0
    for q, (lo, hi) in zip(parts, boxes):
        w, h = hi - lo + 2
        if x and x + w > width:
            x, y, row = 0.0, y - row, 0.0
        pos |= {n: p - lo + (x + 1, y - h + 1) for n, p in q.items()}
        x, row = x + w, max(row, h)
    return pos


def supply_figure(edges: pd.DataFrame, nodes: pd.DataFrame, res: dict, path, top: int = 10) -> str | None:
    """Who supplied whom under the first-source rule: the exposure edges aggregated by (src, dst), laid out by
    ``_packed_layout`` (seeded).  Sources are grey discs sized by the answers they supplied, the ``top`` largest labelled
    with the count; the other agents with an exposed answer are dots coloured by ``nodes['family']``
    (``agents.profile``), the 7 most common families named and the rest 'other'."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    x = edges[edges["kind"] == "exposure"]
    if not len(x):
        return None
    G = nx.Graph()
    G.add_weighted_edges_from((s, d, int(n)) for (s, d), n in x.groupby(["src", "dst"]).size().items())
    supplied = res["outdeg"]
    pos = _packed_layout(G, set(supplied.index))
    recipients = [n for n in G if n not in supplied.index]
    fam = nodes["family"].reindex(recipients).fillna("").replace("", "other") if recipients else pd.Series(dtype=str)
    named = [f for f in fam.value_counts().index if f != "other"][:7]
    colour = {f: plotting.SERIES[i] for i, f in enumerate(named)}
    plotting.setup()
    fig, ax = plt.subplots(figsize=(10, 8.6))
    ax.add_collection(LineCollection([(pos[s], pos[d]) for s, d in G.edges()], colors=plotting.NEUTRAL, linewidths=0.4,
                                     alpha=0.6, zorder=1))
    sxy = np.array([pos[n] for n in supplied.index])
    ax.scatter(sxy[:, 0], sxy[:, 1], s=20 + 600 * supplied.values / supplied.max(), color=plotting.INK_2, alpha=0.7,
               edgecolors=plotting.SURFACE, linewidths=1.2, zorder=2)
    xy = np.array([pos[n] for n in recipients]).reshape(-1, 2)   # above the discs, which can cover their recipients
    ax.scatter(xy[:, 0], xy[:, 1], s=9, c=[colour.get(f, plotting.NEUTRAL) for f in fam], linewidths=0, zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.autoscale_view()
    placed, renderer = [], fig.canvas.get_renderer()
    for n, k in supplied.head(top).items():   # the first of eight directions whose label overlaps no earlier one
        r = 6 + 12 * np.sqrt(k / supplied.max())
        for i, a in enumerate(np.radians([0, 180, 90, 270, 45, 135, 225, 315])):
            d = np.array([np.cos(a), np.sin(a)])
            t = ax.annotate(f"{n} ({k})".replace("$", r"\$"), pos[n], xytext=tuple(r * d), textcoords="offset points",
                            ha="left" if d[0] > 0.3 else "right" if d[0] < -0.3 else "center", va="center",
                            fontsize=7.5, color=plotting.INK, zorder=4,
                            bbox={"boxstyle": "round,pad=0.15", "fc": plotting.SURFACE, "ec": "none", "alpha": 0.85})
            box = t.get_window_extent(renderer)
            if i == 7 or not any(box.overlaps(b) for b in placed):
                break
            t.remove()
        placed.append(box)
    n_exp, share = int(supplied.sum()), res[f"top{top}_share_exposure"]
    ax.set_title(f"Who supplied whom, first-source rule: each exposed answer is credited to whoever first posted its "
                 f"value\nbefore the report; the top {top} sources supplied {share:.0%} of {n_exp:,} exposed answers",
                 loc="left", fontsize=11)
    handles = [Line2D([], [], ls="", marker="o", markersize=10, color=plotting.INK_2, alpha=0.7,
                      label="first source (size: answers supplied)")]
    handles += [Line2D([], [], ls="", marker="o", markersize=5, color=c, label=f) for f, c in colour.items()]
    if (~fam.isin(named)).any():
        handles.append(Line2D([], [], ls="", marker="o", markersize=5, color=plotting.NEUTRAL, label="other families"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)
    fig.tight_layout()
    return plotting.save(fig, path)
