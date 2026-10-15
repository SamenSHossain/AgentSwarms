"""A5 - structure: hub concentration, brokers, relay chain lengths.  A6 - reach.

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


def relay_parents(ex: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    """Each round's same-page relay parent: the ``relay`` edge into the agent for the value the
    round was matched on (``exposure.match_key``); None for a cross-page hop or no edge at all."""
    rel = edges[edges["kind"] == "relay"]
    parent = dict(zip(zip(rel["dst"], rel["family"], rel["item"], rel["value_norm"]), rel["src"]))
    return pd.Series([parent.get((a, f, i, match_key(v, c))) for a, f, i, v, c in
                      zip(ex["agent"], ex["family"], ex["item"], ex["value_norm"], ex["consensus_value"])],
                     index=ex.index, dtype=object)


def concentration(sources: pd.Series, top_k: int = 10) -> dict:
    """Answers credited, distinct sources, the top-k sources' share and the Gini of their counts."""
    deg = sources.dropna().value_counts()
    tot = deg.sum()
    return {"n": int(tot), "sources": int(len(deg)),
            f"top{top_k}_share": float(deg.head(top_k).sum() / tot) if tot else float("nan"),
            "gini": gini(deg.values) if tot else float("nan")}


def attribution(ex: pd.DataFrame, edges: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Concentration of exposed answers (D == 1) under each attribution rule, on every exposed
    answer and on instant ones only.  ``unmatched`` counts the answers a rule credits to nobody."""
    x = ex[ex["D"] == 1]
    rules = {"first source": x["src_agent"], "latest source": x["src_latest"],
             "same-page relay parent": relay_parents(x, edges)}
    rows = []
    for subset, idx in (("all exposed", x.index), ("instant only", x.index[x["y_instant"] == 1])):
        for rule, src in rules.items():
            s = src.loc[idx]
            rows.append({"rule": rule, "answers": subset, **concentration(s, top_k), "unmatched": int(s.isna().sum())})
    return pd.DataFrame(rows)


def analyze(edges: pd.DataFrame, chains: pd.DataFrame, top_k: int = 10, ex: pd.DataFrame | None = None) -> dict:
    """``ex`` (the exposure table) adds the attribution-rule table and the per-item null."""
    outdeg = edges[edges["kind"] == "exposure"].groupby("src").size().sort_values(ascending=False)
    relay_out = edges[edges["kind"] == "relay"].groupby("src").size().sort_values(ascending=False)
    G = to_networkx(edges, kinds=("relay", "citation"))
    k = 300 if len(G) > EXACT_BETWEENNESS_NODES else None
    btw = nx.betweenness_centrality(G, k=k, seed=0) if len(G) else {}
    task_chains = chains[chains["family"] != "url"] if len(chains) else chains
    multi = task_chains[task_chains["n_agents"] >= 2] if len(task_chains) else task_chains
    res = {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "n_exposure_edges": int((edges["kind"] == "exposure").sum()),
        "n_relay_edges": int((edges["kind"] == "relay").sum()),
        "n_xchannel_edges": int((edges["kind"] == "relay_xchannel").sum()),
        "n_citation_edges": int((edges["kind"] == "citation").sum()),
        "exposure_sources": int(len(outdeg)),
        "gini_exposure_outdegree": gini(outdeg.values) if len(outdeg) else float("nan"),
        f"top{top_k}_share_exposure": float(outdeg.head(top_k).sum() / outdeg.sum()) if len(outdeg) else float("nan"),
        "gini_relay_outdegree": gini(relay_out.values) if len(relay_out) else float("nan"),
        "top_sources": outdeg.head(top_k),
        "top_brokers": pd.Series(btw).sort_values(ascending=False).head(top_k),
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
                "attribution": attribution(ex, edges, top_k)}
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
