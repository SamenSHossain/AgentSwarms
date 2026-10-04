"""A5 - structure: hub concentration, brokers, relay chain lengths.  A6 - reach."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .. import plotting
from ..graph import gini, to_networkx
from ..plotting import plt


def analyze(edges: pd.DataFrame, chains: pd.DataFrame, top_k: int = 10) -> dict:
    outdeg = edges[edges["kind"] == "exposure"].groupby("src").size().sort_values(ascending=False)
    relay_out = edges[edges["kind"] == "relay"].groupby("src").size().sort_values(ascending=False)
    G = to_networkx(edges, kinds=("relay", "citation"))
    btw = nx.betweenness_centrality(G, k=min(300, len(G)) if len(G) > 300 else None, seed=0) if len(G) else {}
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
        "chain_depth": multi["max_depth"].value_counts().sort_index() if len(multi) else pd.Series(dtype=int),
        "mean_chain_depth": float(multi["max_depth"].mean()) if len(multi) else float("nan"),
        "mean_agents_per_fact": float(multi["n_agents"].mean()) if len(multi) else float("nan"),
        "outdeg": outdeg,
    }
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
    od = np.sort(res["outdeg"].values) if len(res["outdeg"]) else np.array([])
    if len(od):
        cum = np.concatenate([[0], np.cumsum(od) / od.sum()])
        x = np.linspace(0, 1, len(cum))
        ax.plot(x, cum, color=plotting.SERIES[0], label=f"exposure sources (Gini {res['gini_exposure_outdegree']:.2f})")
        ax.plot([0, 1], [0, 1], color=plotting.NEUTRAL, lw=1, ls="--", label="perfect equality")
        ax.set_xlabel("Share of source agents (ascending)")
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
