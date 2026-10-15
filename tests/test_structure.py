"""A5 brokers: exact betweenness on graphs of the size runs produce, so the published table is not sampling noise."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from swarmprov.analysis import structure
from swarmprov.graph import to_networkx

CHAINS = pd.DataFrame(columns=["family", "n_agents", "max_depth"])


@pytest.fixture(scope="module")
def edges():
    """A 400-agent relay path (above the 300 sources the old sampled estimate used) with citation spurs."""
    rows = [{"src": f"a{i:03d}", "dst": f"a{i + 1:03d}", "kind": "relay"} for i in range(399)]
    rows += [{"src": f"a{i:03d}", "dst": f"c{i}", "kind": "citation"} for i in range(0, 400, 7)]
    return pd.DataFrame(rows)


def test_brokers_are_exact_below_the_threshold(edges):
    r = structure.analyze(edges, CHAINS)
    exact = pd.Series(nx.betweenness_centrality(to_networkx(edges, kinds=("relay", "citation"))))
    assert r["betweenness_sampled_sources"] is None
    assert np.allclose(r["top_brokers"].values, exact.sort_values(ascending=False).head(10).values)
    assert np.allclose(r["top_brokers"].values, exact[r["top_brokers"].index].values)


def test_large_graphs_fall_back_to_sampling(edges, monkeypatch):
    monkeypatch.setattr(structure, "EXACT_BETWEENNESS_NODES", 100)
    assert structure.analyze(edges, CHAINS)["betweenness_sampled_sources"] == 300
