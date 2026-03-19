"""Tests for community detection."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from psynet import estimate_network
from psynet.community import louvain, greedy_modularity, walktrap, communities
from psynet.network import Network
from psynet.plotting import plot_community


class TestLouvain:
    def test_returns_series(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = louvain(net, seed=42)
        assert isinstance(c, pd.Series)
        assert c.dtype == int

    def test_all_nodes_assigned(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = louvain(net, seed=42)
        assert len(c) == net.n_nodes
        assert not c.isna().any()

    def test_finds_two_groups(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = louvain(net, seed=42)
        # V1-V3 should be in one community, V4-V5 in another
        assert c["V1"] == c["V2"] == c["V3"]
        assert c["V4"] == c["V5"]
        assert c["V1"] != c["V4"]

    def test_seed_reproducibility(self, small_data):
        net = estimate_network(small_data, method="cor")
        c1 = louvain(net, seed=123)
        c2 = louvain(net, seed=123)
        pd.testing.assert_series_equal(c1, c2)


class TestGreedyModularity:
    def test_returns_series(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = greedy_modularity(net)
        assert isinstance(c, pd.Series)

    def test_all_nodes_assigned(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = greedy_modularity(net)
        assert len(c) == net.n_nodes
        assert not c.isna().any()

    def test_finds_two_groups(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = greedy_modularity(net)
        assert c["V1"] == c["V2"] == c["V3"]
        assert c["V4"] == c["V5"]
        assert c["V1"] != c["V4"]


class TestWalktrap:
    def test_returns_series(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = walktrap(net)
        assert isinstance(c, pd.Series)

    def test_all_nodes_assigned(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = walktrap(net)
        assert len(c) == net.n_nodes
        assert not c.isna().any()

    def test_finds_groups(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = walktrap(net)
        # V1-V3 should cluster together (strong factor 1)
        assert c["V1"] == c["V2"] == c["V3"]
        # V4 and V5 should be separate from V1-V3
        assert c["V4"] != c["V1"]
        assert c["V5"] != c["V1"]

    def test_steps_parameter(self, small_data):
        net = estimate_network(small_data, method="cor")
        c1 = walktrap(net, steps=2)
        c8 = walktrap(net, steps=8)
        # Both should produce valid assignments
        assert len(c1) == net.n_nodes
        assert len(c8) == net.n_nodes

    def test_known_structure(self):
        """Hand-crafted adjacency with two clear clusters."""
        adj = np.array([
            [0.0, 0.8, 0.7, 0.0, 0.0],
            [0.8, 0.0, 0.6, 0.0, 0.0],
            [0.7, 0.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0, 0.9, 0.0],
        ])
        net = Network(adj, ["A", "B", "C", "D", "E"], "test", 100)
        c = walktrap(net)
        assert c["A"] == c["B"] == c["C"]
        assert c["D"] == c["E"]
        assert c["A"] != c["D"]


class TestCommunities:
    def test_default_walktrap(self, small_data):
        net = estimate_network(small_data, method="cor")
        c_default = communities(net)
        c_walktrap = walktrap(net)
        pd.testing.assert_series_equal(c_default, c_walktrap)

    def test_all_methods_dispatch(self, small_data):
        net = estimate_network(small_data, method="cor")
        for method in ["walktrap", "louvain", "greedy_modularity"]:
            c = communities(net, method=method, seed=42) if method == "louvain" else communities(net, method=method)
            assert isinstance(c, pd.Series)
            assert len(c) == net.n_nodes

    def test_invalid_method(self, small_data):
        net = estimate_network(small_data, method="cor")
        with pytest.raises(ValueError, match="Unknown community detection method"):
            communities(net, method="invalid_method")

    def test_via_network_method(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = net.communities()
        assert isinstance(c, pd.Series)
        assert len(c) == net.n_nodes


class TestCommunityPlot:
    def test_smoke(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = communities(net)
        fig = plot_community(net, c)
        assert fig is not None

    def test_custom_palette(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = communities(net)
        fig = plot_community(net, c, palette=["red", "blue", "green"])
        assert fig is not None

    def test_via_network_method(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = net.plot_communities()
        assert fig is not None
