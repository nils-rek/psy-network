"""Tests for centrality measures."""

import numpy as np
import pandas as pd
import pytest

from psynet import estimate_network
from psynet.centrality import strength, expected_influence, closeness, betweenness, centrality


class TestStrength:
    def test_returns_series(self, small_data):
        net = estimate_network(small_data, method="cor")
        s = strength(net)
        assert isinstance(s, pd.Series)
        assert len(s) == net.n_nodes

    def test_nonnegative(self, small_data):
        net = estimate_network(small_data, method="cor")
        s = strength(net)
        assert np.all(s >= 0)

    def test_known_value(self):
        """For a known adjacency, strength = sum of |weights| per row."""
        from psynet.network import Network
        adj = np.array([
            [0.0, 0.5, -0.3],
            [0.5, 0.0, 0.2],
            [-0.3, 0.2, 0.0],
        ])
        net = Network(adj, ["A", "B", "C"], "test", 100)
        s = strength(net)
        np.testing.assert_almost_equal(s["A"], 0.8)
        np.testing.assert_almost_equal(s["B"], 0.7)
        np.testing.assert_almost_equal(s["C"], 0.5)


class TestExpectedInfluence:
    def test_known_value(self):
        from psynet.network import Network
        adj = np.array([
            [0.0, 0.5, -0.3],
            [0.5, 0.0, 0.2],
            [-0.3, 0.2, 0.0],
        ])
        net = Network(adj, ["A", "B", "C"], "test", 100)
        ei = expected_influence(net)
        np.testing.assert_almost_equal(ei["A"], 0.2)   # 0.5 + (-0.3)
        np.testing.assert_almost_equal(ei["B"], 0.7)   # 0.5 + 0.2
        np.testing.assert_almost_equal(ei["C"], -0.1)  # -0.3 + 0.2


class TestClosenessAndBetweenness:
    def test_returns_series(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = closeness(net)
        b = betweenness(net)
        assert isinstance(c, pd.Series)
        assert isinstance(b, pd.Series)
        assert len(c) == net.n_nodes
        assert len(b) == net.n_nodes

    def test_nonnegative(self, small_data):
        net = estimate_network(small_data, method="cor")
        assert np.all(closeness(net) >= 0)
        assert np.all(betweenness(net) >= 0)


class TestCentrality:
    def test_returns_dataframe(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = centrality(net)
        assert isinstance(c, pd.DataFrame)
        assert set(c.columns) == {"strength", "closeness", "betweenness", "expectedInfluence"}
        assert len(c) == net.n_nodes

    def test_via_network_method(self, small_data):
        net = estimate_network(small_data, method="cor")
        c = net.centrality()
        assert isinstance(c, pd.DataFrame)
