"""Tests for network estimation."""

import numpy as np
import pandas as pd
import pytest

from psynet.estimation import estimate_network, available_methods
from psynet.network import Network


class TestAvailableMethods:
    def test_all_registered(self):
        methods = available_methods()
        assert "cor" in methods
        assert "ebicglasso" in methods
        assert "pcor" in methods

    def test_invalid_method_raises(self):
        df = pd.DataFrame(np.eye(5), columns=list("ABCDE"))
        with pytest.raises(ValueError, match="Unknown estimation method"):
            estimate_network(df, method="nonexistent")


class TestCorEstimator:
    def test_produces_valid_network(self, small_data):
        net = estimate_network(small_data, method="cor")
        assert isinstance(net, Network)
        assert net.n_nodes == 5
        assert net.n_observations == 50
        assert net.method == "cor"

    def test_adjacency_symmetric(self, small_data):
        net = estimate_network(small_data, method="cor")
        np.testing.assert_array_almost_equal(net.adjacency, net.adjacency.T)

    def test_diagonal_zero(self, small_data):
        net = estimate_network(small_data, method="cor")
        np.testing.assert_array_equal(np.diag(net.adjacency), 0)

    def test_weights_bounded(self, small_data):
        net = estimate_network(small_data, method="cor")
        assert np.all(np.abs(net.adjacency) <= 1.0)

    def test_threshold(self, small_data):
        net = estimate_network(small_data, method="cor", threshold=0.3)
        small_vals = np.abs(net.adjacency)
        nonzero = small_vals[small_vals > 0]
        assert np.all(nonzero >= 0.3)


class TestPCorEstimator:
    def test_produces_valid_network(self, small_data):
        net = estimate_network(small_data, method="pcor")
        assert isinstance(net, Network)
        assert net.n_nodes == 5

    def test_adjacency_symmetric(self, small_data):
        net = estimate_network(small_data, method="pcor")
        np.testing.assert_array_almost_equal(net.adjacency, net.adjacency.T)

    def test_diagonal_zero(self, small_data):
        net = estimate_network(small_data, method="pcor")
        np.testing.assert_array_equal(np.diag(net.adjacency), 0)


class TestEBICglassoEstimator:
    def test_produces_valid_network(self, medium_data):
        net = estimate_network(medium_data, method="EBICglasso")
        assert isinstance(net, Network)
        assert net.n_nodes == 10
        assert net.method == "EBICglasso"

    def test_sparse(self, medium_data):
        net = estimate_network(medium_data, method="EBICglasso", gamma=0.5)
        # EBICglasso should produce a sparse network
        n_zero = np.sum(net.adjacency == 0) - net.n_nodes  # exclude diagonal
        total = net.n_nodes ** 2 - net.n_nodes
        sparsity = n_zero / total
        assert sparsity > 0.1  # at least somewhat sparse

    def test_adjacency_symmetric(self, medium_data):
        net = estimate_network(medium_data, method="EBICglasso")
        np.testing.assert_array_almost_equal(net.adjacency, net.adjacency.T)

    def test_higher_gamma_more_sparse(self, medium_data):
        net_low = estimate_network(medium_data, method="EBICglasso", gamma=0.0)
        net_high = estimate_network(medium_data, method="EBICglasso", gamma=1.0)
        edges_low = np.count_nonzero(net_low.adjacency)
        edges_high = np.count_nonzero(net_high.adjacency)
        assert edges_high <= edges_low


class TestNetworkObject:
    def test_edges_df(self, small_data):
        net = estimate_network(small_data, method="cor")
        edges = net.edges_df
        assert "node1" in edges.columns
        assert "node2" in edges.columns
        assert "weight" in edges.columns

    def test_adjacency_df(self, small_data):
        net = estimate_network(small_data, method="cor")
        adj_df = net.adjacency_df
        assert list(adj_df.columns) == net.labels
        assert list(adj_df.index) == net.labels

    def test_to_networkx(self, small_data):
        import networkx as nx
        net = estimate_network(small_data, method="cor")
        G = net.to_networkx()
        assert isinstance(G, nx.Graph)
        assert set(G.nodes()) == set(net.labels)

    def test_frozen(self, small_data):
        net = estimate_network(small_data, method="cor")
        with pytest.raises(AttributeError):
            net.method = "other"
