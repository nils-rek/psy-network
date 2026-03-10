"""Tests for plotting functions (smoke tests)."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import pytest

from psynet import estimate_network
from psynet.plotting import plot_network, plot_centrality
from psynet.bootstrap import bootnet


class TestNetworkPlot:
    def test_default_plot(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net)
        assert fig is not None
        plt.close(fig)

    def test_layout_options(self, small_data):
        net = estimate_network(small_data, method="cor")
        for layout in ["spring", "circular", "kamada_kawai"]:
            fig = plot_network(net, layout=layout)
            assert fig is not None
            plt.close(fig)

    def test_centrality_sized_nodes(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net, node_size="strength")
        assert fig is not None
        plt.close(fig)

    def test_via_network_method(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = net.plot()
        assert fig is not None
        plt.close(fig)


class TestCentralityPlot:
    def test_default_plot(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_centrality(net)
        assert fig is not None
        plt.close(fig)

    def test_select_measures(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_centrality(net, measures=["strength", "closeness"])
        assert fig is not None
        plt.close(fig)


class TestBootstrapPlots:
    def test_edge_accuracy_plot(self, small_data):
        result = bootnet(
            small_data, n_boots=10, method="cor", statistics=["edge"],
            n_cores=1, seed=42, verbose=False,
        )
        fig = result.plot_edge_accuracy()
        assert fig is not None
        plt.close(fig)

    def test_centrality_stability_plot(self, small_data):
        result = bootnet(
            small_data, n_boots=5, boot_type="case", method="cor",
            n_cores=1, case_n=3, seed=42, verbose=False,
        )
        fig = result.plot_centrality_stability()
        assert fig is not None
        plt.close(fig)

    def test_difference_plot(self, small_data):
        result = bootnet(
            small_data, n_boots=15, method="cor", statistics=["edge"],
            n_cores=1, seed=42, verbose=False,
        )
        fig = result.plot_difference()
        assert fig is not None
        plt.close(fig)
