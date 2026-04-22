"""Tests for plotting functions (smoke tests)."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import pytest

from psynet import estimate_network
from psynet.plotting import plot_network, plot_centrality, set_theme, get_theme
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


class TestNetworkPlotLegend:
    def test_legend_shown_by_default(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net)
        # Should have 2 axes: network + legend panel
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_legend_disabled(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net, show_legend=False)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_numbered_labels(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net)
        # Network axes is the first one; node labels should be numeric
        net_ax = fig.axes[0]
        texts = [t.get_text() for t in net_ax.texts]
        assert all(t.isdigit() for t in texts if t.strip())
        plt.close(fig)

    def test_custom_ax_no_side_legend(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig, ax = plt.subplots()
        plot_network(net, ax=ax, show_legend=True)
        # When ax is provided, no extra axes are created
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_centrality_aura(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net, centrality_aura="strength")
        assert fig is not None
        plt.close(fig)

    def test_centrality_aura_with_sized_nodes(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net, node_size="strength", centrality_aura="betweenness")
        assert fig is not None
        plt.close(fig)

    def test_centrality_aura_disabled(self, small_data):
        net = estimate_network(small_data, method="cor")
        fig = plot_network(net, centrality_aura=None)
        assert fig is not None
        plt.close(fig)


class TestThemeConstants:
    def test_theme_constants_accessible(self):
        from psynet.plotting._theme import (
            EDGE_COLOR_POS, EDGE_COLOR_NEG, NODE_FILL_COLOR,
            NODE_SIZE_DEFAULT, EDGE_WIDTH_MAX, ACCENT_COLORS,
        )
        assert isinstance(NODE_SIZE_DEFAULT, (int, float))
        assert NODE_SIZE_DEFAULT > 0
        assert isinstance(ACCENT_COLORS, list)
        assert len(ACCENT_COLORS) >= 5

    def test_aura_and_alpha_constants(self):
        from psynet.plotting._theme import (
            AURA_N_SEGMENTS, AURA_ALPHA_START, EDGE_ALPHA_MIN, EDGE_ALPHA_MAX,
        )
        assert AURA_N_SEGMENTS > 0
        assert 0 < AURA_ALPHA_START <= 1.0
        assert 0 < EDGE_ALPHA_MIN < EDGE_ALPHA_MAX <= 1.0


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


class TestDarkTheme:
    def setup_method(self):
        set_theme("light")

    def teardown_method(self):
        set_theme("light")

    def test_set_theme_changes_constants(self):
        from psynet.plotting import _theme
        set_theme("dark")
        assert get_theme() == "dark"
        assert _theme.BACKGROUND_COLOR == "#0e0f14"
        assert _theme.EDGE_COLOR_NEG == "#c96b62"
        assert _theme.EDGE_COLOR_POS == "#6b9fd4"

    def test_network_plot_uses_dark_background(self, small_data):
        set_theme("dark")
        net = estimate_network(small_data, method="cor")
        fig = net.plot()
        fc = fig.get_facecolor()
        # #0e0f14 is very dark — all channels below 0.1
        assert fc[0] < 0.1 and fc[1] < 0.1 and fc[2] < 0.1
        plt.close(fig)

    def test_community_plot_uses_dark_palette(self):
        from psynet.plotting import _theme
        set_theme("dark")
        assert _theme.COMMUNITY_PALETTE[0] == "#5bb98b"
        assert _theme.COMMUNITY_PALETTE[1] == "#d4915e"
        assert _theme.COMMUNITY_PALETTE[2] == "#6b9fd4"

    def test_theme_round_trip(self):
        from psynet.plotting import _theme
        light_bg = _theme.BACKGROUND_COLOR
        set_theme("dark")
        assert _theme.BACKGROUND_COLOR != light_bg
        set_theme("light")
        assert _theme.BACKGROUND_COLOR == light_bg

    def test_light_theme_is_default(self):
        assert get_theme() == "light"

    def test_invalid_theme_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown theme"):
            set_theme("purple")
