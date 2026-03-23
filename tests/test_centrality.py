"""Tests for centrality measures."""

import numpy as np
import pandas as pd
import pytest

from psynet import estimate_network
from psynet.centrality import (
    strength, expected_influence, closeness, betweenness, centrality,
    in_strength, out_strength, in_expected_influence, out_expected_influence,
)


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


    def test_directed_strength(self):
        """For directed networks, strength = in-strength + out-strength."""
        from psynet.network import Network
        # Asymmetric adjacency: A[i,j] means edge from i to j
        adj = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.3],
            [0.2, 0.0, 0.0],
        ])
        net = Network(adj, ["A", "B", "C"], "test", 100, directed=True)
        s = strength(net)
        # A: out=0.5, in=0.2 → 0.7
        # B: out=0.3, in=0.5 → 0.8
        # C: out=0.2, in=0.3 → 0.5
        np.testing.assert_almost_equal(s["A"], 0.7)
        np.testing.assert_almost_equal(s["B"], 0.8)
        np.testing.assert_almost_equal(s["C"], 0.5)

    def test_directed_ei(self):
        """For directed networks, EI = in-EI + out-EI."""
        from psynet.network import Network
        adj = np.array([
            [0.0, 0.5, -0.3],
            [0.0, 0.0, 0.2],
            [0.1, 0.0, 0.0],
        ])
        net = Network(adj, ["A", "B", "C"], "test", 100, directed=True)
        ei = expected_influence(net)
        # A: out=0.5+(-0.3)=0.2, in=0.0+0.1=0.1 → 0.3
        # B: out=0.2, in=0.5 → 0.7
        # C: out=0.1, in=-0.3+0.2=-0.1 → 0.0
        np.testing.assert_almost_equal(ei["A"], 0.3)
        np.testing.assert_almost_equal(ei["B"], 0.7)
        np.testing.assert_almost_equal(ei["C"], 0.0)


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


class TestNormalization:
    """Test normalized parameter for closeness and betweenness."""

    def test_closeness_unnormalized_smaller(self, small_data):
        """Unnormalized closeness should be smaller by factor (n-1)."""
        net = estimate_network(small_data, method="cor")
        c_norm = closeness(net, normalized=True)
        c_raw = closeness(net, normalized=False)
        n = net.n_nodes
        # normalized = raw * (n-1), so raw = normalized / (n-1)
        np.testing.assert_allclose(c_raw * (n - 1), c_norm, rtol=1e-10)

    def test_betweenness_unnormalized_larger(self, small_data):
        """Unnormalized betweenness should be larger by normalization factor."""
        net = estimate_network(small_data, method="cor")
        b_norm = betweenness(net, normalized=True)
        b_raw = betweenness(net, normalized=False)
        n = net.n_nodes
        # For undirected: normalized = raw * 2 / ((n-1)(n-2))
        factor = 2.0 / ((n - 1) * (n - 2))
        np.testing.assert_allclose(b_norm, b_raw * factor, rtol=1e-10)

    def test_centrality_passes_normalized(self, small_data):
        """centrality(normalized=False) should propagate to closeness/betweenness."""
        net = estimate_network(small_data, method="cor")
        df = centrality(net, normalized=False)
        c_raw = closeness(net, normalized=False)
        b_raw = betweenness(net, normalized=False)
        pd.testing.assert_series_equal(df["closeness"], c_raw, check_names=False)
        pd.testing.assert_series_equal(df["betweenness"], b_raw, check_names=False)


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

    def test_undirected_lacks_directed_columns(self, small_data):
        """Undirected networks should not have in/out columns."""
        net = estimate_network(small_data, method="cor")
        c = centrality(net)
        assert "inStrength" not in c.columns
        assert "outStrength" not in c.columns

    def test_directed_has_directed_columns(self):
        """Directed networks should include in/out strength and EI columns."""
        from psynet.network import Network
        adj = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
        net = Network(adj, ["A", "B", "C"], "test", 100, directed=True)
        c = centrality(net)
        assert "inStrength" in c.columns
        assert "outStrength" in c.columns
        assert "inExpectedInfluence" in c.columns
        assert "outExpectedInfluence" in c.columns


class TestDirectedCentrality:
    """Tests for in/out strength and expected influence on directed networks."""

    def _make_directed(self):
        from psynet.network import Network
        # adjacency[j, k] = effect of k on j (k -> j)
        adj = np.array([
            [0.0, 0.4, -0.2],
            [0.3, 0.0,  0.0],
            [0.0, 0.5,  0.0],
        ])
        return Network(adj, ["A", "B", "C"], "test", 100, directed=True)

    def test_in_strength(self):
        net = self._make_directed()
        s = in_strength(net)
        # A: |0.4| + |-0.2| = 0.6
        # B: |0.3| + |0.0| = 0.3
        # C: |0.0| + |0.5| = 0.5
        np.testing.assert_almost_equal(s["A"], 0.6)
        np.testing.assert_almost_equal(s["B"], 0.3)
        np.testing.assert_almost_equal(s["C"], 0.5)

    def test_out_strength(self):
        net = self._make_directed()
        s = out_strength(net)
        # A: col 0 = |0.0| + |0.3| + |0.0| = 0.3
        # B: col 1 = |0.4| + |0.0| + |0.5| = 0.9
        # C: col 2 = |-0.2| + |0.0| + |0.0| = 0.2
        np.testing.assert_almost_equal(s["A"], 0.3)
        np.testing.assert_almost_equal(s["B"], 0.9)
        np.testing.assert_almost_equal(s["C"], 0.2)

    def test_in_plus_out_equals_total(self):
        net = self._make_directed()
        total = strength(net)
        ins = in_strength(net)
        outs = out_strength(net)
        np.testing.assert_almost_equal(total.values, ins.values + outs.values)

    def test_in_expected_influence(self):
        net = self._make_directed()
        ei = in_expected_influence(net)
        # A: 0.4 + (-0.2) = 0.2
        # B: 0.3 + 0.0 = 0.3
        # C: 0.0 + 0.5 = 0.5
        np.testing.assert_almost_equal(ei["A"], 0.2)
        np.testing.assert_almost_equal(ei["B"], 0.3)
        np.testing.assert_almost_equal(ei["C"], 0.5)

    def test_out_expected_influence(self):
        net = self._make_directed()
        ei = out_expected_influence(net)
        # A: col 0 = 0.0 + 0.3 + 0.0 = 0.3
        # B: col 1 = 0.4 + 0.0 + 0.5 = 0.9
        # C: col 2 = -0.2 + 0.0 + 0.0 = -0.2
        np.testing.assert_almost_equal(ei["A"], 0.3)
        np.testing.assert_almost_equal(ei["B"], 0.9)
        np.testing.assert_almost_equal(ei["C"], -0.2)

    def test_raises_on_undirected(self, small_data):
        net = estimate_network(small_data, method="cor")
        with pytest.raises(ValueError, match="directed"):
            in_strength(net)
        with pytest.raises(ValueError, match="directed"):
            out_strength(net)
        with pytest.raises(ValueError, match="directed"):
            in_expected_influence(net)
        with pytest.raises(ValueError, match="directed"):
            out_expected_influence(net)
