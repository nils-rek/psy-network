"""Tests for time-series (graphicalVAR) network estimation."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from psynet.datasets import make_var_data
from psynet.timeseries._validation import validate_ts_data, make_lag_matrix
from psynet.timeseries._var import estimate_temporal
from psynet.timeseries._contemporaneous import estimate_contemporaneous
from psynet.timeseries import estimate_var_network, TSNetwork
from psynet.plotting.ts_plot import plot_ts_networks


# ── Validation ──────────────────────────────────────────────────────


class TestValidation:
    def test_valid_data(self, var_data):
        cols = validate_ts_data(var_data)
        assert cols == list(var_data.columns)

    def test_too_few_timepoints(self):
        df = pd.DataFrame({"V1": [1.0], "V2": [2.0]})
        with pytest.raises(ValueError, match="at least 2"):
            validate_ts_data(df)

    def test_nan_raises(self, var_data):
        bad = var_data.copy()
        bad.iloc[5, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_ts_data(bad)

    def test_beep_day_columns(self, var_data):
        df = var_data.copy()
        df["beep"] = range(len(df))
        df["day"] = 1
        cols = validate_ts_data(df, beep="beep", day="day")
        assert "beep" not in cols
        assert "day" not in cols

    def test_missing_beep_col(self, var_data):
        with pytest.raises(ValueError, match="not found"):
            validate_ts_data(var_data, beep="nonexistent")


# ── Lag Matrix ──────────────────────────────────────────────────────


class TestLagMatrix:
    def test_shapes(self, var_data):
        cols = list(var_data.columns)
        X, Y = make_lag_matrix(var_data, cols)
        assert X.shape == (len(var_data) - 1, len(cols))
        assert Y.shape == X.shape

    def test_lag_relationship(self, var_data):
        cols = list(var_data.columns)
        X, Y = make_lag_matrix(var_data, cols)
        # X[t] should be data[t], Y[t] should be data[t+1]
        np.testing.assert_array_equal(X[0], var_data[cols].iloc[0].values)
        np.testing.assert_array_equal(Y[0], var_data[cols].iloc[1].values)

    def test_beep_day_gap_handling(self):
        """Observations across day boundaries should be excluded."""
        df = pd.DataFrame({
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "V2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "beep": [1, 2, 3, 1, 2],
            "day": [1, 1, 1, 2, 2],
        })
        cols = ["V1", "V2"]
        X, Y = make_lag_matrix(df, cols, beep="beep", day="day")
        # Valid pairs: (row0→1), (row1→2), (row3→4)  — row2→3 crosses day boundary
        assert X.shape[0] == 3
        assert Y.shape[0] == 3


# ── Temporal Estimation ─────────────────────────────────────────────


class TestTemporalEstimation:
    def test_directed_network(self, var_data):
        cols = list(var_data.columns)
        X, Y = make_lag_matrix(var_data, cols)
        net, residuals = estimate_temporal(X, Y, cols, cv=3)
        assert net.directed is True
        assert net.method == "graphicalVAR"

    def test_correct_shape(self, var_data):
        cols = list(var_data.columns)
        X, Y = make_lag_matrix(var_data, cols)
        net, residuals = estimate_temporal(X, Y, cols, cv=3)
        p = len(cols)
        assert net.adjacency.shape == (p, p)
        assert residuals.shape == Y.shape

    def test_sparsity(self, var_data):
        cols = list(var_data.columns)
        X, Y = make_lag_matrix(var_data, cols)
        net, _ = estimate_temporal(X, Y, cols, cv=3)
        # With regularization, some entries should be zero
        assert np.sum(net.adjacency == 0) > 0


# ── Contemporaneous Estimation ──────────────────────────────────────


class TestContemporaneousEstimation:
    def test_undirected_symmetric(self, var_data):
        cols = list(var_data.columns)
        X, Y = make_lag_matrix(var_data, cols)
        _, residuals = estimate_temporal(X, Y, cols, cv=3)
        net = estimate_contemporaneous(residuals, cols)
        assert net.directed is False
        np.testing.assert_array_almost_equal(
            net.adjacency, net.adjacency.T
        )

    def test_partial_correlations_bounded(self, var_data):
        cols = list(var_data.columns)
        X, Y = make_lag_matrix(var_data, cols)
        _, residuals = estimate_temporal(X, Y, cols, cv=3)
        net = estimate_contemporaneous(residuals, cols)
        assert np.all(np.abs(net.adjacency) <= 1.0 + 1e-10)


# ── Integration: estimate_var_network ───────────────────────────────


class TestEstimateVarNetwork:
    def test_returns_tsnetwork(self, var_data):
        ts = estimate_var_network(var_data, cv=3, n_lambda=20)
        assert isinstance(ts, TSNetwork)

    def test_attributes(self, var_data):
        ts = estimate_var_network(var_data, cv=3, n_lambda=20)
        assert ts.method == "graphicalVAR"
        assert ts.n_timepoints == len(var_data)
        assert ts.n_observations == len(var_data) - 1
        assert ts.n_nodes == len(var_data.columns)
        assert ts.labels == list(var_data.columns)

    def test_temporal_directed(self, var_data):
        ts = estimate_var_network(var_data, cv=3, n_lambda=20)
        assert ts.temporal.directed is True

    def test_contemporaneous_undirected(self, var_data):
        ts = estimate_var_network(var_data, cv=3, n_lambda=20)
        assert ts.contemporaneous.directed is False

    def test_centrality(self, var_data):
        ts = estimate_var_network(var_data, cv=3, n_lambda=20)
        cent = ts.centrality()
        assert "network" in cent.columns
        assert set(cent["network"].unique()) == {"temporal", "contemporaneous"}


# ── make_var_data ───────────────────────────────────────────────────


class TestMakeVarData:
    def test_shape(self):
        df = make_var_data(n_timepoints=100, p=5)
        assert df.shape == (100, 5)

    def test_columns(self):
        df = make_var_data(p=4)
        assert list(df.columns) == ["V1", "V2", "V3", "V4"]

    def test_stationarity(self):
        """Generated data should be finite (non-exploding)."""
        df = make_var_data(n_timepoints=1000, p=6)
        assert np.all(np.isfinite(df.values))

    def test_integrates_with_estimation(self):
        df = make_var_data(n_timepoints=200, p=4, seed=99)
        ts = estimate_var_network(df, cv=3, n_lambda=20)
        assert isinstance(ts, TSNetwork)
        assert ts.n_nodes == 4


# ── Plotting ────────────────────────────────────────────────────────


class TestTSPlotting:
    def test_plot_returns_figure(self, var_data):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        ts = estimate_var_network(var_data, cv=3, n_lambda=20)
        fig = ts.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_ts_networks_direct(self, var_data):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        ts = estimate_var_network(var_data, cv=3, n_lambda=20)
        fig = plot_ts_networks(ts, layout="circular")
        assert isinstance(fig, Figure)
        plt.close(fig)
