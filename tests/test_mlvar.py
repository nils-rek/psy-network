"""Tests for multilevel VAR (mlVAR) network estimation."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from psynet.datasets import make_mlvar_data
from psynet.mlvar._validation import make_mlvar_lag_data, validate_mlvar_data
from psynet.mlvar import estimate_mlvar_network, MLVARNetwork


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestMLVARValidation:
    def test_missing_subject_column(self, mlvar_data):
        with pytest.raises(ValueError, match="subject column"):
            validate_mlvar_data(mlvar_data, "nonexistent")

    def test_too_few_subjects(self):
        df = pd.DataFrame({"subject": ["S1"] * 10, "V1": range(10), "V2": range(10)})
        with pytest.raises(ValueError, match="at least 2 subjects"):
            validate_mlvar_data(df, "subject")

    def test_nan_values(self, mlvar_data):
        mlvar_data.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_mlvar_data(mlvar_data, "subject")

    def test_too_few_observations_per_subject(self):
        df = pd.DataFrame({
            "subject": ["S1", "S1", "S2", "S2", "S2"],
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "V2": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        with pytest.raises(ValueError, match="fewer than 3"):
            validate_mlvar_data(df, "subject")

    def test_valid_returns_var_cols(self, mlvar_data):
        var_cols = validate_mlvar_data(mlvar_data, "subject", beep="beep")
        assert var_cols == ["V1", "V2", "V3", "V4"]


# ---------------------------------------------------------------------------
# Lag data construction
# ---------------------------------------------------------------------------

class TestMLVARLagData:
    def test_shape(self, mlvar_data):
        var_cols = validate_mlvar_data(mlvar_data, "subject", beep="beep")
        lag = make_mlvar_lag_data(mlvar_data, var_cols, "subject", beep="beep")
        # Each subject contributes (n_timepoints - 1) rows
        n_subjects = mlvar_data["subject"].nunique()
        n_tp = len(mlvar_data) // n_subjects
        expected_rows = n_subjects * (n_tp - 1)
        assert len(lag) == expected_rows

    def test_columns(self, mlvar_data):
        var_cols = validate_mlvar_data(mlvar_data, "subject", beep="beep")
        lag = make_mlvar_lag_data(mlvar_data, var_cols, "subject", beep="beep")
        for col in var_cols:
            assert col in lag.columns
            assert f"{col}_lag" in lag.columns
        assert "subject" in lag.columns

    def test_subject_boundaries(self, mlvar_data):
        """Lag data should not cross subject boundaries."""
        var_cols = validate_mlvar_data(mlvar_data, "subject", beep="beep")
        lag = make_mlvar_lag_data(mlvar_data, var_cols, "subject", beep="beep")
        # All rows should have a valid subject
        assert lag["subject"].isin(mlvar_data["subject"].unique()).all()

    def test_beep_boundaries(self):
        """Non-consecutive beeps should be excluded."""
        df = pd.DataFrame({
            "subject": ["S1"] * 5 + ["S2"] * 5,
            "beep": [1, 2, 4, 5, 6, 1, 2, 3, 4, 5],  # gap at beep 3 for S1
            "V1": np.random.default_rng(0).standard_normal(10),
            "V2": np.random.default_rng(1).standard_normal(10),
        })
        var_cols = ["V1", "V2"]
        lag = make_mlvar_lag_data(df, var_cols, "subject", beep="beep")
        # S1: (1,2), (4,5), (5,6) = 3 pairs; S2: (1,2), (2,3), (3,4), (4,5) = 4 pairs
        assert len(lag) == 7

    def test_day_boundaries(self):
        """Lag should not cross day boundaries."""
        df = pd.DataFrame({
            "subject": ["S1"] * 6 + ["S2"] * 6,
            "beep": [1, 2, 3, 1, 2, 3] * 2,
            "day": [1, 1, 1, 2, 2, 2] * 2,
            "V1": np.random.default_rng(0).standard_normal(12),
            "V2": np.random.default_rng(1).standard_normal(12),
        })
        var_cols = ["V1", "V2"]
        lag = make_mlvar_lag_data(df, var_cols, "subject", beep="beep", day="day")
        # Each subject: 2 pairs per day × 2 days = 4 pairs, × 2 subjects = 8
        assert len(lag) == 8


# ---------------------------------------------------------------------------
# Temporal estimation
# ---------------------------------------------------------------------------

class TestMLVARTemporalEstimation:
    def test_fixed_coef_shape(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        p = 4
        assert result.temporal.adjacency.shape == (p, p)

    def test_temporal_is_directed(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert result.temporal.directed is True

    def test_pvalues_shape(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert result.pvalues.shape == (4, 4)

    def test_pvalues_range(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert np.all(result.pvalues >= 0)
        assert np.all(result.pvalues <= 1)

    def test_per_subject_networks(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert len(result.subject_temporal) == 10
        for s, net in result.subject_temporal.items():
            assert net.adjacency.shape == (4, 4)
            assert net.directed is True


# ---------------------------------------------------------------------------
# Contemporaneous
# ---------------------------------------------------------------------------

class TestMLVARContemporaneous:
    def test_undirected(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert result.contemporaneous.directed is False

    def test_symmetric(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        adj = result.contemporaneous.adjacency
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_partial_correlation_bounds(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        adj = result.contemporaneous.adjacency
        assert np.all(np.abs(adj) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Between-subjects
# ---------------------------------------------------------------------------

class TestMLVARBetweenSubjects:
    def test_undirected(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert result.between_subjects.directed is False

    def test_symmetric(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        adj = result.between_subjects.adjacency
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_n_observations_equals_n_subjects(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert result.between_subjects.n_observations == 10


# ---------------------------------------------------------------------------
# Integration: estimate_mlvar_network
# ---------------------------------------------------------------------------

class TestEstimateMLVARNetwork:
    def test_returns_mlvar_network(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert isinstance(result, MLVARNetwork)

    def test_method(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert result.method == "mlVAR"

    def test_all_attributes(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        assert result.n_nodes == 4
        assert result.n_subjects == 10
        assert result.labels == ["V1", "V2", "V3", "V4"]
        assert len(result.subject_ids) == 10
        assert isinstance(result.fit_info, dict)

    def test_centrality(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        cent = result.centrality()
        assert "network" in cent.columns
        assert set(cent["network"].unique()) == {"temporal", "contemporaneous", "between_subjects"}

    def test_subject_network_accessor(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        net = result.subject_network(result.subject_ids[0])
        assert net.directed is True
        assert net.adjacency.shape == (4, 4)

    def test_subject_network_invalid(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        with pytest.raises(KeyError, match="not found"):
            result.subject_network("NONEXISTENT")


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class TestMakeMLVARData:
    def test_shape(self):
        df = make_mlvar_data(n_subjects=5, n_timepoints=20, p=3, seed=0)
        assert len(df) == 5 * 20
        assert "subject" in df.columns
        assert "beep" in df.columns
        assert all(f"V{i+1}" in df.columns for i in range(3))

    def test_n_subjects(self):
        df = make_mlvar_data(n_subjects=8, n_timepoints=10, p=3, seed=0)
        assert df["subject"].nunique() == 8

    def test_no_nan(self):
        df = make_mlvar_data(seed=0)
        assert not df.isna().any().any()

    def test_columns(self):
        df = make_mlvar_data(p=5, seed=0)
        expected = ["V1", "V2", "V3", "V4", "V5", "subject", "beep"]
        assert sorted(df.columns) == sorted(expected)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class TestMLVARPlotting:
    def test_returns_figure(self, mlvar_data):
        from matplotlib.figure import Figure
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        fig = result.plot()
        assert isinstance(fig, Figure)

    def test_three_axes(self, mlvar_data):
        result = estimate_mlvar_network(mlvar_data, "subject", beep="beep")
        fig = result.plot()
        assert len(fig.axes) == 3
