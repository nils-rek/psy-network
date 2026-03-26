"""Tests for multilevel VAR network estimation."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from psynet.datasets import make_multilevel_data
from psynet.multilevel._validation import make_multilevel_lag_data, validate_multilevel_data
from psynet.multilevel import estimate_multilevel_network, MultilevelNetwork


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestMultilevelValidation:
    def test_missing_subject_column(self, multilevel_data):
        with pytest.raises(ValueError, match="subject column"):
            validate_multilevel_data(multilevel_data, "nonexistent")

    def test_too_few_subjects(self):
        df = pd.DataFrame({"subject": ["S1"] * 10, "V1": range(10), "V2": range(10)})
        with pytest.raises(ValueError, match="at least 2 subjects"):
            validate_multilevel_data(df, "subject")

    def test_nan_values_allowed(self, multilevel_data):
        """Multilevel validation now allows NaN (handled per-subject during lag construction)."""
        multilevel_data.iloc[0, 0] = np.nan
        var_cols = validate_multilevel_data(multilevel_data, "subject")
        assert len(var_cols) >= 2

    def test_too_few_observations_per_subject(self):
        df = pd.DataFrame({
            "subject": ["S1", "S1", "S2", "S2", "S2"],
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "V2": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        with pytest.raises(ValueError, match="fewer than 3"):
            validate_multilevel_data(df, "subject")

    def test_valid_returns_var_cols(self, multilevel_data):
        var_cols = validate_multilevel_data(multilevel_data, "subject", beep="beep")
        assert var_cols == ["V1", "V2", "V3", "V4"]


# ---------------------------------------------------------------------------
# Lag data construction
# ---------------------------------------------------------------------------

class TestMultilevelLagData:
    def test_shape(self, multilevel_data):
        var_cols = validate_multilevel_data(multilevel_data, "subject", beep="beep")
        lag = make_multilevel_lag_data(multilevel_data, var_cols, "subject", beep="beep")
        # Each subject contributes (n_timepoints - 1) rows
        n_subjects = multilevel_data["subject"].nunique()
        n_tp = len(multilevel_data) // n_subjects
        expected_rows = n_subjects * (n_tp - 1)
        assert len(lag) == expected_rows

    def test_columns(self, multilevel_data):
        var_cols = validate_multilevel_data(multilevel_data, "subject", beep="beep")
        lag = make_multilevel_lag_data(multilevel_data, var_cols, "subject", beep="beep")
        for col in var_cols:
            assert col in lag.columns
            assert f"{col}_lag" in lag.columns
        assert "subject" in lag.columns

    def test_subject_boundaries(self, multilevel_data):
        """Lag data should not cross subject boundaries."""
        var_cols = validate_multilevel_data(multilevel_data, "subject", beep="beep")
        lag = make_multilevel_lag_data(multilevel_data, var_cols, "subject", beep="beep")
        # All rows should have a valid subject
        assert lag["subject"].isin(multilevel_data["subject"].unique()).all()

    def test_beep_boundaries(self):
        """Non-consecutive beeps should be excluded."""
        df = pd.DataFrame({
            "subject": ["S1"] * 5 + ["S2"] * 5,
            "beep": [1, 2, 4, 5, 6, 1, 2, 3, 4, 5],  # gap at beep 3 for S1
            "V1": np.random.default_rng(0).standard_normal(10),
            "V2": np.random.default_rng(1).standard_normal(10),
        })
        var_cols = ["V1", "V2"]
        lag = make_multilevel_lag_data(df, var_cols, "subject", beep="beep")
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
        lag = make_multilevel_lag_data(df, var_cols, "subject", beep="beep", day="day")
        # Each subject: 2 pairs per day × 2 days = 4 pairs, × 2 subjects = 8
        assert len(lag) == 8


# ---------------------------------------------------------------------------
# Temporal estimation
# ---------------------------------------------------------------------------

class TestMultilevelTemporalEstimation:
    def test_fixed_coef_shape(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        p = 4
        assert result.temporal.adjacency.shape == (p, p)

    def test_temporal_is_directed(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.temporal.directed is True

    def test_pvalues_shape(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.pvalues.shape == (4, 4)

    def test_pvalues_range(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert np.all(result.pvalues >= 0)
        assert np.all(result.pvalues <= 1)

    def test_per_subject_networks(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert len(result.subject_temporal) == 10
        for s, net in result.subject_temporal.items():
            assert net.adjacency.shape == (4, 4)
            assert net.directed is True


# ---------------------------------------------------------------------------
# Contemporaneous
# ---------------------------------------------------------------------------

class TestMultilevelContemporaneous:
    def test_undirected(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.contemporaneous.directed is False

    def test_symmetric(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        adj = result.contemporaneous.adjacency
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_partial_correlation_bounds(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        adj = result.contemporaneous.adjacency
        assert np.all(np.abs(adj) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Between-subjects
# ---------------------------------------------------------------------------

class TestMultilevelBetweenSubjects:
    def test_undirected(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.between_subjects.directed is False

    def test_symmetric(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        adj = result.between_subjects.adjacency
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_n_observations_equals_n_subjects(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.between_subjects.n_observations == 10


# ---------------------------------------------------------------------------
# Integration: estimate_multilevel_network
# ---------------------------------------------------------------------------

class TestEstimateMultilevelNetwork:
    def test_returns_multilevel_network(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert isinstance(result, MultilevelNetwork)

    def test_method(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.method == "mlVAR"

    def test_all_attributes(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.n_nodes == 4
        assert result.n_subjects == 10
        assert result.labels == ["V1", "V2", "V3", "V4"]
        assert len(result.subject_ids) == 10
        assert isinstance(result.fit_info, dict)

    def test_centrality(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        cent = result.centrality()
        assert "network" in cent.columns
        assert set(cent["network"].unique()) == {"temporal", "contemporaneous", "between_subjects"}

    def test_subject_network_accessor(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        net = result.subject_network(result.subject_ids[0])
        assert net.directed is True
        assert net.adjacency.shape == (4, 4)

    def test_subject_network_invalid(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        with pytest.raises(KeyError, match="not found"):
            result.subject_network("NONEXISTENT")


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class TestMakeMultilevelData:
    def test_shape(self):
        df = make_multilevel_data(n_subjects=5, n_timepoints=20, p=3, seed=0)
        assert len(df) == 5 * 20
        assert "subject" in df.columns
        assert "beep" in df.columns
        assert all(f"V{i+1}" in df.columns for i in range(3))

    def test_n_subjects(self):
        df = make_multilevel_data(n_subjects=8, n_timepoints=10, p=3, seed=0)
        assert df["subject"].nunique() == 8

    def test_no_nan(self):
        df = make_multilevel_data(seed=0)
        assert not df.isna().any().any()

    def test_columns(self):
        df = make_multilevel_data(p=5, seed=0)
        expected = ["V1", "V2", "V3", "V4", "V5", "subject", "beep"]
        assert sorted(df.columns) == sorted(expected)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class TestMultilevelPlotting:
    def test_returns_figure(self, multilevel_data):
        from matplotlib.figure import Figure
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        fig = result.plot()
        assert isinstance(fig, Figure)

    def test_three_network_panels(self, multilevel_data):
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        fig = result.plot()
        # 3 network panels + 1 legend panel
        assert len(fig.axes) == 4


# ---------------------------------------------------------------------------
# P-value thresholding
# ---------------------------------------------------------------------------

class TestTemporalPvalueThresholding:
    def test_default_alpha_zeros_nonsignificant(self, multilevel_data):
        """Default temporal_alpha=0.05 should zero out non-significant edges."""
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        adj = result.temporal.adjacency
        pvals = result.pvalues
        # All non-zero edges should have p <= 0.05
        nonzero_mask = adj != 0
        if nonzero_mask.any():
            assert np.all(pvals[nonzero_mask] <= 0.05)

    def test_alpha_none_preserves_all_edges(self, multilevel_data):
        """temporal_alpha=None should keep all non-threshold coefficients."""
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", temporal_alpha=None,
        )
        # With no p-value thresholding, unthresholded_temporal should be None
        assert result.unthresholded_temporal is None

    def test_unthresholded_temporal_stored(self, multilevel_data):
        """When temporal_alpha is set, unthresholded_temporal should be stored."""
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        assert result.unthresholded_temporal is not None
        # Unthresholded should have >= as many non-zero edges
        n_thresh = np.count_nonzero(result.temporal.adjacency)
        n_unthresh = np.count_nonzero(result.unthresholded_temporal.adjacency)
        assert n_unthresh >= n_thresh

    def test_temporal_thresholded_method(self, multilevel_data):
        """Post-hoc thresholding should produce valid networks."""
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", temporal_alpha=None,
        )
        threshed = result.temporal_thresholded(alpha=0.05)
        assert threshed.directed is True
        assert threshed.adjacency.shape == result.temporal.adjacency.shape
        # All non-zero edges should have p <= 0.05
        nonzero = threshed.adjacency != 0
        if nonzero.any():
            assert np.all(result.pvalues[nonzero] <= 0.05)

    def test_subject_temporal_inherits_mask(self, multilevel_data):
        """Per-subject temporal networks should also be masked by p-value."""
        result = estimate_multilevel_network(multilevel_data, "subject", beep="beep")
        pval_mask = result.pvalues > 0.05
        for s, net in result.subject_temporal.items():
            # Where fixed-effect p > alpha, subject network should also be zero
            assert np.all(net.adjacency[pval_mask] == 0)


# ---------------------------------------------------------------------------
# Random effects structure
# ---------------------------------------------------------------------------

class TestRandomEffectsStructure:
    def test_fixed_mode(self, multilevel_data):
        """temporal='fixed' should estimate successfully."""
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep",
            temporal="fixed", temporal_alpha=None,
        )
        assert result.temporal.adjacency.shape == (4, 4)

    def test_fixed_mode_subjects_equal_fixed(self, multilevel_data):
        """In 'fixed' mode, all per-subject networks should equal fixed effects."""
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep",
            temporal="fixed", temporal_alpha=None,
        )
        for s, net in result.subject_temporal.items():
            np.testing.assert_array_almost_equal(
                net.adjacency, result.temporal.adjacency,
            )

    def test_invalid_temporal_raises(self, multilevel_data):
        with pytest.raises(ValueError, match="correlated.*orthogonal.*fixed"):
            estimate_multilevel_network(
                multilevel_data, "subject", beep="beep", temporal="bad",
            )


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestMultilevelNanHandling:
    def test_estimation_with_nan(self):
        """Multilevel estimation should handle scattered NaN values."""
        from psynet.datasets import make_multilevel_data
        df = make_multilevel_data(n_subjects=5, n_timepoints=30, p=3, seed=0)
        # Introduce NaN
        rng = np.random.default_rng(99)
        nan_idx = rng.choice(len(df), size=5, replace=False)
        df.iloc[nan_idx, 0] = np.nan
        result = estimate_multilevel_network(
            df, "subject", beep="beep", temporal="fixed",
        )
        assert isinstance(result, MultilevelNetwork)

    def test_nan_warning_without_beep(self):
        """Should warn when dropping all-NaN rows without beep column."""
        from psynet.datasets import make_multilevel_data
        df = make_multilevel_data(n_subjects=5, n_timepoints=30, p=3, seed=0)
        # Set ALL variable columns to NaN for one row (triggers how="all" drop)
        var_cols = validate_multilevel_data(df, "subject")
        for col in var_cols:
            df.iloc[0, df.columns.get_loc(col)] = np.nan
        with pytest.warns(UserWarning, match="NaN rows were dropped"):
            make_multilevel_lag_data(df, var_cols, "subject")

    def test_nan_no_warning_with_beep(self):
        """Should NOT warn when dropping NaN with beep column provided."""
        import warnings
        from psynet.datasets import make_multilevel_data
        df = make_multilevel_data(n_subjects=5, n_timepoints=30, p=3, seed=0)
        df.iloc[0, 0] = np.nan
        var_cols = validate_multilevel_data(df, "subject", beep="beep")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            make_multilevel_lag_data(df, var_cols, "subject", beep="beep")
