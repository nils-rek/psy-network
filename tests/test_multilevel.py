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

    def test_between_gamma_parameter(self, multilevel_data):
        """Lower between_gamma should produce at least as many edges."""
        result_default = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep",
            temporal="fixed", gamma=0.5,
        )
        result_low = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep",
            temporal="fixed", gamma=0.5, between_gamma=0.25,
        )
        edges_default = np.count_nonzero(result_default.between_subjects.adjacency)
        edges_low = np.count_nonzero(result_low.between_subjects.adjacency)
        assert edges_low >= edges_default

    def test_between_gamma_default_uses_gamma(self, multilevel_data):
        """When between_gamma is not set, between-subjects uses gamma."""
        r1 = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep",
            temporal="fixed", gamma=0.5,
        )
        r2 = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep",
            temporal="fixed", gamma=0.5, between_gamma=0.5,
        )
        np.testing.assert_array_almost_equal(
            r1.between_subjects.adjacency, r2.between_subjects.adjacency,
        )


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

    def test_warning_fallback_to_simpler_re(self, multilevel_data):
        """Severe convergence warnings should trigger fallback to simpler RE."""
        from unittest.mock import patch, MagicMock
        from psynet.multilevel._temporal import _fit_one_dv, _RE_FALLBACK_CHAIN

        var_cols = [c for c in multilevel_data.columns
                    if c not in ("subject", "beep")]
        from psynet.multilevel._validation import make_multilevel_lag_data
        lag_data = make_multilevel_lag_data(
            multilevel_data, var_cols, "subject", beep="beep",
        )

        severe_msg = "The Hessian matrix at the estimated parameter values is not positive definite."

        def mock_try_fit(model_kwargs, method="lbfgs", start_params=None):
            """Return severe warnings for correlated/orthogonal, clean for fixed."""
            import statsmodels.formula.api as smf
            model = smf.mixedlm(**model_kwargs)
            result = model.fit(reml=True, method=method)
            # correlated has lag cols in re_formula; orthogonal has vc_formula
            re_formula = model_kwargs.get("re_formula", "1")
            has_vc = "vc_formula" in model_kwargs
            is_fixed = re_formula == "1" and not has_vc
            if is_fixed:
                return result, []
            else:
                return result, [severe_msg]

        with patch("psynet.multilevel._temporal._try_fit", side_effect=mock_try_fit):
            info = _fit_one_dv(0, var_cols, lag_data, "subject",
                               temporal_re="correlated")

        assert info["actual_re"] == "fixed"

    def test_accept_warned_result_at_simplest_re(self, multilevel_data):
        """When all RE structures have warnings, accept the simplest one."""
        from unittest.mock import patch
        from psynet.multilevel._temporal import _fit_one_dv

        var_cols = [c for c in multilevel_data.columns
                    if c not in ("subject", "beep")]
        from psynet.multilevel._validation import make_multilevel_lag_data
        lag_data = make_multilevel_lag_data(
            multilevel_data, var_cols, "subject", beep="beep",
        )

        severe_msg = "The Hessian matrix at the estimated parameter values is not positive definite."

        def mock_try_fit(model_kwargs, method="lbfgs", start_params=None):
            import statsmodels.formula.api as smf
            model = smf.mixedlm(**model_kwargs)
            result = model.fit(reml=True, method=method)
            return result, [severe_msg]

        with patch("psynet.multilevel._temporal._try_fit", side_effect=mock_try_fit):
            info = _fit_one_dv(0, var_cols, lag_data, "subject",
                               temporal_re="correlated")

        # Should accept the result at "fixed" (simplest), even with warnings
        assert info["actual_re"] == "fixed"


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


# ---------------------------------------------------------------------------
# Within-subject scaling
# ---------------------------------------------------------------------------

class TestScaleParameter:
    def test_scale_runs(self, multilevel_data):
        """scale=True should produce a valid MultilevelNetwork."""
        # Widen the scale to simulate 0-100 VAS items
        wide_data = multilevel_data.copy()
        var_cols = [c for c in wide_data.columns if c not in ("subject", "beep")]
        wide_data[var_cols] = wide_data[var_cols] * 100
        result = estimate_multilevel_network(
            wide_data, "subject", beep="beep", temporal="fixed", scale=True,
        )
        assert isinstance(result, MultilevelNetwork)
        assert result.temporal.adjacency.shape == (4, 4)

    def test_scale_does_not_mutate_input(self, multilevel_data):
        """scale=True must not modify the caller's DataFrame."""
        original = multilevel_data.copy()
        estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", temporal="fixed", scale=True,
        )
        pd.testing.assert_frame_equal(multilevel_data, original)

    def test_scale_false_is_default(self, multilevel_data):
        """Calling without scale should match scale=False explicitly."""
        r1 = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", temporal="fixed",
        )
        r2 = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", temporal="fixed", scale=False,
        )
        np.testing.assert_array_almost_equal(
            r1.temporal.adjacency, r2.temporal.adjacency,
        )


# ---------------------------------------------------------------------------
# Optimizer chain improvements
# ---------------------------------------------------------------------------

class TestOptimizerChain:
    def test_multiple_optimizers_tried_before_fallback(self, multilevel_data):
        """All optimizers should be tried before falling back to simpler RE."""
        from unittest.mock import patch, call
        from psynet.multilevel._temporal import (
            _fit_one_dv, _OPTIMIZER_CHAIN,
        )

        var_cols = [c for c in multilevel_data.columns
                    if c not in ("subject", "beep")]
        from psynet.multilevel._validation import make_multilevel_lag_data
        lag_data = make_multilevel_lag_data(
            multilevel_data, var_cols, "subject", beep="beep",
        )

        severe_msg = "optimization failed"
        call_count = {"n": 0}

        def mock_try_fit(model_kwargs, method="lbfgs", start_params=None):
            call_count["n"] += 1
            import statsmodels.formula.api as smf
            model = smf.mixedlm(**model_kwargs)
            result = model.fit(reml=True, method="lbfgs")
            # Only succeed on the last optimizer for correlated RE
            re_formula = model_kwargs.get("re_formula", "1")
            has_vc = "vc_formula" in model_kwargs
            is_complex = re_formula != "1" or has_vc
            if is_complex and method != _OPTIMIZER_CHAIN[-1]:
                return result, [severe_msg]
            return result, []

        with patch("psynet.multilevel._temporal._try_fit", side_effect=mock_try_fit):
            info = _fit_one_dv(0, var_cols, lag_data, "subject",
                               temporal_re="correlated")

        # Should have converged on correlated (the nm optimizer succeeds)
        assert info["actual_re"] == "correlated"
        # All 4 optimizers should have been tried for correlated
        assert call_count["n"] == len(_OPTIMIZER_CHAIN)

    def test_warm_start_params_passed(self, multilevel_data):
        """OLS warm-start should be computed and passed to _try_fit."""
        from unittest.mock import patch
        from psynet.multilevel._temporal import _fit_one_dv

        var_cols = [c for c in multilevel_data.columns
                    if c not in ("subject", "beep")]
        from psynet.multilevel._validation import make_multilevel_lag_data
        lag_data = make_multilevel_lag_data(
            multilevel_data, var_cols, "subject", beep="beep",
        )

        received_start_params = []

        def mock_try_fit(model_kwargs, method="lbfgs", start_params=None):
            received_start_params.append(start_params)
            import statsmodels.formula.api as smf
            model = smf.mixedlm(**model_kwargs)
            result = model.fit(reml=True, method="lbfgs")
            return result, []  # succeed immediately

        with patch("psynet.multilevel._temporal._try_fit", side_effect=mock_try_fit):
            _fit_one_dv(0, var_cols, lag_data, "subject",
                        temporal_re="fixed")

        # At least one call should have received start_params
        assert any(sp is not None for sp in received_start_params)


class TestAutoReStructure:
    def test_downgrades_large_p(self):
        """Should downgrade correlated to orthogonal for p > 8."""
        from psynet.multilevel._temporal import _auto_re_structure
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _auto_re_structure(10, "correlated")
        assert result == "orthogonal"

    def test_preserves_small_p(self):
        """Should keep correlated for p <= 8."""
        from psynet.multilevel._temporal import _auto_re_structure
        result = _auto_re_structure(6, "correlated")
        assert result == "correlated"

    def test_preserves_orthogonal(self):
        """Should not change orthogonal regardless of p."""
        from psynet.multilevel._temporal import _auto_re_structure
        assert _auto_re_structure(20, "orthogonal") == "orthogonal"

    def test_preserves_fixed(self):
        """Should not change fixed regardless of p."""
        from psynet.multilevel._temporal import _auto_re_structure
        assert _auto_re_structure(20, "fixed") == "fixed"

    def test_auto_re_disabled(self, multilevel_data):
        """auto_re=False should not downgrade RE structure."""
        # Use 10 variables to trigger auto-downgrade
        from psynet.datasets import make_multilevel_data
        df = make_multilevel_data(n_subjects=5, n_timepoints=30, p=10, seed=0)
        # With auto_re=False and temporal="fixed", it should use "fixed" as-is
        result = estimate_multilevel_network(
            df, "subject", beep="beep", temporal="fixed", auto_re=False,
        )
        assert isinstance(result, MultilevelNetwork)

    def test_emits_warning_on_downgrade(self):
        """Should emit a UserWarning when downgrading."""
        from psynet.multilevel._temporal import _auto_re_structure
        with pytest.warns(UserWarning, match="Defaulting to 'orthogonal'"):
            _auto_re_structure(10, "correlated")


# ---------------------------------------------------------------------------
# Engine parameter
# ---------------------------------------------------------------------------

class TestEngineParameter:
    def test_invalid_engine_raises(self, multilevel_data):
        with pytest.raises(ValueError, match="engine must be"):
            estimate_multilevel_network(
                multilevel_data, "subject", beep="beep", engine="bad",
            )

    def test_default_engine_is_statsmodels(self, multilevel_data):
        """Default engine should work (statsmodels)."""
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", temporal="fixed",
        )
        assert isinstance(result, MultilevelNetwork)

    def test_lme4_without_rpy2_raises(self, multilevel_data):
        """engine='lme4' without rpy2 should raise ImportError."""
        from unittest.mock import patch
        with patch.dict("sys.modules", {"rpy2": None, "rpy2.robjects": None}):
            with pytest.raises(ImportError, match="rpy2"):
                estimate_multilevel_network(
                    multilevel_data, "subject", beep="beep", engine="lme4",
                )


# ---------------------------------------------------------------------------
# lme4 backend unit tests (skip if R not available)
# ---------------------------------------------------------------------------

_has_lme4 = False
try:
    import rpy2.robjects
    from rpy2.robjects.packages import importr
    importr("lme4")
    importr("lmerTest")
    _has_lme4 = True
except Exception:
    pass


@pytest.mark.skipif(not _has_lme4, reason="rpy2 + lme4/lmerTest not available")
class TestLme4Backend:
    def test_lme4_produces_valid_result(self, multilevel_data):
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", engine="lme4",
        )
        assert isinstance(result, MultilevelNetwork)
        assert result.temporal.adjacency.shape == (4, 4)

    def test_lme4_matches_output_format(self, multilevel_data):
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", engine="lme4",
        )
        assert result.pvalues.shape == (4, 4)
        assert len(result.subject_temporal) == 10
        assert result.fit_info is not None

    def test_lme4_pvalues_range(self, multilevel_data):
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep", engine="lme4",
        )
        assert np.all(result.pvalues >= 0)
        assert np.all(result.pvalues <= 1)

    def test_lme4_fixed_re(self, multilevel_data):
        result = estimate_multilevel_network(
            multilevel_data, "subject", beep="beep",
            engine="lme4", temporal="fixed",
        )
        assert result.temporal.adjacency.shape == (4, 4)

    def test_build_lmer_formula_correlated(self):
        from psynet.multilevel._lme4_backend import _build_lmer_formula
        f = _build_lmer_formula("V1", ["V1_lag", "V2_lag"], "subject", "correlated")
        assert "(V1_lag + V2_lag | subject)" in f

    def test_build_lmer_formula_orthogonal(self):
        from psynet.multilevel._lme4_backend import _build_lmer_formula
        f = _build_lmer_formula("V1", ["V1_lag", "V2_lag"], "subject", "orthogonal")
        assert "(1 | subject)" in f
        assert "(0 + V1_lag | subject)" in f

    def test_build_lmer_formula_fixed(self):
        from psynet.multilevel._lme4_backend import _build_lmer_formula
        f = _build_lmer_formula("V1", ["V1_lag", "V2_lag"], "subject", "fixed")
        assert "(1 | subject)" in f
        assert "V1_lag" not in f.split("(1 | subject)")[1]
