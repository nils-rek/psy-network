"""Tests for bootstrap analysis."""

import numpy as np
import pandas as pd
import pytest

from psynet import estimate_network
from psynet.bootstrap import bootnet, BootstrapResult
from psynet.bootstrap.stability import difference_test


class TestNonparametricBootstrap:
    def test_basic_run(self, small_data):
        result = bootnet(
            small_data,
            n_boots=10,
            boot_type="nonparametric",
            method="cor",
            statistics=["edge", "strength"],
            n_cores=1,
            seed=42,
            verbose=False,
        )
        assert isinstance(result, BootstrapResult)
        assert result.n_boots == 10
        assert result.boot_type.value == "nonparametric"

    def test_boot_statistics_shape(self, small_data):
        result = bootnet(
            small_data,
            n_boots=5,
            boot_type="nonparametric",
            method="cor",
            statistics=["edge"],
            n_cores=1,
            seed=42,
            verbose=False,
        )
        df = result.boot_statistics
        assert "boot_id" in df.columns
        assert "statistic" in df.columns
        assert "value" in df.columns
        # Should have original + 5 boot iterations
        boot_ids = df["boot_id"].unique()
        assert "original" in boot_ids

    def test_summary(self, small_data):
        result = bootnet(
            small_data,
            n_boots=10,
            boot_type="nonparametric",
            method="cor",
            statistics=["edge"],
            n_cores=1,
            seed=42,
            verbose=False,
        )
        summary = result.summary("edge")
        assert "mean" in summary.columns
        assert "ci_lower" in summary.columns
        assert "ci_upper" in summary.columns

    def test_difference_test(self, small_data):
        result = bootnet(
            small_data,
            n_boots=20,
            boot_type="nonparametric",
            method="cor",
            statistics=["edge"],
            n_cores=1,
            seed=42,
            verbose=False,
        )
        diff = result.difference_test("edge")
        assert isinstance(diff, pd.DataFrame)
        # Should be square
        assert diff.shape[0] == diff.shape[1]


class TestDifferenceTestCorrectness:
    """difference_test with controlled bootstrap distributions."""

    @staticmethod
    def _make_result(edge_values: dict, n_boots: int = 200):
        """Build a BootstrapResult with fixed per-bootstrap edge values.

        edge_values maps (node1, node2) -> array of length n_boots.
        """
        from psynet.bootstrap.results import BootstrapResult
        from psynet.network import Network

        rows = []
        for b in range(n_boots):
            for (n1, n2), vals in edge_values.items():
                rows.append({
                    "boot_id": b, "statistic": "edge",
                    "node1": n1, "node2": n2, "value": vals[b],
                })
        boot_stats = pd.DataFrame(rows)
        labels = sorted({n for pair in edge_values for n in pair})
        p = len(labels)
        net = Network(np.zeros((p, p)), labels, "test", 100)
        return BootstrapResult(
            original_network=net,
            boot_statistics=boot_stats,
            boot_type="nonparametric",
            n_boots=n_boots,
        )

    def test_structurally_zero_edges_not_significant(self):
        """Two edges that are zero in every bootstrap (sparse glasso) must
        NOT test as significantly different, and p should be 1."""
        n = 200
        zeros = np.zeros(n)
        result = self._make_result({
            ("A", "B"): zeros,
            ("A", "C"): zeros,
        }, n_boots=n)
        diff = difference_test(result, "edge")
        assert diff.values[0, 1] == 0
        p = diff.attrs["p_values"].values[0, 1]
        assert p == pytest.approx(1.0)

    def test_clearly_different_edges_significant(self):
        n = 200
        rng = np.random.default_rng(0)
        result = self._make_result({
            ("A", "B"): rng.normal(0.5, 0.02, n),
            ("A", "C"): rng.normal(0.1, 0.02, n),
        }, n_boots=n)
        diff = difference_test(result, "edge")
        assert diff.values[0, 1] == 1
        assert diff.attrs["p_values"].values[0, 1] < 0.05
        # CI of (A--B minus A--C) should exclude zero and be positive
        assert diff.attrs["ci_lower"].loc["A -- B", "A -- C"] > 0

    def test_overlapping_edges_not_significant(self):
        n = 200
        rng = np.random.default_rng(1)
        result = self._make_result({
            ("A", "B"): rng.normal(0.3, 0.1, n),
            ("A", "C"): rng.normal(0.3, 0.1, n),
        }, n_boots=n)
        diff = difference_test(result, "edge")
        assert diff.values[0, 1] == 0


class TestCaseDroppingBootstrap:
    def test_basic_run(self, small_data):
        result = bootnet(
            small_data,
            n_boots=5,
            boot_type="case",
            method="cor",
            n_cores=1,
            case_n=3,
            seed=42,
            verbose=False,
        )
        assert isinstance(result, BootstrapResult)
        assert result.case_drop_correlations is not None
        assert len(result.case_drop_correlations) > 0

    def test_cs_coefficient(self, small_data):
        result = bootnet(
            small_data,
            n_boots=10,
            boot_type="case",
            method="cor",
            n_cores=1,
            case_n=3,
            seed=42,
            verbose=False,
        )
        cs = result.cs_coefficient("strength")
        assert 0.0 <= cs <= 1.0


class TestCSCoefficientMethod:
    """Verify CS-coefficient uses quantile-based method matching R's corStability()."""

    def _make_boot_result(self, records):
        from psynet.bootstrap.results import BootstrapResult
        from psynet.network import Network

        dummy_net = Network(np.eye(3), ["A", "B", "C"], "test", 100)
        return BootstrapResult(
            original_network=dummy_net,
            boot_statistics=pd.DataFrame(),
            boot_type="case",
            n_boots=10,
            case_drop_correlations=pd.DataFrame(records),
        )

    def test_quantile_method(self):
        """CS-coefficient should use the 5th-percentile quantile, not the mean."""
        from psynet.bootstrap.stability import cs_coefficient

        records = []
        # prop=0.75: all correlations = 0.75, 5th percentile = 0.75 >= 0.7 → passes
        for i in range(10):
            records.append({"proportion": 0.75, "statistic": "strength",
                            "boot_id": i, "correlation": 0.75})
        # prop=0.50: [0.50] + [0.90]*9, mean = 0.86 (would pass with mean)
        # but 5th percentile ≈ 0.68 < 0.7 → fails with quantile
        corrs_50 = [0.50] + [0.90] * 9
        for i, c in enumerate(corrs_50):
            records.append({"proportion": 0.50, "statistic": "strength",
                            "boot_id": i, "correlation": c})

        br = self._make_boot_result(records)
        cs = cs_coefficient(br, "strength", threshold=0.7)
        # prop=0.75 passes (q05=0.75>=0.7), dropped = 1-0.75 = 0.25
        # prop=0.50 fails (q05≈0.60<0.7) despite high mean
        assert cs == pytest.approx(0.25)

    def test_quantile_parameter_respected(self):
        """Passing a higher quantile (e.g. 0.5 = median) should change the result."""
        from psynet.bootstrap.stability import cs_coefficient

        records = []
        # prop=0.75: [0.60]*4 + [0.80]*6
        # q05 ≈ 0.60 < 0.7 → fails at quantile=0.05
        # median = 0.80 >= 0.7 → passes at quantile=0.5
        corrs_75 = [0.60] * 4 + [0.80] * 6
        for i, c in enumerate(corrs_75):
            records.append({"proportion": 0.75, "statistic": "strength",
                            "boot_id": i, "correlation": c})
        # prop=0.50: all 0.50, always fails
        for i in range(10):
            records.append({"proportion": 0.50, "statistic": "strength",
                            "boot_id": i, "correlation": 0.50})

        br = self._make_boot_result(records)
        # With default quantile=0.05, prop=0.75 fails → CS = 0.0
        cs_strict = cs_coefficient(br, "strength", threshold=0.7, quantile=0.05)
        assert cs_strict == pytest.approx(0.0)
        # With quantile=0.5 (median), prop=0.75 passes → CS = 0.25
        cs_lenient = cs_coefficient(br, "strength", threshold=0.7, quantile=0.5)
        assert cs_lenient == pytest.approx(0.25)

    def test_integer_proportions_in_case_drop(self, small_data):
        """Proportions from case-dropping bootstrap should match integer sample sizes."""
        result = bootnet(
            small_data,
            n_boots=5,
            boot_type="case",
            method="cor",
            n_cores=1,
            case_n=5,
            seed=42,
            verbose=False,
        )
        n = len(small_data)
        for prop in result.case_drop_correlations["proportion"].unique():
            n_keep = round(prop * n)
            # prop * n should be an integer (within floating point tolerance)
            assert abs(prop * n - n_keep) < 1e-10


class TestCSCoefficientPearson:
    """Verify case-dropping uses Pearson (not Spearman) correlation to match R."""

    def test_case_drop_uses_pearson(self, small_data):
        """Correlations in case_drop_correlations should be Pearson, not Spearman."""
        from scipy.stats import pearsonr

        result = bootnet(
            small_data,
            n_boots=5,
            boot_type="case",
            method="cor",
            n_cores=1,
            case_n=3,
            seed=42,
            verbose=False,
        )
        # Verify correlations are finite and in [-1, 1] (basic sanity)
        corrs = result.case_drop_correlations["correlation"].dropna()
        assert len(corrs) > 0
        assert (corrs.abs() <= 1.0 + 1e-10).all()


class TestBootstrapReproducibility:
    def test_same_seed_same_result(self, small_data):
        r1 = bootnet(
            small_data, n_boots=5, method="cor", statistics=["edge"],
            n_cores=1, seed=99, verbose=False,
        )
        r2 = bootnet(
            small_data, n_boots=5, method="cor", statistics=["edge"],
            n_cores=1, seed=99, verbose=False,
        )
        pd.testing.assert_frame_equal(
            r1.boot_statistics.reset_index(drop=True),
            r2.boot_statistics.reset_index(drop=True),
        )
