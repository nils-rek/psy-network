"""Tests for bootstrap analysis."""

import numpy as np
import pandas as pd
import pytest

from psynet import estimate_network
from psynet.bootstrap import bootnet, BootstrapResult


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
    """Verify CS-coefficient uses mean-correlation method matching R's corStability()."""

    def test_mean_correlation_method(self):
        """CS-coefficient should use mean correlation, not proportion of samples."""
        from psynet.bootstrap.results import BootstrapResult
        from psynet.bootstrap.stability import cs_coefficient
        from psynet.network import Network

        records = []
        # prop=0.75: mean correlation = (0.65 + 0.85*9)/10 = 0.83 >= 0.7 → passes
        corrs_75 = [0.65] + [0.85] * 9
        for i, c in enumerate(corrs_75):
            records.append({"proportion": 0.75, "statistic": "strength",
                            "boot_id": i, "correlation": c})
        # prop=0.50: mean correlation = 0.60 < 0.7 → fails
        corrs_50 = [0.60] * 10
        for i, c in enumerate(corrs_50):
            records.append({"proportion": 0.50, "statistic": "strength",
                            "boot_id": i, "correlation": c})

        dummy_net = Network(np.eye(3), ["A", "B", "C"], "test", 100)
        br = BootstrapResult(
            original_network=dummy_net,
            boot_statistics=pd.DataFrame(),
            boot_type="case",
            n_boots=10,
            case_drop_correlations=pd.DataFrame(records),
        )

        cs = cs_coefficient(br, "strength", threshold=0.7)
        # prop=0.75 passes (mean=0.83>=0.7), dropped = 1-0.75 = 0.25
        # prop=0.50 fails (mean=0.60<0.7)
        assert cs == pytest.approx(0.25)

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
