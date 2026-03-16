"""Tests for the group (JGL) subpackage."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from psynet.group._jgl import joint_graphical_lasso
from psynet.group._selection import select_lambdas, _group_ebic
from psynet.group.network import GroupNetwork
from psynet.group.bootstrap import GroupBootstrapResult, bootnet_group
from psynet.group import estimate_group_network
from psynet.datasets import make_multigroup
from psynet.plotting.group_plot import (
    plot_group_networks,
    plot_group_edge_accuracy,
    plot_group_centrality_comparison,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_simple_covariances(p=5, seed=42):
    """Create two simple covariance matrices for testing."""
    rng = np.random.default_rng(seed)
    S_list = []
    for _ in range(2):
        A = rng.standard_normal((50, p))
        cov = (A.T @ A) / 50
        # Normalize to correlation
        d = np.sqrt(np.diag(cov))
        cov = cov / np.outer(d, d)
        S_list.append(cov)
    return S_list


# ------------------------------------------------------------------ #
# TestJGL — ADMM solver
# ------------------------------------------------------------------ #

class TestJGL:
    def test_identity_covariance_low_lambda(self):
        """With identity covariance and low lambda, precision ≈ identity."""
        p = 5
        S = [np.eye(p), np.eye(p)]
        result = joint_graphical_lasso(S, [100, 100], 0.01, 0.01, "fused")
        assert len(result) == 2
        for P in result:
            assert P.shape == (p, p)
            # Should be close to identity
            np.testing.assert_allclose(P, np.eye(p), atol=0.15)

    def test_symmetry(self):
        """Output precision matrices should be symmetric."""
        S = _make_simple_covariances()
        result = joint_graphical_lasso(S, [50, 50], 0.1, 0.05, "group")
        for P in result:
            np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_sparsity_high_lambda(self):
        """High lambda1 should produce sparser off-diagonal."""
        S = _make_simple_covariances()
        result_low = joint_graphical_lasso(S, [50, 50], 0.01, 0.0, "fused")
        result_high = joint_graphical_lasso(S, [50, 50], 0.5, 0.0, "fused")
        for P_low, P_high in zip(result_low, result_high):
            nnz_low = np.count_nonzero(np.abs(np.triu(P_low, k=1)) > 1e-4)
            nnz_high = np.count_nonzero(np.abs(np.triu(P_high, k=1)) > 1e-4)
            assert nnz_high <= nnz_low

    def test_group_penalty(self):
        """Group penalty should run without error."""
        S = _make_simple_covariances()
        result = joint_graphical_lasso(S, [50, 50], 0.1, 0.1, "group")
        assert len(result) == 2

    def test_fused_penalty(self):
        """Fused penalty should run without error."""
        S = _make_simple_covariances()
        result = joint_graphical_lasso(S, [50, 50], 0.1, 0.1, "fused")
        assert len(result) == 2

    def test_three_groups(self):
        """JGL should work with K > 2 groups."""
        p = 4
        S = [np.eye(p) for _ in range(3)]
        result = joint_graphical_lasso(S, [50, 50, 50], 0.05, 0.05, "fused")
        assert len(result) == 3

    def test_positive_definite(self):
        """Output precision matrices should be positive definite."""
        S = _make_simple_covariances()
        result = joint_graphical_lasso(S, [50, 50], 0.1, 0.05, "group")
        for P in result:
            eigvals = np.linalg.eigvalsh(P)
            assert eigvals.min() > -1e-8


# ------------------------------------------------------------------ #
# TestLambdaSelection
# ------------------------------------------------------------------ #

class TestLambdaSelection:
    def test_sequential_returns_valid(self):
        """Sequential search should return valid lambdas and precisions."""
        S = _make_simple_covariances()
        l1, l2, prec = select_lambdas(
            S, [50, 50], "fused", n_lambda1=5, n_lambda2=5,
        )
        assert l1 > 0
        assert l2 >= 0
        assert len(prec) == 2

    def test_simultaneous_returns_valid(self):
        """Simultaneous search should return valid results."""
        S = _make_simple_covariances()
        l1, l2, prec = select_lambdas(
            S, [50, 50], "group", search="simultaneous",
            n_lambda1=3, n_lambda2=3,
        )
        assert l1 > 0
        assert l2 >= 0
        assert len(prec) == 2

    def test_group_ebic_computation(self):
        """EBIC computation should return finite values for valid input."""
        p = 5
        P = np.eye(p) * 1.5
        S = np.eye(p)
        score = _group_ebic([P], [S], [100], 0.5, "ebic")
        assert np.isfinite(score)

    def test_bic_criterion(self):
        """BIC criterion should work."""
        S = _make_simple_covariances()
        l1, l2, prec = select_lambdas(
            S, [50, 50], "fused", criterion="bic",
            n_lambda1=3, n_lambda2=3,
        )
        assert l1 > 0


# ------------------------------------------------------------------ #
# TestEstimateGroupNetwork
# ------------------------------------------------------------------ #

class TestEstimateGroupNetwork:
    def test_from_dataframe_with_group_col(self, two_group_data):
        """Estimate from single DataFrame with group column."""
        gn = estimate_group_network(
            two_group_data, group_col="group",
            penalty="fused", n_lambda1=5, n_lambda2=5,
        )
        assert isinstance(gn, GroupNetwork)
        assert gn.n_groups == 2
        assert set(gn.group_labels) == {"Group1", "Group2"}
        # Each network has correct structure
        for label in gn.group_labels:
            net = gn[label]
            assert net.n_nodes == 5
            assert net.adjacency.shape == (5, 5)

    def test_from_list_of_dataframes(self, two_group_data):
        """Estimate from list of DataFrames."""
        g1 = two_group_data[two_group_data["group"] == "Group1"].drop(columns="group")
        g2 = two_group_data[two_group_data["group"] == "Group2"].drop(columns="group")
        gn = estimate_group_network(
            [g1, g2], penalty="group", n_lambda1=5, n_lambda2=5,
        )
        assert isinstance(gn, GroupNetwork)
        assert gn.n_groups == 2

    def test_manual_lambdas(self, two_group_data):
        """Manual lambda1/lambda2 should skip selection."""
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        assert gn.lambda1 == 0.1
        assert gn.lambda2 == 0.05

    def test_both_penalty_types(self, two_group_data):
        """Both fused and group penalties should work."""
        for pen in ["fused", "group"]:
            gn = estimate_group_network(
                two_group_data, group_col="group",
                penalty=pen, lambda1=0.1, lambda2=0.05,
            )
            assert gn.penalty == pen

    def test_missing_group_col_raises(self, two_group_data):
        """Should raise if group_col not provided for DataFrame input."""
        with pytest.raises(ValueError, match="group_col"):
            estimate_group_network(two_group_data)

    def test_mismatched_columns_raises(self):
        """Should raise if groups have different variable names."""
        rng = np.random.default_rng(42)
        df1 = pd.DataFrame(rng.standard_normal((50, 3)), columns=["A", "B", "C"])
        df2 = pd.DataFrame(rng.standard_normal((50, 3)), columns=["X", "Y", "Z"])
        with pytest.raises(ValueError, match="same variable names"):
            estimate_group_network([df1, df2], lambda1=0.1, lambda2=0.05)


# ------------------------------------------------------------------ #
# TestGroupNetwork
# ------------------------------------------------------------------ #

class TestGroupNetwork:
    def test_indexing(self, two_group_data):
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        net = gn["Group1"]
        assert net.n_nodes == 5
        with pytest.raises(KeyError):
            gn["NonExistent"]

    def test_centrality(self, two_group_data):
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        cent = gn.centrality()
        assert "group" in cent.columns
        assert "node" in cent.columns
        assert "strength" in cent.columns
        assert set(cent["group"].unique()) == {"Group1", "Group2"}

    def test_compare_edges(self, two_group_data):
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        edges = gn.compare_edges()
        assert "group" in edges.columns
        assert "node1" in edges.columns
        assert "weight" in edges.columns


# ------------------------------------------------------------------ #
# TestGroupBootstrap
# ------------------------------------------------------------------ #

class TestGroupBootstrap:
    def test_small_run(self, two_group_data):
        """Small bootstrap run with n_boots=10."""
        result = bootnet_group(
            two_group_data, group_col="group",
            n_boots=10, statistics=["edge"],
            lambda1=0.1, lambda2=0.05,
            verbose=False,
        )
        assert isinstance(result, GroupBootstrapResult)
        assert result.n_boots == 10
        assert "group" in result.boot_statistics.columns
        assert "boot_id" in result.boot_statistics.columns
        groups_in_result = result.boot_statistics["group"].unique()
        assert "Group1" in groups_in_result
        assert "Group2" in groups_in_result

    def test_summary(self, two_group_data):
        """Summary should return per-group statistics."""
        result = bootnet_group(
            two_group_data, group_col="group",
            n_boots=5, statistics=["edge"],
            lambda1=0.1, lambda2=0.05,
            verbose=False,
        )
        summary = result.summary(statistic="edge")
        assert "group" in summary.columns
        assert "mean" in summary.columns
        assert "ci_lower" in summary.columns

    def test_summary_single_group(self, two_group_data):
        """Summary filtered to a single group."""
        result = bootnet_group(
            two_group_data, group_col="group",
            n_boots=5, statistics=["edge"],
            lambda1=0.1, lambda2=0.05,
            verbose=False,
        )
        summary = result.summary(statistic="edge", group="Group1")
        assert all(summary["group"] == "Group1")

    def test_custom_group_labels(self, two_group_data):
        """Bootstrap should preserve original group labels, not auto-generate."""
        # Rename groups to non-default labels
        data = two_group_data.copy()
        data["group"] = data["group"].replace({"Group1": "Control", "Group2": "Treatment"})
        result = bootnet_group(
            data, group_col="group",
            n_boots=5, statistics=["edge"],
            lambda1=0.1, lambda2=0.05,
            verbose=False,
        )
        groups_in_result = set(result.boot_statistics["group"].unique())
        assert groups_in_result == {"Control", "Treatment"}


# ------------------------------------------------------------------ #
# TestGroupPlotting — smoke tests
# ------------------------------------------------------------------ #

class TestGroupPlotting:
    def test_plot_group_networks(self, two_group_data):
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        fig = plot_group_networks(gn)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_group_networks_shared_layout(self, two_group_data):
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        fig = plot_group_networks(gn, shared_layout=True, layout="circular")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_group_centrality_comparison(self, two_group_data):
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        fig = plot_group_centrality_comparison(gn, statistic="strength")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_via_group_network(self, two_group_data):
        """GroupNetwork.plot() should delegate properly."""
        gn = estimate_group_network(
            two_group_data, group_col="group",
            lambda1=0.1, lambda2=0.05,
        )
        fig = gn.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_edge_accuracy(self, two_group_data):
        result = bootnet_group(
            two_group_data, group_col="group",
            n_boots=5, statistics=["edge"],
            lambda1=0.1, lambda2=0.05,
            verbose=False,
        )
        fig = plot_group_edge_accuracy(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_edge_accuracy_via_result(self, two_group_data):
        """GroupBootstrapResult.plot_edge_accuracy() should delegate."""
        result = bootnet_group(
            two_group_data, group_col="group",
            n_boots=5, statistics=["edge"],
            lambda1=0.1, lambda2=0.05,
            verbose=False,
        )
        fig = result.plot_edge_accuracy()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ------------------------------------------------------------------ #
# TestMakeMultigroup
# ------------------------------------------------------------------ #

class TestMakeMultigroup:
    def test_basic(self):
        df = make_multigroup(n_per_group=50, n_groups=2, p=5)
        assert "group" in df.columns
        assert len(df) == 100
        assert df.drop(columns="group").shape[1] == 5
        assert set(df["group"].unique()) == {"Group1", "Group2"}

    def test_three_groups(self):
        df = make_multigroup(n_per_group=30, n_groups=3, p=4)
        assert len(df) == 90
        assert len(df["group"].unique()) == 3

    def test_integration_with_estimate(self):
        """make_multigroup output should work with estimate_group_network."""
        df = make_multigroup(n_per_group=100, n_groups=2, p=5, seed=123)
        gn = estimate_group_network(
            df, group_col="group", lambda1=0.1, lambda2=0.05,
        )
        assert isinstance(gn, GroupNetwork)
        assert gn.n_groups == 2
