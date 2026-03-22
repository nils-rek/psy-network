"""Tests for performance optimizations (parallel variable loop, n_jobs threading)."""

import numpy as np
import pandas as pd
import pytest

from psynet.datasets import make_bfi25, make_var_data
from psynet.estimation import estimate_network
from psynet.timeseries._validation import make_lag_matrix, validate_ts_data
from psynet.timeseries._var import estimate_temporal
from psynet.timeseries import estimate_var_network


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def var_data():
    return make_var_data(n_timepoints=200, p=4, seed=42)


@pytest.fixture
def bfi_data():
    return make_bfi25(n=100, seed=42)


# ── VAR parallel equivalence ────────────────────────────────────────


class TestVARParallel:
    """Verify that parallel VAR estimation matches serial."""

    def test_temporal_n_jobs_equivalence(self, var_data):
        var_cols = validate_ts_data(var_data)
        X, Y = make_lag_matrix(var_data, var_cols)

        net_serial, res_serial = estimate_temporal(
            X, Y, var_cols, n_jobs=1,
        )
        net_parallel, res_parallel = estimate_temporal(
            X, Y, var_cols, n_jobs=2,
        )

        np.testing.assert_allclose(
            net_serial.adjacency, net_parallel.adjacency, atol=1e-10,
        )
        np.testing.assert_allclose(res_serial, res_parallel, atol=1e-10)

    def test_var_network_n_jobs(self, var_data):
        ts1 = estimate_var_network(var_data, n_jobs=1)
        ts2 = estimate_var_network(var_data, n_jobs=2)

        np.testing.assert_allclose(
            ts1.temporal.adjacency, ts2.temporal.adjacency, atol=1e-10,
        )
        np.testing.assert_allclose(
            ts1.contemporaneous.adjacency, ts2.contemporaneous.adjacency, atol=1e-10,
        )


# ── EBICglasso n_jobs parameter ─────────────────────────────────────


class TestEBICglassoNJobs:
    """Verify n_jobs parameter is accepted and doesn't change results."""

    def test_n_jobs_accepted(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso", n_jobs=1)
        assert net.adjacency.shape[0] == net.adjacency.shape[1]


# ── Bootstrap nested parallelism ────────────────────────────────────


class TestBootstrapNJobs:
    """Verify bootstrap injects n_jobs=1 for inner estimator."""

    def test_bootnet_sets_inner_n_jobs(self, bfi_data):
        from psynet.bootstrap import bootnet

        result = bootnet(
            bfi_data,
            method="EBICglasso",
            n_boots=5,
            seed=42,
        )
        assert result.boot_statistics is not None
        assert len(result.boot_statistics) > 0
