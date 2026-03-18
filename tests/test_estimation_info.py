"""Tests for EstimationInfo integration (R1–R4 recommendations)."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from psynet import (
    EstimationInfo,
    Network,
    bootnet,
    estimate_network,
    make_bfi25,
)


@pytest.fixture(scope="module")
def bfi_data():
    return make_bfi25(n=500, seed=42).iloc[:, :8]


# ── R1: cor_matrix is populated ──────────────────────────────────────

class TestCorMatrix:
    def test_ebicglasso_has_cor_matrix(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso")
        info = net.estimation_info
        assert info is not None
        assert info.cor_matrix is not None
        assert info.cor_matrix.shape == (bfi_data.shape[1], bfi_data.shape[1])

    def test_cor_has_cor_matrix(self, bfi_data):
        net = estimate_network(bfi_data, method="cor")
        assert net.estimation_info is not None
        assert net.estimation_info.cor_matrix is not None
        assert net.estimation_info.cor_matrix.shape[0] == bfi_data.shape[1]

    def test_pcor_has_cor_matrix(self, bfi_data):
        net = estimate_network(bfi_data, method="pcor")
        assert net.estimation_info is not None
        assert net.estimation_info.cor_matrix is not None


# ── R2: lambda / EBIC diagnostics ───────────────────────────────────

class TestLambdaDiagnostics:
    def test_selected_lambda_positive(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso")
        info = net.estimation_info
        assert info.selected_lambda is not None
        assert info.selected_lambda > 0

    def test_selected_ebic_finite(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso")
        assert np.isfinite(net.estimation_info.selected_ebic)

    def test_lambda_ebic_curve(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso")
        curve = net.estimation_info.lambda_ebic_curve
        assert curve is not None
        assert list(curve.columns) == ["lambda", "ebic"]
        assert len(curve) > 0

    def test_cor_has_no_lambda(self, bfi_data):
        net = estimate_network(bfi_data, method="cor")
        assert net.estimation_info.selected_lambda is None
        assert net.estimation_info.lambda_ebic_curve is None


# ── R3: n_cores default is -1 ───────────────────────────────────────

class TestNCoresDefault:
    def test_bootnet_default_n_cores(self):
        sig = inspect.signature(bootnet)
        assert sig.parameters["n_cores"].default == -1


# ── R4: bootnet inherits from network ───────────────────────────────

class TestBootnetNetworkKwarg:
    def test_inherits_method_and_kwargs(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso", cor_method="spearman")
        result = bootnet(
            bfi_data, network=net, n_boots=25, n_cores=1, verbose=False,
        )
        assert result.original_network.method == "EBICglasso"

    def test_warns_on_conflict(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso", cor_method="spearman")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bootnet(
                bfi_data, network=net, n_boots=25, n_cores=1,
                verbose=False, cor_method="pearson",
            )
            conflict_warnings = [
                x for x in w
                if "overrides network's estimation_info" in str(x.message)
            ]
            assert len(conflict_warnings) >= 1

    def test_default_method_without_network(self, bfi_data):
        result = bootnet(bfi_data, n_boots=25, n_cores=1, verbose=False)
        assert result.original_network.method == "EBICglasso"


# ── Backward compatibility ───────────────────────────────────────────

class TestBackwardCompat:
    def test_network_without_estimation_info(self):
        adj = np.eye(3)
        net = Network(
            adjacency=adj,
            labels=["a", "b", "c"],
            method="test",
            n_observations=10,
        )
        assert net.estimation_info is None

    def test_est_kwargs_captured(self, bfi_data):
        net = estimate_network(bfi_data, method="EBICglasso", gamma=0.25)
        assert net.estimation_info.est_kwargs["gamma"] == 0.25
