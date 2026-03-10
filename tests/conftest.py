"""Shared fixtures for psynet tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_data() -> pd.DataFrame:
    """Small synthetic dataset (50 × 5) with known correlation structure."""
    rng = np.random.default_rng(42)
    n, p = 50, 5
    # Two correlated groups: (0,1,2) and (3,4)
    factor1 = rng.standard_normal(n)
    factor2 = rng.standard_normal(n)
    data = np.column_stack([
        factor1 + rng.standard_normal(n) * 0.3,
        factor1 + rng.standard_normal(n) * 0.3,
        factor1 + rng.standard_normal(n) * 0.3,
        factor2 + rng.standard_normal(n) * 0.3,
        factor2 + rng.standard_normal(n) * 0.3,
    ])
    return pd.DataFrame(data, columns=[f"V{i+1}" for i in range(p)])


@pytest.fixture
def medium_data() -> pd.DataFrame:
    """Medium dataset (200 × 10) for estimation tests."""
    rng = np.random.default_rng(123)
    n, p = 200, 10
    cov = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            r = rng.uniform(-0.1, 0.4) if abs(i - j) <= 2 else 0.0
            cov[i, j] = cov[j, i] = r
    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0.01:
        cov += (0.02 - eigvals.min()) * np.eye(p)
    L = np.linalg.cholesky(cov)
    data = rng.standard_normal((n, p)) @ L.T
    return pd.DataFrame(data, columns=[f"X{i+1}" for i in range(p)])
