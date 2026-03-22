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


@pytest.fixture
def two_group_data() -> pd.DataFrame:
    """Two-group dataset (100 obs each, 5 variables, 'group' column)."""
    rng = np.random.default_rng(99)
    n, p = 100, 5

    frames = []
    for g in range(2):
        # Slightly different correlation structures per group
        cov = np.eye(p)
        for i in range(p - 1):
            val = 0.4 + g * 0.1 + rng.uniform(-0.05, 0.05)
            cov[i, i + 1] = val
            cov[i + 1, i] = val
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0.01:
            cov += (0.02 - eigvals.min()) * np.eye(p)
        L = np.linalg.cholesky(cov)
        data = rng.standard_normal((n, p)) @ L.T
        df = pd.DataFrame(data, columns=[f"V{i+1}" for i in range(p)])
        df["group"] = f"Group{g+1}"
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def var_data() -> pd.DataFrame:
    """Small VAR(1) time-series dataset (200 × 4) for time-series tests."""
    from psynet.datasets import make_var_data
    return make_var_data(n_timepoints=200, p=4, seed=42)


@pytest.fixture
def multilevel_data() -> pd.DataFrame:
    """Multilevel VAR dataset (10 subjects × 30 timepoints × 4 variables)."""
    from psynet.datasets import make_multilevel_data
    return make_multilevel_data(n_subjects=10, n_timepoints=30, p=4, seed=42)
