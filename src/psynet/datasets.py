"""Built-in example datasets for psynet."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_bfi25(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic BFI-25 personality data (5 factors × 5 items).

    This creates plausible questionnaire data with a known 5-factor
    structure (Openness, Conscientiousness, Extraversion, Agreeableness,
    Neuroticism), useful for demonstrations and testing.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with 25 columns (O1-O5, C1-C5, E1-E5, A1-A5, N1-N5).
    """
    rng = np.random.default_rng(seed)

    factors = ["O", "C", "E", "A", "N"]
    items_per_factor = 5
    n_items = len(factors) * items_per_factor

    # Factor loadings: each item loads ~0.6-0.8 on its factor
    loadings = np.zeros((n_items, len(factors)))
    for f_idx, factor in enumerate(factors):
        for i in range(items_per_factor):
            item_idx = f_idx * items_per_factor + i
            loadings[item_idx, f_idx] = rng.uniform(0.55, 0.80)

    # Small cross-loadings
    for i in range(n_items):
        for j in range(len(factors)):
            if loadings[i, j] == 0:
                loadings[i, j] = rng.uniform(-0.05, 0.10)

    # Generate factor scores
    factor_corr = np.eye(len(factors))
    # Small inter-factor correlations
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            r = rng.uniform(-0.15, 0.25)
            factor_corr[i, j] = r
            factor_corr[j, i] = r

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(factor_corr)
    if eigvals.min() < 0.01:
        factor_corr += (0.02 - eigvals.min()) * np.eye(len(factors))

    L = np.linalg.cholesky(factor_corr)
    factor_scores = rng.standard_normal((n, len(factors))) @ L.T

    # Item scores = loadings @ factors + noise
    noise = rng.standard_normal((n, n_items)) * 0.4
    raw = factor_scores @ loadings.T + noise

    # Scale to 1-6 Likert range
    from scipy.stats import norm
    percentiles = norm.cdf(raw)
    likert = np.ceil(percentiles * 6).astype(int)
    likert = np.clip(likert, 1, 6)

    columns = []
    for factor in factors:
        for i in range(1, items_per_factor + 1):
            columns.append(f"{factor}{i}")

    return pd.DataFrame(likert, columns=columns)


def make_depression9(n: int = 300, seed: int = 123) -> pd.DataFrame:
    """Generate synthetic PHQ-9-like depression symptom data.

    Nine items reflecting depressive symptoms with realistic covariance.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with 9 columns (dep1-dep9).
    """
    rng = np.random.default_rng(seed)

    # Correlation structure with symptom clusters
    p = 9
    cormat = np.eye(p)
    # Somatic cluster (items 0-2)
    for i in range(3):
        for j in range(i + 1, 3):
            r = rng.uniform(0.35, 0.55)
            cormat[i, j] = cormat[j, i] = r
    # Cognitive cluster (items 3-5)
    for i in range(3, 6):
        for j in range(i + 1, 6):
            r = rng.uniform(0.40, 0.60)
            cormat[i, j] = cormat[j, i] = r
    # Behavioral cluster (items 6-8)
    for i in range(6, 9):
        for j in range(i + 1, 9):
            r = rng.uniform(0.30, 0.50)
            cormat[i, j] = cormat[j, i] = r
    # Cross-cluster
    for i in range(3):
        for j in range(3, 9):
            r = rng.uniform(0.10, 0.30)
            cormat[i, j] = cormat[j, i] = r
    for i in range(3, 6):
        for j in range(6, 9):
            r = rng.uniform(0.15, 0.35)
            cormat[i, j] = cormat[j, i] = r

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(cormat)
    if eigvals.min() < 0.01:
        cormat += (0.02 - eigvals.min()) * np.eye(p)
    # Re-normalize to correlation
    d = np.sqrt(np.diag(cormat))
    cormat = cormat / np.outer(d, d)

    L = np.linalg.cholesky(cormat)
    raw = rng.standard_normal((n, p)) @ L.T

    from scipy.stats import norm
    percentiles = norm.cdf(raw)
    likert = np.ceil(percentiles * 4).astype(int)
    likert = np.clip(likert, 0, 3)

    columns = [f"dep{i}" for i in range(1, 10)]
    return pd.DataFrame(likert, columns=columns)
