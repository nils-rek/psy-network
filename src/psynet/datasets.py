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


def make_multigroup(
    n_per_group: int = 200,
    n_groups: int = 2,
    p: int = 9,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-group data with shared base structure.

    Groups share ~70% of edges with the same correlation structure, and
    ~30% of edges differ across groups. Useful for testing Joint Graphical
    Lasso estimation.

    Parameters
    ----------
    n_per_group : int
        Number of observations per group.
    n_groups : int
        Number of groups.
    p : int
        Number of variables.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with V1..Vp columns + ``'group'`` column.
    """
    rng = np.random.default_rng(seed)

    # Build a shared base precision matrix (sparse)
    base_prec = np.eye(p)
    # Add some off-diagonal structure (banded + a few random entries)
    for i in range(p - 1):
        val = rng.uniform(0.2, 0.4)
        base_prec[i, i + 1] = val
        base_prec[i + 1, i] = val
    if p > 3:
        base_prec[0, p - 1] = 0.25
        base_prec[p - 1, 0] = 0.25

    frames = []
    for g in range(n_groups):
        # Create group-specific precision by modifying ~30% of off-diagonal entries
        prec_g = base_prec.copy()
        n_edges = p * (p - 1) // 2
        n_diff = max(1, int(0.3 * n_edges))

        # Pick random off-diagonal positions to modify
        upper_idx = np.array([(i, j) for i in range(p) for j in range(i + 1, p)])
        diff_idx = rng.choice(len(upper_idx), size=n_diff, replace=False)
        for idx in diff_idx:
            i, j = upper_idx[idx]
            # Either add, remove, or change an edge
            if prec_g[i, j] != 0:
                new_val = rng.choice([0.0, rng.uniform(0.1, 0.5)])
            else:
                new_val = rng.choice([0.0, rng.uniform(0.15, 0.4)])
            prec_g[i, j] = new_val
            prec_g[j, i] = new_val

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(prec_g)
        if eigvals.min() < 0.1:
            prec_g += (0.15 - eigvals.min()) * np.eye(p)

        # Generate data from this precision matrix
        cov_g = np.linalg.inv(prec_g)
        # Normalize to correlation
        d = np.sqrt(np.diag(cov_g))
        cov_g = cov_g / np.outer(d, d)

        L = np.linalg.cholesky(cov_g)
        data = rng.standard_normal((n_per_group, p)) @ L.T

        df = pd.DataFrame(data, columns=[f"V{i+1}" for i in range(p)])
        df["group"] = f"Group{g+1}"
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _make_sparse_var_matrix(
    rng: np.random.Generator,
    p: int,
    sparsity: float,
    spectral_radius: float,
) -> np.ndarray:
    """Create a sparse VAR(1) transition matrix scaled to a target spectral radius.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    p : int
        Number of variables.
    sparsity : float
        Proportion of off-diagonal entries to zero out.
    spectral_radius : float
        Target spectral radius (must be < 1 for stationarity).

    Returns
    -------
    ndarray, shape (p, p)
        Sparse transition matrix.
    """
    B = rng.uniform(-0.5, 0.5, size=(p, p))
    mask = rng.random((p, p)) < sparsity
    np.fill_diagonal(mask, False)
    B[mask] = 0.0
    np.fill_diagonal(B, rng.uniform(0.2, 0.5, size=p))

    eigvals = np.linalg.eigvals(B)
    current_radius = np.max(np.abs(eigvals))
    if current_radius > 0:
        B = B * (spectral_radius / current_radius)

    return B


def make_var_data(
    n_timepoints: int = 500,
    p: int = 6,
    *,
    spectral_radius: float = 0.8,
    sparsity: float = 0.5,
    seed: int = 42,
    burn_in: int = 200,
) -> pd.DataFrame:
    """Generate synthetic VAR(1) time-series data.

    Creates data from a known VAR(1) process with a sparse transition
    matrix B and sparse innovation covariance, useful for testing
    time-series network estimation.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints to return (after burn-in).
    p : int
        Number of variables.
    spectral_radius : float
        Target spectral radius for B (must be < 1 for stationarity).
    sparsity : float
        Proportion of off-diagonal entries in B to zero out.
    seed : int
        Random seed.
    burn_in : int
        Number of burn-in samples to discard.

    Returns
    -------
    pd.DataFrame
        DataFrame with V1..Vp columns, ``n_timepoints`` rows.
    """
    rng = np.random.default_rng(seed)

    B = _make_sparse_var_matrix(rng, p, sparsity, spectral_radius)

    # Create sparse innovation precision → covariance
    prec = np.eye(p)
    for i in range(p - 1):
        val = rng.uniform(0.2, 0.4)
        prec[i, i + 1] = val
        prec[i + 1, i] = val
    # Ensure positive definite
    eigvals_p = np.linalg.eigvalsh(prec)
    if eigvals_p.min() < 0.1:
        prec += (0.15 - eigvals_p.min()) * np.eye(p)
    sigma = np.linalg.inv(prec)
    # Normalize to correlation-like scale
    d = np.sqrt(np.diag(sigma))
    sigma = sigma / np.outer(d, d)

    L = np.linalg.cholesky(sigma)

    # Simulate VAR(1) process
    total = n_timepoints + burn_in
    Y = np.zeros((total, p))
    Y[0] = rng.standard_normal(p)
    for t in range(1, total):
        eps = rng.standard_normal(p) @ L.T
        Y[t] = B @ Y[t - 1] + eps

    # Discard burn-in
    Y = Y[burn_in:]

    columns = [f"V{i+1}" for i in range(p)]
    return pd.DataFrame(Y, columns=columns)


def make_multilevel_data(
    n_subjects: int = 20,
    n_timepoints: int = 50,
    p: int = 4,
    *,
    spectral_radius: float = 0.6,
    sparsity: float = 0.3,
    random_effect_sd: float = 0.1,
    seed: int = 42,
    burn_in: int = 50,
) -> pd.DataFrame:
    """Generate synthetic multilevel VAR(1) data for multiple subjects.

    Creates data from a shared VAR(1) process with subject-specific random
    effects on the transition matrix. Useful for testing mlVAR estimation.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_timepoints : int
        Number of timepoints per subject (after burn-in).
    p : int
        Number of variables.
    spectral_radius : float
        Target spectral radius for the population B matrix (must be < 1).
    sparsity : float
        Proportion of off-diagonal entries in B to zero out.
    random_effect_sd : float
        Standard deviation of subject-specific random effects on B.
    seed : int
        Random seed.
    burn_in : int
        Number of burn-in samples to discard per subject.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: ``subject``, ``beep``,
        ``V1``..``Vp``.
    """
    rng = np.random.default_rng(seed)

    B = _make_sparse_var_matrix(rng, p, sparsity, spectral_radius)

    # Innovation covariance (shared across subjects)
    sigma = np.eye(p)
    for i in range(p - 1):
        val = rng.uniform(0.1, 0.3)
        sigma[i, i + 1] = val
        sigma[i + 1, i] = val
    eigvals_s = np.linalg.eigvalsh(sigma)
    if eigvals_s.min() < 0.1:
        sigma += (0.15 - eigvals_s.min()) * np.eye(p)
    L_sigma = np.linalg.cholesky(sigma)

    frames = []
    for s in range(n_subjects):
        # Subject-specific B with random effects
        B_s = B + rng.normal(0, random_effect_sd, size=(p, p))
        # Ensure stationarity: rescale if spectral radius >= 1
        sr = np.max(np.abs(np.linalg.eigvals(B_s)))
        if sr >= 1.0:
            B_s = B_s * (0.95 / sr)

        # Simulate VAR(1)
        total = n_timepoints + burn_in
        Y = np.zeros((total, p))
        Y[0] = rng.standard_normal(p)
        for t in range(1, total):
            eps = rng.standard_normal(p) @ L_sigma.T
            Y[t] = B_s @ Y[t - 1] + eps

        Y = Y[burn_in:]
        df = pd.DataFrame(Y, columns=[f"V{i+1}" for i in range(p)])
        df["subject"] = f"S{s+1}"
        df["beep"] = np.arange(1, n_timepoints + 1)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
