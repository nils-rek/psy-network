"""Shared utilities for graphical lasso estimation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso


def _ebic(
    precision: np.ndarray,
    cov: np.ndarray,
    n: int,
    gamma: float,
    threshold: float = 0.0,
) -> float:
    """Compute Extended BIC for a given precision matrix.

    EBIC = -2 * loglik + E * log(n) + 4 * E * gamma * log(p)

    where loglik = (n/2) * (log|P| - trace(S @ P)) and E is the number
    of off-diagonal elements with ``|value| > threshold`` (counted once
    per pair).  The threshold matters because solvers leave numerically
    tiny non-zeros that would otherwise inflate the edge count and bias
    lambda selection relative to the thresholded network that is
    ultimately returned.
    """
    p = precision.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return np.inf
    loglik = (n / 2) * (logdet - np.trace(cov @ precision))
    # Count unique off-diagonal edges above threshold
    upper = np.triu(precision, k=1)
    n_edges = np.count_nonzero(np.abs(upper) > threshold)
    ebic_val = -2 * loglik + n_edges * np.log(n) + 4 * n_edges * gamma * np.log(p)
    return ebic_val


def precision_to_pcor(
    precision: np.ndarray, *, threshold: float = 0.0,
) -> np.ndarray:
    """Convert a precision matrix to a partial correlation matrix.

    pcor_ij = -P_ij / sqrt(P_ii * P_jj), zero diagonal, entries with
    ``|value| < threshold`` zeroed when a threshold is given.
    """
    diag = np.sqrt(np.abs(np.diag(precision)))
    diag[diag < 1e-10] = 1.0
    pcor = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcor, 0.0)
    if threshold > 0:
        pcor[np.abs(pcor) < threshold] = 0.0
    return pcor


def _fit_ebic_glasso(
    cormat: np.ndarray,
    n: int,
    *,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
    track_curve: bool = False,
) -> tuple[np.ndarray, float, float, pd.DataFrame | None]:
    """Run EBIC-tuned graphical lasso on a correlation matrix.

    Parameters
    ----------
    cormat : ndarray, shape (p, p)
        Correlation matrix.
    n : int
        Sample size (used in EBIC calculation).
    gamma : float
        EBIC gamma parameter (sparsity tuning).
    n_lambda : int
        Number of lambda values in the search grid.
    lambda_min_ratio : float
        Ratio of minimum to maximum lambda.
    threshold : float
        Threshold for zeroing small partial correlations.
    track_curve : bool
        If True, return a DataFrame of (lambda, ebic) for each candidate.

    Returns
    -------
    pcor : ndarray, shape (p, p)
        Partial correlation matrix (thresholded).
    best_lambda : float
        Selected lambda value.
    best_ebic : float
        EBIC at the selected lambda.
    curve_df : pd.DataFrame or None
        Lambda–EBIC curve if ``track_curve`` is True, else None.
    """
    # Lambda grid (log-spaced)
    lambda_max = np.max(np.abs(np.triu(cormat, k=1)))
    lambda_min = lambda_max * lambda_min_ratio
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)

    best_ebic_val = np.inf
    best_precision = None
    best_lambda: float | None = None
    curve_records: list[dict[str, float]] | None = [] if track_curve else None

    for alpha in lambdas:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gl = GraphicalLasso(
                    alpha=alpha,
                    covariance="precomputed",
                    max_iter=500,
                    tol=1e-4,
                )
                gl.fit(cormat)
            precision = gl.precision_
            score = _ebic(precision, cormat, n, gamma, threshold=threshold)
            if curve_records is not None:
                curve_records.append({"lambda": alpha, "ebic": score})
            if score < best_ebic_val:
                best_ebic_val = score
                best_precision = precision
                best_lambda = alpha
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            # Convergence/numerical failure at this lambda — skip it
            continue

    if best_precision is None:
        raise RuntimeError("GraphicalLasso failed to converge at all lambda values")

    # Convert precision to partial correlations
    pcor = precision_to_pcor(best_precision, threshold=threshold)

    curve_df = pd.DataFrame(curve_records) if curve_records is not None else None
    return pcor, best_lambda, best_ebic_val, curve_df


def contemporaneous_from_residuals(
    residuals: np.ndarray,
    labels: list[str],
    method: str,
    *,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
):
    """Estimate an undirected GGM from (pooled) VAR residuals.

    Shared by the timeseries and multilevel contemporaneous estimators:
    correlation matrix of the residuals -> EBIC-tuned graphical lasso ->
    partial correlation Network.
    """
    from .network import Network

    T = residuals.shape[0]
    cormat = np.corrcoef(residuals, rowvar=False)

    pcor, _, _, _ = _fit_ebic_glasso(
        cormat, T,
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
    )

    return Network(
        adjacency=pcor,
        labels=labels,
        method=method,
        n_observations=T,
        weighted=True,
        signed=True,
        directed=False,
    )
