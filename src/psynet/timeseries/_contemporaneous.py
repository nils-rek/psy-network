"""Contemporaneous network estimation via EBIC graphical lasso on VAR residuals."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import GraphicalLasso

from .._glasso_utils import _ebic
from ..network import Network


def estimate_contemporaneous(
    residuals: np.ndarray,
    labels: list[str],
    *,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
) -> Network:
    """Estimate contemporaneous (undirected) network from VAR residuals.

    Applies EBIC-tuned graphical lasso to the correlation matrix of
    the residuals, then converts the precision matrix to partial
    correlations.

    Parameters
    ----------
    residuals : ndarray, shape (T, p)
        VAR residuals.
    labels : list[str]
        Variable names.
    gamma : float
        EBIC gamma parameter (sparsity tuning).
    n_lambda : int
        Number of lambda values in the search grid.
    lambda_min_ratio : float
        Ratio of minimum to maximum lambda.
    threshold : float
        Threshold for zeroing small partial correlations.

    Returns
    -------
    Network
        Undirected contemporaneous network.
    """
    T, p = residuals.shape

    # Correlation matrix of residuals
    cormat = np.corrcoef(residuals, rowvar=False)

    # Lambda grid
    lambda_max = np.max(np.abs(np.triu(cormat, k=1)))
    lambda_min = lambda_max * lambda_min_ratio
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)

    best_ebic = np.inf
    best_precision = None

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
            score = _ebic(precision, cormat, T, gamma)
            if score < best_ebic:
                best_ebic = score
                best_precision = precision
        except Exception:
            continue

    if best_precision is None:
        raise RuntimeError(
            "GraphicalLasso failed to converge at all lambda values for residuals"
        )

    # Convert precision to partial correlations
    diag = np.sqrt(np.diag(best_precision))
    pcor = -best_precision / np.outer(diag, diag)
    np.fill_diagonal(pcor, 0.0)
    pcor[np.abs(pcor) < threshold] = 0.0

    return Network(
        adjacency=pcor,
        labels=labels,
        method="graphicalVAR",
        n_observations=T,
        weighted=True,
        signed=True,
        directed=False,
    )
