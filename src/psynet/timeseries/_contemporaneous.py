"""Contemporaneous network estimation via EBIC graphical lasso on VAR residuals."""

from __future__ import annotations

import numpy as np

from .._glasso_utils import contemporaneous_from_residuals
from ..network import Network


def estimate_contemporaneous(
    residuals: np.ndarray,
    labels: list[str],
    *,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
    n_jobs: int = 1,
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
    return contemporaneous_from_residuals(
        residuals, labels, "graphicalVAR",
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
    )
