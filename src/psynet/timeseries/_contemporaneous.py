"""Contemporaneous network estimation via EBIC graphical lasso on VAR residuals."""

from __future__ import annotations

import numpy as np

from .._glasso_utils import _fit_ebic_glasso
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
        method="graphicalVAR",
        n_observations=T,
        weighted=True,
        signed=True,
        directed=False,
    )
