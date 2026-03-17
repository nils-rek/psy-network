"""Temporal network estimation via column-wise Lasso VAR(1)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV

from ..network import Network


def estimate_temporal(
    X: np.ndarray,
    Y: np.ndarray,
    labels: list[str],
    *,
    cv: int = 10,
    n_alphas: int = 100,
    max_iter: int = 10000,
    threshold: float = 1e-4,
) -> tuple[Network, np.ndarray]:
    """Estimate temporal (directed) network via sparse VAR(1).

    Fits a separate LassoCV for each variable, regressing Y[:,j] on X.
    The coefficient matrix B has B[j, k] = effect of variable k at t-1
    on variable j at t.

    Parameters
    ----------
    X : ndarray, shape (T, p)
        Lagged predictors.
    Y : ndarray, shape (T, p)
        Outcome matrix.
    labels : list[str]
        Variable names.
    cv : int
        Number of cross-validation folds for LassoCV.
    n_alphas : int
        Number of alphas for LassoCV.
    max_iter : int
        Maximum iterations for Lasso.
    threshold : float
        Coefficients with absolute value below this are zeroed.

    Returns
    -------
    temporal_net : Network
        Directed network with B as adjacency matrix.
    residuals : ndarray, shape (T, p)
        VAR residuals (Y - X @ B.T).
    """
    T, p = Y.shape
    B = np.zeros((p, p))

    for j in range(p):
        lasso = LassoCV(
            cv=min(cv, T),
            alphas=n_alphas,
            max_iter=max_iter,
        )
        lasso.fit(X, Y[:, j])
        B[j, :] = lasso.coef_

    # Threshold small coefficients
    B[np.abs(B) < threshold] = 0.0

    residuals = Y - X @ B.T

    temporal_net = Network(
        adjacency=B,
        labels=labels,
        method="graphicalVAR",
        n_observations=T,
        weighted=True,
        signed=True,
        directed=True,
    )

    return temporal_net, residuals
