"""Temporal network estimation via column-wise Lasso VAR(1)."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV

from ..network import Network


def _fit_one_var_column(
    j: int,
    X: np.ndarray,
    Y: np.ndarray,
    cv: int,
    n_alphas: int,
    max_iter: int,
) -> tuple[int, np.ndarray]:
    """Fit a single LassoCV for variable j. Module-level for joblib pickling."""
    T = Y.shape[0]
    lasso = LassoCV(
        cv=min(cv, T),
        alphas=n_alphas,
        max_iter=max_iter,
    )
    lasso.fit(X, Y[:, j])
    return j, lasso.coef_


def estimate_temporal(
    X: np.ndarray,
    Y: np.ndarray,
    labels: list[str],
    *,
    cv: int = 10,
    n_alphas: int = 100,
    max_iter: int = 10000,
    threshold: float = 1e-4,
    n_jobs: int = 1,
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
    n_jobs : int
        Number of parallel jobs for fitting variable-wise models.
        ``1`` (default) runs serially; ``-1`` uses all cores.

    Returns
    -------
    temporal_net : Network
        Directed network with B as adjacency matrix.
    residuals : ndarray, shape (T, p)
        VAR residuals (Y - X @ B.T).
    """
    T, p = Y.shape
    B = np.zeros((p, p))

    if n_jobs == 1:
        for j in range(p):
            _, coef = _fit_one_var_column(j, X, Y, cv, n_alphas, max_iter)
            B[j, :] = coef
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_var_column)(j, X, Y, cv, n_alphas, max_iter)
            for j in range(p)
        )
        for j, coef in results:
            B[j, :] = coef

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
