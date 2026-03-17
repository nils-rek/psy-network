"""Shared utilities for graphical lasso estimation."""

from __future__ import annotations

import numpy as np


def _ebic(precision: np.ndarray, cov: np.ndarray, n: int, gamma: float) -> float:
    """Compute Extended BIC for a given precision matrix.

    EBIC = -2 * loglik + E * log(n) + 4 * E * gamma * log(p)

    where loglik = (n/2) * (log|P| - trace(S @ P)) and E is the number
    of non-zero off-diagonal elements (counted once per pair).
    """
    p = precision.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return np.inf
    loglik = (n / 2) * (logdet - np.trace(cov @ precision))
    # Count unique non-zero off-diagonal edges
    upper = np.triu(precision, k=1)
    n_edges = np.count_nonzero(upper)
    ebic_val = -2 * loglik + n_edges * np.log(n) + 4 * n_edges * gamma * np.log(p)
    return ebic_val
