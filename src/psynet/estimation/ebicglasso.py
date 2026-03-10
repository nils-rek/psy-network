"""EBICglasso estimator — EBIC-tuned graphical lasso."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

from .._types import CorMethod
from ..network import Network
from ._registry import register


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


@register("EBICglasso")
class EBICglassoEstimator:
    """Graphical lasso with EBIC model selection over a lambda grid."""

    name: str = "EBICglasso"

    def estimate(
        self,
        data: pd.DataFrame,
        *,
        gamma: float = 0.5,
        n_lambda: int = 100,
        lambda_min_ratio: float = 0.01,
        cor_method: str | CorMethod = CorMethod.PEARSON,
        threshold: float = 1e-4,
        **kwargs,
    ) -> Network:
        cor_method = CorMethod(cor_method)
        n, p = data.shape
        cormat = data.corr(method=cor_method.value).values.copy()

        # Lambda grid (log-spaced)
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
                score = _ebic(precision, cormat, n, gamma)
                if score < best_ebic:
                    best_ebic = score
                    best_precision = precision
            except Exception:
                continue

        if best_precision is None:
            raise RuntimeError("GraphicalLasso failed to converge at all lambda values")

        # Convert precision to partial correlations
        diag = np.sqrt(np.diag(best_precision))
        pcor = -best_precision / np.outer(diag, diag)
        np.fill_diagonal(pcor, 0.0)

        # Threshold small values
        pcor[np.abs(pcor) < threshold] = 0.0

        return Network(
            adjacency=pcor,
            labels=list(data.columns),
            method=self.name,
            n_observations=n,
            weighted=True,
            signed=True,
            directed=False,
        )
