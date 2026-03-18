"""EBICglasso estimator — EBIC-tuned graphical lasso."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

from .._glasso_utils import _ebic
from .._types import CorMethod
from ..estimation_info import EstimationInfo
from ..network import Network
from ._registry import register


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
        best_lambda: float | None = None
        curve_records: list[dict[str, float]] = []

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
                curve_records.append({"lambda": alpha, "ebic": score})
                if score < best_ebic:
                    best_ebic = score
                    best_precision = precision
                    best_lambda = alpha
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

        info = EstimationInfo(
            method=self.name,
            est_kwargs={
                "gamma": gamma,
                "n_lambda": n_lambda,
                "lambda_min_ratio": lambda_min_ratio,
                "cor_method": cor_method.value,
                "threshold": threshold,
                **kwargs,
            },
            cor_matrix=cormat,
            selected_lambda=best_lambda,
            selected_ebic=best_ebic,
            lambda_ebic_curve=pd.DataFrame(curve_records),
        )

        return Network(
            adjacency=pcor,
            labels=list(data.columns),
            method=self.name,
            n_observations=n,
            weighted=True,
            signed=True,
            directed=False,
            estimation_info=info,
        )
