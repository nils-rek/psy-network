"""EBICglasso estimator — EBIC-tuned graphical lasso."""

from __future__ import annotations

import pandas as pd

from .._glasso_utils import _fit_ebic_glasso
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
        n_jobs: int = 1,
        **kwargs,
    ) -> Network:
        cor_method = CorMethod(cor_method)
        n, p = data.shape
        cormat = data.corr(method=cor_method.value).values.copy()

        pcor, best_lambda, best_ebic, curve_df = _fit_ebic_glasso(
            cormat, n,
            gamma=gamma,
            n_lambda=n_lambda,
            lambda_min_ratio=lambda_min_ratio,
            threshold=threshold,
            track_curve=True,
        )

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
            lambda_ebic_curve=curve_df,
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
