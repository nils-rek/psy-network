"""Correlation network estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._types import CorMethod
from ..estimation_info import EstimationInfo
from ..network import Network
from ._registry import register
from ._validate import _validate_estimation_data


@register("cor")
class CorEstimator:
    """Network where edge weights are pairwise correlations."""

    name: str = "cor"

    def estimate(
        self,
        data: pd.DataFrame,
        *,
        cor_method: str | CorMethod = CorMethod.PEARSON,
        threshold: float = 0.0,
        **kwargs,
    ) -> Network:
        cor_method = CorMethod(cor_method)
        n_effective = _validate_estimation_data(data)
        full_cormat = data.corr(method=cor_method.value).values.copy()
        cormat = full_cormat.copy()
        np.fill_diagonal(cormat, 0.0)
        if threshold > 0:
            cormat[np.abs(cormat) < threshold] = 0.0
        info = EstimationInfo(
            method=self.name,
            est_kwargs={
                "cor_method": cor_method.value,
                "threshold": threshold,
                **kwargs,
            },
            # The true correlation matrix (unit diagonal), matching what
            # pcor and EBICglasso store
            cor_matrix=full_cormat,
        )
        return Network(
            adjacency=cormat,
            labels=list(data.columns),
            method=self.name,
            n_observations=n_effective,
            weighted=True,
            signed=True,
            directed=False,
            estimation_info=info,
        )
