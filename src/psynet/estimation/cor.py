"""Correlation network estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._types import CorMethod
from ..network import Network
from ._registry import register


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
        cormat = data.corr(method=cor_method.value).values.copy()
        np.fill_diagonal(cormat, 0.0)
        if threshold > 0:
            cormat[np.abs(cormat) < threshold] = 0.0
        return Network(
            adjacency=cormat,
            labels=list(data.columns),
            method=self.name,
            n_observations=len(data),
            weighted=True,
            signed=True,
            directed=False,
        )
