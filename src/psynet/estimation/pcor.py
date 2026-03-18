"""Partial correlation network estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from .._types import CorMethod
from ..estimation_info import EstimationInfo
from ..network import Network
from ._registry import register


def _partial_correlations(cormat: np.ndarray) -> np.ndarray:
    """Compute partial correlations from a correlation matrix.

    pcor_ij = -P_ij / sqrt(P_ii * P_jj) where P = inv(cormat).
    """
    precision = linalg.inv(cormat)
    diag = np.sqrt(np.diag(precision))
    pcor = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcor, 0.0)
    return pcor


@register("pcor")
class PCorEstimator:
    """Network where edge weights are partial correlations."""

    name: str = "pcor"

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
        pcor = _partial_correlations(cormat)
        if threshold > 0:
            pcor[np.abs(pcor) < threshold] = 0.0
        info = EstimationInfo(
            method=self.name,
            est_kwargs={
                "cor_method": cor_method.value,
                "threshold": threshold,
                **kwargs,
            },
            cor_matrix=cormat,
        )
        return Network(
            adjacency=pcor,
            labels=list(data.columns),
            method=self.name,
            n_observations=len(data),
            weighted=True,
            signed=True,
            directed=False,
            estimation_info=info,
        )
