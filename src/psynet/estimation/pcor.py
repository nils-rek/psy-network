"""Partial correlation network estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from .._types import CorMethod
from ..estimation_info import EstimationInfo
from ..network import Network
from ._registry import register
from ._validate import _validate_estimation_data


def _partial_correlations(cormat: np.ndarray) -> np.ndarray:
    """Compute partial correlations from a correlation matrix.

    pcor_ij = -P_ij / sqrt(P_ii * P_jj) where P = inv(cormat).
    Falls back to the pseudo-inverse when the correlation matrix is
    singular or near-singular (e.g. collinear items or p >= n), where
    a plain inverse returns numerically meaningless values.
    """
    import warnings

    cond = np.linalg.cond(cormat)
    if not np.isfinite(cond) or cond > 1e12:
        warnings.warn(
            "Correlation matrix is singular or near-singular (collinear "
            "variables or p >= n); using the pseudo-inverse. Partial "
            "correlations may be unstable — consider EBICglasso instead.",
            UserWarning,
            stacklevel=3,
        )
        precision = linalg.pinv(cormat)
    else:
        precision = linalg.inv(cormat)
    from .._glasso_utils import precision_to_pcor
    return precision_to_pcor(precision)


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
        n_effective = _validate_estimation_data(data)
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
            n_observations=n_effective,
            weighted=True,
            signed=True,
            directed=False,
            estimation_info=info,
        )
