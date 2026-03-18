"""EstimationInfo dataclass — metadata from network estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EstimationInfo:
    """Immutable container for estimation diagnostics and metadata.

    Parameters
    ----------
    method : str
        Estimation method used.
    est_kwargs : dict[str, Any]
        Keyword arguments passed to the estimator.
    cor_matrix : np.ndarray | None
        Input correlation matrix used for estimation.
    selected_lambda : float | None
        Chosen regularization parameter (EBICglasso only).
    selected_ebic : float | None
        EBIC score at the chosen lambda (EBICglasso only).
    lambda_ebic_curve : pd.DataFrame | None
        Full lambda–EBIC curve with columns ``["lambda", "ebic"]``
        (EBICglasso only).
    """

    method: str
    est_kwargs: dict[str, Any] = field(default_factory=dict)
    cor_matrix: np.ndarray | None = None
    selected_lambda: float | None = None
    selected_ebic: float | None = None
    lambda_ebic_curve: pd.DataFrame | None = None
