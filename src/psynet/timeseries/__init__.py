"""Time-series network estimation via graphicalVAR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..network import Network
from ._contemporaneous import estimate_contemporaneous
from ._validation import make_lag_matrix, validate_ts_data
from ._var import estimate_temporal
from .network import TSNetwork


def estimate_var_network(
    data: pd.DataFrame,
    *,
    beep: str | None = None,
    day: str | None = None,
    cv: int = 10,
    n_alphas: int = 100,
    max_iter: int = 10000,
    var_threshold: float = 1e-4,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    contemp_threshold: float = 1e-4,
    n_jobs: int = 1,
) -> TSNetwork:
    """Estimate a time-series network using graphicalVAR.

    Two-step approach:
    1. Sparse VAR(1) via column-wise LassoCV → temporal (directed) network
    2. EBIC graphical lasso on VAR residuals → contemporaneous (undirected) network

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data with variables as columns. Rows are ordered
        timepoints.
    beep : str, optional
        Column for beep/measurement index within a day. If provided with
        ``day``, only consecutive observations within the same day are used.
    day : str, optional
        Column for day identifier.
    cv : int
        Cross-validation folds for LassoCV (temporal estimation).
    n_alphas : int
        Number of alpha values for LassoCV.
    max_iter : int
        Maximum iterations for Lasso.
    var_threshold : float
        Threshold for zeroing small VAR coefficients.
    gamma : float
        EBIC gamma for contemporaneous network estimation.
    n_lambda : int
        Number of lambda values in the EBIC glasso grid.
    lambda_min_ratio : float
        Ratio of minimum to maximum lambda.
    contemp_threshold : float
        Threshold for zeroing small partial correlations.
    n_jobs : int
        Number of parallel jobs for variable-wise model fitting.
        ``1`` (default) runs serially; ``-1`` uses all cores.

    Returns
    -------
    TSNetwork
    """
    var_cols = validate_ts_data(data, beep, day)
    n_timepoints = len(data)

    X, Y = make_lag_matrix(data, var_cols, beep, day)

    temporal_net, residuals = estimate_temporal(
        X, Y, var_cols,
        cv=cv,
        n_alphas=n_alphas,
        max_iter=max_iter,
        threshold=var_threshold,
        n_jobs=n_jobs,
    )

    contemporaneous_net = estimate_contemporaneous(
        residuals, var_cols,
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=contemp_threshold,
        n_jobs=n_jobs,
    )

    return TSNetwork(
        temporal=temporal_net,
        contemporaneous=contemporaneous_net,
        labels=var_cols,
        method="graphicalVAR",
        n_observations=len(Y),
        n_timepoints=n_timepoints,
    )


__all__ = [
    "estimate_var_network",
    "TSNetwork",
]
