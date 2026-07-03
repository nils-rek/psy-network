"""Contemporaneous network estimation from pooled multilevel VAR residuals."""

from __future__ import annotations

import numpy as np

from .._glasso_utils import contemporaneous_from_residuals
from ..network import Network


def estimate_multilevel_contemporaneous(
    residuals_df,
    var_cols: list[str],
    subject: str,
    *,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
    n_jobs: int = 1,
) -> Network:
    """Estimate contemporaneous network from pooled residuals via EBIC-glasso.

    Pools residuals across all subjects and applies the graphical lasso
    with EBIC model selection.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Residuals with variable columns and a subject column.
    var_cols : list[str]
        Variable column names.
    subject : str
        Subject identifier column name.
    gamma : float
        EBIC gamma parameter.
    n_lambda : int
        Number of lambda values in the search grid.
    lambda_min_ratio : float
        Ratio of minimum to maximum lambda.
    threshold : float
        Threshold for zeroing small partial correlations.

    Returns
    -------
    Network
        Undirected contemporaneous network.
    """
    # Drop rows with NaN (from per-model listwise deletion in temporal step)
    residuals = residuals_df[var_cols].dropna().values
    return contemporaneous_from_residuals(
        residuals, var_cols, "mlVAR",
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
    )
