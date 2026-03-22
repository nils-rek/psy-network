"""Contemporaneous network estimation from pooled mlVAR residuals."""

from __future__ import annotations

import numpy as np

from .._glasso_utils import _fit_ebic_glasso
from ..network import Network


def estimate_mlvar_contemporaneous(
    residuals_df,
    var_cols: list[str],
    subject: str,
    *,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
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
    residuals = residuals_df[var_cols].values
    T, p = residuals.shape
    cormat = np.corrcoef(residuals, rowvar=False)

    pcor, _, _, _ = _fit_ebic_glasso(
        cormat, T,
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
    )

    return Network(
        adjacency=pcor,
        labels=var_cols,
        method="mlVAR",
        n_observations=T,
        weighted=True,
        signed=True,
        directed=False,
    )
