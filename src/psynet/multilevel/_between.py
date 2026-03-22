"""Between-subjects network estimation via EBIC-glasso on subject means."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._glasso_utils import _fit_ebic_glasso
from ..network import Network


def estimate_between_subjects(
    data: pd.DataFrame,
    var_cols: list[str],
    subject: str,
    *,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
) -> Network:
    """Estimate between-subjects network from subject-level means.

    Computes per-subject means and applies EBIC-tuned graphical lasso
    to the correlation matrix of those means.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data with subject column.
    var_cols : list[str]
        Variable column names.
    subject : str
        Subject identifier column.
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
        Undirected between-subjects network.
    """
    means = data.groupby(subject)[var_cols].mean()
    n_subjects = len(means)
    cormat = np.corrcoef(means.values, rowvar=False)

    pcor, _, _, _ = _fit_ebic_glasso(
        cormat, n_subjects,
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
    )

    return Network(
        adjacency=pcor,
        labels=var_cols,
        method="mlVAR",
        n_observations=n_subjects,
        weighted=True,
        signed=True,
        directed=False,
    )
