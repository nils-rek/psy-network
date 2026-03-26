"""Between-subjects network estimation via EBIC-glasso on subject intercepts."""

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
    intercepts: dict | None = None,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
    n_jobs: int = 1,
) -> Network:
    """Estimate between-subjects network from subject-level intercepts.

    When random intercepts from the temporal mixed-effects models are
    available, uses those to build the between-subjects correlation
    matrix (matching R's mlVAR).  Random intercepts account for temporal
    structure and unequal observations per subject.  Falls back to raw
    subject means when intercepts are not available.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data with subject column.
    var_cols : list[str]
        Variable column names.
    subject : str
        Subject identifier column.
    intercepts : dict or None
        Subject-level intercepts from temporal estimation.
        Mapping ``{subject_id: np.ndarray of shape (p,)}``.
        When provided, the correlation matrix is built from these
        intercepts instead of raw means.
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
    use_intercepts = False
    if intercepts is not None and len(intercepts) >= 3:
        # Use random intercepts from mixed-effects models (R's mlVAR approach)
        subject_ids = sorted(intercepts.keys())
        n_subjects = len(subject_ids)
        intercept_matrix = np.array([intercepts[s] for s in subject_ids])
        # Check for zero-variance columns (degenerate intercepts)
        stds = np.std(intercept_matrix, axis=0)
        if np.all(stds > 1e-10):
            cormat = np.corrcoef(intercept_matrix, rowvar=False)
            use_intercepts = True

    if not use_intercepts:
        # Fallback: raw subject means
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
