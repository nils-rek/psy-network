"""Multilevel VAR network estimation for multi-subject ESM data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..network import Network
from ._between import estimate_between_subjects
from ._contemporaneous import estimate_multilevel_contemporaneous
from ._temporal import estimate_multilevel_temporal
from ._validation import make_multilevel_lag_data, validate_multilevel_data
from .network import MultilevelNetwork


def estimate_multilevel_network(
    data: pd.DataFrame,
    subject: str,
    *,
    beep: str | None = None,
    day: str | None = None,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
    n_cores: int = 1,
) -> MultilevelNetwork:
    """Estimate a multilevel VAR network from ESM data.

    Produces three networks:
    1. **Temporal** — directed, lag-1 fixed effects from mixed-effects VAR
    2. **Contemporaneous** — undirected, EBIC-glasso on pooled residuals
    3. **Between-subjects** — undirected, EBIC-glasso on subject means

    Parameters
    ----------
    data : pd.DataFrame
        Long-format ESM data with a subject identifier column and
        numeric variable columns.
    subject : str
        Name of the subject identifier column.
    beep : str, optional
        Column for beep/measurement index within a day.
    day : str, optional
        Column for day identifier.
    gamma : float
        EBIC gamma parameter for contemporaneous and between-subjects networks.
    n_lambda : int
        Number of lambda values in the EBIC glasso grid.
    lambda_min_ratio : float
        Ratio of minimum to maximum lambda.
    threshold : float
        Threshold for zeroing small coefficients/partial correlations.
    n_cores : int
        Number of parallel jobs for temporal estimation.

    Returns
    -------
    MultilevelNetwork
    """
    var_cols = validate_multilevel_data(data, subject, beep=beep, day=day)
    lag_data = make_multilevel_lag_data(data, var_cols, subject, beep=beep, day=day)

    # Step 1: Temporal network via mixed-effects models
    temporal_result = estimate_multilevel_temporal(
        lag_data, var_cols, subject, n_cores=n_cores,
    )

    # Threshold small temporal coefficients
    fixed_coef = temporal_result.fixed_coef.copy()
    fixed_coef[np.abs(fixed_coef) < threshold] = 0.0

    temporal_net = Network(
        adjacency=fixed_coef,
        labels=var_cols,
        method="mlVAR",
        n_observations=len(lag_data),
        weighted=True,
        signed=True,
        directed=True,
    )

    # Per-subject temporal networks
    subject_ids = sorted(data[subject].unique())
    subject_temporal = {}
    for s in subject_ids:
        s_coef = temporal_result.subject_coefs[s].copy()
        s_coef[np.abs(s_coef) < threshold] = 0.0
        subject_temporal[s] = Network(
            adjacency=s_coef,
            labels=var_cols,
            method="mlVAR",
            n_observations=int((lag_data[subject] == s).sum()),
            weighted=True,
            signed=True,
            directed=True,
        )

    # Step 2: Contemporaneous network from pooled residuals
    contemporaneous_net = estimate_multilevel_contemporaneous(
        temporal_result.residuals_df, var_cols, subject,
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
    )

    # Step 3: Between-subjects network from subject means
    between_net = estimate_between_subjects(
        data, var_cols, subject,
        gamma=gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
    )

    return MultilevelNetwork(
        temporal=temporal_net,
        contemporaneous=contemporaneous_net,
        between_subjects=between_net,
        subject_temporal=subject_temporal,
        labels=var_cols,
        subject_ids=subject_ids,
        method="mlVAR",
        pvalues=temporal_result.pvalues,
        fit_info=temporal_result.fit_info,
    )


__all__ = [
    "estimate_multilevel_network",
    "MultilevelNetwork",
]
