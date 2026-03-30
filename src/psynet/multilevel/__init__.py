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
    temporal_alpha: float | None = 0.05,
    temporal: str = "correlated",
    gamma: float = 0.5,
    between_gamma: float | None = None,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    threshold: float = 1e-4,
    n_cores: int = 1,
    scale: bool = False,
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
    temporal_alpha : float or None
        Significance level for thresholding temporal edges.  Edges with
        p-value > ``temporal_alpha`` are zeroed out, matching R's mlVAR
        ``nonsig="hide"`` behaviour.  Set to ``None`` to keep all edges
        (equivalent to R's ``nonsig="show"``).  Default ``0.05``.
    temporal : str
        Random effects structure for temporal estimation, matching R's
        mlVAR ``temporal`` argument: ``"correlated"`` (full random slopes
        covariance, default), ``"orthogonal"`` (uncorrelated random
        slopes), or ``"fixed"`` (random intercepts only, no random
        slopes).  Use ``"orthogonal"`` or ``"fixed"`` when the number
        of variables is large and convergence issues arise.
    gamma : float
        EBIC gamma parameter for contemporaneous and between-subjects networks.
    between_gamma : float or None
        EBIC gamma for the between-subjects network only.  Defaults to
        ``gamma`` if not specified.  Lower values (e.g. 0.25) produce denser
        networks, which may be appropriate when the number of subjects is
        small relative to the number of variables.
    n_lambda : int
        Number of lambda values in the EBIC glasso grid.
    lambda_min_ratio : float
        Ratio of minimum to maximum lambda.
    threshold : float
        Threshold for zeroing small coefficients/partial correlations.
    n_cores : int
        Number of parallel jobs for temporal estimation.
    scale : bool
        If True, z-score standardize each variable column within each
        subject before estimation.  This improves optimizer convergence
        for variables on wide scales (e.g. 0–100 VAS) by reducing the
        chance that all models fall back to simpler random-effects
        structures.  Coefficients represent standardised effects.
        Default False.

    Returns
    -------
    MultilevelNetwork
    """
    var_cols = validate_multilevel_data(data, subject, beep=beep, day=day)

    # Keep original data for between-subjects network (subject means
    # must be computed on unscaled data; z-scoring centres everyone at 0).
    original_data = data
    if scale:
        data = data.copy()
        for col in var_cols:
            data[col] = data.groupby(subject)[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
            )

    lag_data = make_multilevel_lag_data(data, var_cols, subject, beep=beep, day=day)

    # Step 1: Temporal network via mixed-effects models
    temporal_result = estimate_multilevel_temporal(
        lag_data, var_cols, subject, temporal_re=temporal, n_cores=n_cores,
    )

    # Threshold small temporal coefficients
    fixed_coef = temporal_result.fixed_coef.copy()
    fixed_coef[np.abs(fixed_coef) < threshold] = 0.0

    # P-value thresholding (matches R's mlVAR nonsig="hide")
    unthresholded_temporal = None
    pval_mask = None
    if temporal_alpha is not None:
        pval_mask = temporal_result.pvalues > temporal_alpha
        # Store unthresholded version before zeroing
        unthresholded_temporal = Network(
            adjacency=fixed_coef.copy(),
            labels=var_cols,
            method="mlVAR",
            n_observations=len(lag_data),
            weighted=True,
            signed=True,
            directed=True,
        )
        fixed_coef[pval_mask] = 0.0

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
        if pval_mask is not None:
            s_coef[pval_mask] = 0.0
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
        n_jobs=n_cores,
    )

    # Step 3: Between-subjects network from random intercepts (or raw means)
    between_net = estimate_between_subjects(
        original_data, var_cols, subject,
        intercepts=temporal_result.intercepts,
        gamma=between_gamma if between_gamma is not None else gamma,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        threshold=threshold,
        n_jobs=n_cores,
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
        unthresholded_temporal=unthresholded_temporal,
    )


__all__ = [
    "estimate_multilevel_network",
    "MultilevelNetwork",
]
