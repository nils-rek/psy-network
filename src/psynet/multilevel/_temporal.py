"""Temporal network estimation via mixed-effects models."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    raise ImportError(
        "statsmodels is required for multilevel VAR estimation. "
        "Install it with: pip install psynet[multilevel]"
    )

from joblib import Parallel, delayed


class _TemporalResult(NamedTuple):
    """Result container for temporal network estimation."""
    fixed_coef: np.ndarray          # p x p fixed-effect coefficient matrix
    pvalues: np.ndarray             # p x p p-value matrix
    subject_coefs: dict             # subject_id -> p x p coefficient matrix
    residuals_df: pd.DataFrame      # residuals with subject column
    fit_info: dict                  # BIC, AIC per DV


def _fit_one_dv(
    j: int,
    var_cols: list[str],
    lag_data: pd.DataFrame,
    subject: str,
) -> dict:
    """Fit a single mixed-effects model for DV var_cols[j]."""
    dv = var_cols[j]
    lag_cols = [f"{c}_lag" for c in var_cols]

    formula = f"{dv} ~ " + " + ".join(lag_cols)

    # Random slopes for all lagged predictors
    re_formula = " + ".join(lag_cols)

    model = smf.mixedlm(
        formula,
        data=lag_data,
        groups=lag_data[subject],
        re_formula=re_formula,
    )
    result = model.fit(reml=True, method="lbfgs")

    # Fixed effects (skip intercept)
    fixed = np.array([result.fe_params.get(lc, 0.0) for lc in lag_cols])
    pvals = np.array([result.pvalues.get(lc, 1.0) for lc in lag_cols])

    # Per-subject coefficients (fixed + random effects)
    subject_coefs = {}
    for subj_id, re_vals in result.random_effects.items():
        subj_fixed = fixed.copy()
        for k, lc in enumerate(lag_cols):
            if lc in re_vals.index:
                subj_fixed[k] += re_vals[lc]
        subject_coefs[subj_id] = subj_fixed

    # Residuals
    residuals = result.resid

    return {
        "j": j,
        "fixed": fixed,
        "pvals": pvals,
        "subject_coefs": subject_coefs,
        "residuals": residuals,
        "bic": result.bic,
        "aic": result.aic,
    }


def estimate_multilevel_temporal(
    lag_data: pd.DataFrame,
    var_cols: list[str],
    subject: str,
    *,
    n_cores: int = 1,
) -> _TemporalResult:
    """Estimate population and subject-level temporal networks via mixed-effects.

    For each variable as DV, fits a linear mixed-effects model with all
    lagged variables as fixed effects and random slopes per subject.

    Parameters
    ----------
    lag_data : pd.DataFrame
        Lagged data from :func:`make_multilevel_lag_data`.
    var_cols : list[str]
        Variable column names.
    subject : str
        Subject identifier column.
    n_cores : int
        Number of parallel jobs for fitting models.

    Returns
    -------
    _TemporalResult
    """
    p = len(var_cols)

    results = Parallel(n_jobs=n_cores)(
        delayed(_fit_one_dv)(j, var_cols, lag_data, subject)
        for j in range(p)
    )

    # Assemble matrices
    fixed_coef = np.zeros((p, p))
    pvalues = np.zeros((p, p))
    fit_info = {}

    # Collect all subject IDs
    all_subjects = sorted(lag_data[subject].unique())
    subject_coefs = {s: np.zeros((p, p)) for s in all_subjects}

    # Collect residuals
    residual_series = {}
    for res in results:
        j = res["j"]
        fixed_coef[j, :] = res["fixed"]
        pvalues[j, :] = res["pvals"]
        fit_info[var_cols[j]] = {"bic": res["bic"], "aic": res["aic"]}

        for s, coef_row in res["subject_coefs"].items():
            subject_coefs[s][j, :] = coef_row

        residual_series[var_cols[j]] = res["residuals"]

    # Build residuals DataFrame
    residuals_df = pd.DataFrame(residual_series, index=lag_data.index)
    residuals_df[subject] = lag_data[subject].values

    return _TemporalResult(
        fixed_coef=fixed_coef,
        pvalues=pvalues,
        subject_coefs=subject_coefs,
        residuals_df=residuals_df,
        fit_info=fit_info,
    )
