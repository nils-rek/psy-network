"""Temporal network estimation via mixed-effects models."""

from __future__ import annotations

import warnings as _warnings
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
    fit_info: dict                  # BIC, AIC, warnings per DV
    intercepts: dict | None         # subject_id -> length-p array of intercepts


# Patterns indicating severe convergence issues that warrant RE fallback
_SEVERE_WARNING_PATTERNS = (
    "singular",
    "not positive definite",
    "optimization failed",
    "on the boundary",
)


def _has_severe_warnings(warn_messages: list[str]) -> bool:
    """Check if any warning messages indicate severe convergence issues."""
    for msg in warn_messages:
        msg_lower = msg.lower()
        if any(pat in msg_lower for pat in _SEVERE_WARNING_PATTERNS):
            return True
    return False


def _build_model_kwargs(
    formula: str,
    model_data: pd.DataFrame,
    subject: str,
    lag_cols: list[str],
    temporal_re: str,
) -> dict:
    """Build keyword arguments for statsmodels MixedLM."""
    model_kwargs: dict = {
        "formula": formula,
        "data": model_data,
        "groups": model_data[subject],
    }

    if temporal_re == "correlated":
        model_kwargs["re_formula"] = " + ".join(lag_cols)
    elif temporal_re == "orthogonal":
        model_kwargs["re_formula"] = "1"
        model_kwargs["vc_formula"] = {lc: f"0 + {lc}" for lc in lag_cols}
    elif temporal_re == "fixed":
        model_kwargs["re_formula"] = "1"
    else:
        raise ValueError(
            f"temporal must be 'correlated', 'orthogonal', or 'fixed', "
            f"got {temporal_re!r}"
        )
    return model_kwargs


def _try_fit(model_kwargs: dict, method: str = "lbfgs") -> tuple:
    """Fit a mixed-effects model, capturing warnings.

    Returns (result, warn_messages) or raises on total failure.
    """
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        model = smf.mixedlm(**model_kwargs)
        result = model.fit(reml=True, method=method)
    warn_messages = [str(w.message) for w in caught]
    return result, warn_messages


# RE structures from most complex to simplest
_RE_FALLBACK_CHAIN = ("correlated", "orthogonal", "fixed")


def _fit_one_dv(
    j: int,
    var_cols: list[str],
    lag_data: pd.DataFrame,
    subject: str,
    temporal_re: str = "correlated",
) -> dict:
    """Fit a single mixed-effects model for DV var_cols[j].

    Captures convergence warnings and falls back to simpler random-effects
    structures when severe convergence issues occur.

    Parameters
    ----------
    temporal_re : str
        Random effects structure: ``"correlated"`` (full covariance),
        ``"orthogonal"`` (diagonal / uncorrelated random slopes),
        or ``"fixed"`` (random intercepts only, no random slopes).
    """
    dv = var_cols[j]
    lag_cols = [f"{c}_lag" for c in var_cols]

    # Per-model listwise deletion: only drop rows where this DV or its
    # lagged predictors are NaN (matching R's mlVAR behavior)
    needed_cols = [dv] + lag_cols
    model_data = lag_data.dropna(subset=needed_cols)

    formula = f"{dv} ~ " + " + ".join(lag_cols)

    # Validate temporal_re before determining fallback chain
    if temporal_re not in _RE_FALLBACK_CHAIN:
        raise ValueError(
            f"temporal must be 'correlated', 'orthogonal', or 'fixed', "
            f"got {temporal_re!r}"
        )

    # Determine fallback chain starting from the requested RE structure
    start_idx = _RE_FALLBACK_CHAIN.index(temporal_re)
    re_chain = _RE_FALLBACK_CHAIN[start_idx:]

    result = None
    warn_messages: list[str] = []
    actual_re = temporal_re

    for re_structure in re_chain:
        try:
            model_kwargs = _build_model_kwargs(
                formula, model_data, subject, lag_cols, re_structure,
            )
        except ValueError:
            raise  # Invalid temporal_re value — don't catch

        try:
            result, warn_messages = _try_fit(model_kwargs, method="lbfgs")
            actual_re = re_structure

            # If severe convergence warnings, try Nelder-Mead optimizer
            if _has_severe_warnings(warn_messages):
                try:
                    result_nm, warns_nm = _try_fit(model_kwargs, method="nm")
                    if not _has_severe_warnings(warns_nm):
                        result = result_nm
                        warn_messages = warns_nm
                except Exception:
                    pass  # keep lbfgs result

            break  # Accept the result (warnings are surfaced, not fatal)

        except Exception:
            # Total fit failure — try next simpler RE structure
            if re_structure == re_chain[-1]:
                raise  # nothing left to try
            _warnings.warn(
                f"mlVAR DV={dv!r}: fit failed with "
                f"temporal={re_structure!r}, falling back to simpler "
                f"random-effects structure.",
                UserWarning,
                stacklevel=2,
            )
            continue

    if result is None:
        raise RuntimeError(f"All RE structures failed for DV={dv!r}")

    # Fixed effects (skip intercept)
    fixed = np.array([result.fe_params.get(lc, 0.0) for lc in lag_cols])
    pvals = np.array([result.pvalues.get(lc, 1.0) for lc in lag_cols])

    # Per-subject coefficients (fixed + random effects)
    # Also extract random intercepts for between-subjects network
    fixed_intercept = result.fe_params.get("Intercept", 0.0)
    subject_coefs = {}
    subject_intercepts = {}
    if actual_re == "fixed":
        # Random intercepts only — no random slopes
        try:
            re_dict = result.random_effects
        except (np.linalg.LinAlgError, ValueError):
            re_dict = None
        for subj_id in model_data[subject].unique():
            subject_coefs[subj_id] = fixed.copy()
            re_intercept = 0.0
            if re_dict is not None and subj_id in re_dict:
                re_vals = re_dict[subj_id]
                if "Group" in re_vals.index:
                    re_intercept = re_vals["Group"]
                elif "Intercept" in re_vals.index:
                    re_intercept = re_vals["Intercept"]
            subject_intercepts[subj_id] = fixed_intercept + re_intercept
    else:
        try:
            re_dict = result.random_effects
        except (np.linalg.LinAlgError, ValueError):
            # Singular covariance — fall back to fixed-effect coefficients
            re_dict = None
        if re_dict is not None:
            for subj_id, re_vals in re_dict.items():
                subj_fixed = fixed.copy()
                for k, lc in enumerate(lag_cols):
                    if lc in re_vals.index:
                        subj_fixed[k] += re_vals[lc]
                subject_coefs[subj_id] = subj_fixed
                # Extract random intercept (key varies by RE structure)
                re_intercept = 0.0
                if "Group" in re_vals.index:
                    re_intercept = re_vals["Group"]
                elif "Intercept" in re_vals.index:
                    re_intercept = re_vals["Intercept"]
                subject_intercepts[subj_id] = fixed_intercept + re_intercept
        else:
            for subj_id in model_data[subject].unique():
                subject_coefs[subj_id] = fixed.copy()
                subject_intercepts[subj_id] = fixed_intercept

    # Residuals — fall back to fixed-effect-only residuals when
    # statsmodels cannot predict random effects (singular covariance)
    try:
        residuals = result.resid
    except (np.linalg.LinAlgError, ValueError):
        y = model_data[dv].values
        X_mat = model_data[lag_cols].values
        residuals = pd.Series(
            y - fixed_intercept - X_mat @ fixed,
            index=model_data.index,
        )

    return {
        "j": j,
        "fixed": fixed,
        "pvals": pvals,
        "subject_coefs": subject_coefs,
        "subject_intercepts": subject_intercepts,
        "residuals": residuals,
        "bic": result.bic,
        "aic": result.aic,
        "warnings": warn_messages,
        "actual_re": actual_re,
    }


def _emit_convergence_summary(fit_info: dict, requested_re: str) -> None:
    """Emit a UserWarning summarising convergence issues across DVs."""
    warn_dvs = []
    fallback_dvs = []
    for var, info in fit_info.items():
        if info.get("warnings"):
            warn_dvs.append(var)
        actual = info.get("actual_re", requested_re)
        if actual != requested_re:
            fallback_dvs.append(f"{var} ({requested_re} -> {actual})")

    parts = []
    if fallback_dvs:
        parts.append(
            "RE structure fallback occurred for: "
            + ", ".join(fallback_dvs)
        )
    if warn_dvs:
        parts.append(
            "Convergence warnings for: "
            + ", ".join(warn_dvs)
            + ". Inspect result.fit_info[var]['warnings'] for details."
        )
    if parts:
        _warnings.warn(
            "mlVAR temporal estimation: " + " | ".join(parts),
            UserWarning,
            stacklevel=3,
        )


def estimate_multilevel_temporal(
    lag_data: pd.DataFrame,
    var_cols: list[str],
    subject: str,
    *,
    temporal_re: str = "correlated",
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
    temporal_re : str
        Random effects structure: ``"correlated"``, ``"orthogonal"``,
        or ``"fixed"``.
    n_cores : int
        Number of parallel jobs for fitting models.

    Returns
    -------
    _TemporalResult
    """
    p = len(var_cols)

    results = Parallel(n_jobs=n_cores)(
        delayed(_fit_one_dv)(j, var_cols, lag_data, subject, temporal_re)
        for j in range(p)
    )

    # Assemble matrices
    fixed_coef = np.zeros((p, p))
    pvalues = np.zeros((p, p))
    fit_info = {}

    # Collect all subject IDs
    all_subjects = sorted(lag_data[subject].unique())
    subject_coefs = {s: np.zeros((p, p)) for s in all_subjects}

    # Collect residuals and intercepts
    residual_series = {}
    intercept_data: dict[str, np.ndarray] = {}  # subject_id -> length-p array
    intercepts_available = True

    for res in results:
        j = res["j"]
        fixed_coef[j, :] = res["fixed"]
        pvalues[j, :] = res["pvals"]
        fit_info[var_cols[j]] = {
            "bic": res["bic"],
            "aic": res["aic"],
            "warnings": res.get("warnings", []),
            "actual_re": res.get("actual_re", temporal_re),
        }

        for s, coef_row in res["subject_coefs"].items():
            subject_coefs[s][j, :] = coef_row

        # Assemble per-subject intercepts across DVs
        subj_intercepts = res.get("subject_intercepts", {})
        if not subj_intercepts:
            intercepts_available = False
        for s, intercept_val in subj_intercepts.items():
            if s not in intercept_data:
                intercept_data[s] = np.zeros(p)
            intercept_data[s][j] = intercept_val

        residual_series[var_cols[j]] = res["residuals"]

    # Surface convergence summary to the user
    _emit_convergence_summary(fit_info, temporal_re)

    # Build residuals DataFrame — each DV model may have used a different
    # subset of rows (per-model listwise deletion), so NaN fills gaps
    residuals_df = pd.DataFrame(index=lag_data.index)
    for col_name, resid_series in residual_series.items():
        residuals_df[col_name] = resid_series
    residuals_df[subject] = lag_data[subject].values

    return _TemporalResult(
        fixed_coef=fixed_coef,
        pvalues=pvalues,
        subject_coefs=subject_coefs,
        residuals_df=residuals_df,
        fit_info=fit_info,
        intercepts=intercept_data if intercepts_available else None,
    )
