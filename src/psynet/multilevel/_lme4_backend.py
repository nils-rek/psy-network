"""Optional lme4 backend for temporal mixed-effects estimation via rpy2.

Requires R with packages ``lme4`` and ``lmerTest``, plus the Python
package ``rpy2``.  All R interaction is isolated in this module.
"""

from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_lme4_available() -> None:
    """Raise ImportError with actionable message if rpy2/lme4 unavailable."""
    try:
        import rpy2.robjects  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "engine='lme4' requires the rpy2 Python package. "
            "Install with: pip install rpy2"
        ) from e

    from rpy2.robjects.packages import importr, PackageNotInstalledError

    for pkg in ("lme4", "lmerTest"):
        try:
            importr(pkg)
        except PackageNotInstalledError as e:
            raise ImportError(
                f"engine='lme4' requires the R package '{pkg}'. "
                f"Install with: Rscript -e 'install.packages(\"{pkg}\")'"
            ) from e


# ---------------------------------------------------------------------------
# Formula builder
# ---------------------------------------------------------------------------

def _build_lmer_formula(
    dv: str,
    lag_cols: list[str],
    subject: str,
    temporal_re: str,
) -> str:
    """Build an lmer formula string for the given RE structure."""
    predictors = " + ".join(lag_cols)

    if temporal_re == "correlated":
        re_term = f"({predictors} | {subject})"
    elif temporal_re == "orthogonal":
        re_terms = [f"(1 | {subject})"] + [
            f"(0 + {lc} | {subject})" for lc in lag_cols
        ]
        re_term = " + ".join(re_terms)
    elif temporal_re == "fixed":
        re_term = f"(1 | {subject})"
    else:
        raise ValueError(
            f"temporal must be 'correlated', 'orthogonal', or 'fixed', "
            f"got {temporal_re!r}"
        )

    return f"{dv} ~ {predictors} + {re_term}"


# ---------------------------------------------------------------------------
# R warning detection
# ---------------------------------------------------------------------------

_R_SEVERE_PATTERNS = (
    "failed to converge",
    "singular",
    "unable to evaluate",
    "boundary",
)


def _has_severe_r_warnings(warnings_list: list[str]) -> bool:
    """Check if any R warning messages indicate severe convergence issues."""
    for msg in warnings_list:
        if any(pat in msg.lower() for pat in _R_SEVERE_PATTERNS):
            return True
    return False


# ---------------------------------------------------------------------------
# R fitting helper
# ---------------------------------------------------------------------------

def _setup_r_env():
    """Set up the R environment and return helper objects.

    Returns (ro, pandas2ri, lmerTest_pkg, base_pkg).
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    lmerTest_pkg = importr("lmerTest")
    base_pkg = importr("base")
    return ro, pandas2ri, lmerTest_pkg, base_pkg


def _fit_lmer_with_warnings(formula_str: str, data_r, ro, lmerTest_pkg):
    """Call lmerTest::lmer() and capture R warnings.

    Returns (model, warnings_list).
    """
    # Define R helper that captures warnings
    ro.r('''
    .psynet_fit_lmer <- function(formula_str, data) {
        warns <- character(0)
        result <- tryCatch(
            withCallingHandlers(
                lmerTest::lmer(as.formula(formula_str), data = data, REML = TRUE),
                warning = function(w) {
                    warns <<- c(warns, conditionMessage(w))
                    invokeRestart("muffleWarning")
                }
            ),
            error = function(e) {
                list(error = conditionMessage(e))
            }
        )
        list(model = result, warnings = warns)
    }
    ''')

    fit_result = ro.r['.psynet_fit_lmer'](formula_str, data_r)

    # Extract model and warnings
    model = fit_result.rx2('model')
    r_warnings = list(fit_result.rx2('warnings'))

    # Check if an error occurred (model will be a list with 'error' key)
    try:
        error_msg = model.rx2('error')
        if error_msg is not None:
            raise RuntimeError(f"R lmer() failed: {error_msg[0]}")
    except Exception:
        pass  # model is an lmerMod object, not a list — this is expected

    return model, r_warnings


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _extract_lme4_results(
    r_model,
    j: int,
    dv: str,
    lag_cols: list[str],
    model_data: pd.DataFrame,
    subject: str,
    actual_re: str,
    warn_messages: list[str],
    ro,
) -> dict:
    """Extract fixed effects, random effects, residuals, p-values from lmer."""
    from rpy2.robjects.packages import importr
    stats_pkg = importr("stats")

    # Fixed effects
    fe_r = ro.r['fixef'](r_model)
    fe_names = list(fe_r.names)
    fe_vals = dict(zip(fe_names, fe_r))
    fixed = np.array([fe_vals.get(lc, 0.0) for lc in lag_cols])

    # P-values from lmerTest summary (Satterthwaite)
    summary_r = ro.r['summary'](r_model)
    coef_table = ro.r['coef'](summary_r)
    coef_rownames = list(ro.r['rownames'](coef_table))
    # coef_table columns: Estimate, Std. Error, df, t value, Pr(>|t|)
    # Last column (index 4) is the p-value
    n_rows = coef_table.nrow
    n_cols = coef_table.ncol
    pval_col = n_cols - 1  # last column

    pval_dict = {}
    for row_idx, name in enumerate(coef_rownames):
        pval_dict[name] = coef_table.rx(row_idx + 1, pval_col + 1)[0]

    pvals = np.array([pval_dict.get(lc, 1.0) for lc in lag_cols])

    # Random effects per subject
    fixed_intercept = fe_vals.get("(Intercept)", 0.0)
    ranef_r = ro.r['ranef'](r_model)
    # ranef_r is a list of data.frames, one per grouping factor
    # For our models, there's one grouping factor (subject)
    re_df = ranef_r.rx2(subject)
    re_colnames = list(re_df.colnames)
    re_rownames = list(re_df.rownames)

    subject_coefs = {}
    subject_intercepts = {}

    for subj_id in model_data[subject].unique():
        subj_str = str(subj_id)
        subj_fixed = fixed.copy()
        re_intercept = 0.0

        if subj_str in re_rownames:
            row_idx = re_rownames.index(subj_str)
            # Add random slopes to fixed effects
            if actual_re != "fixed":
                for k, lc in enumerate(lag_cols):
                    if lc in re_colnames:
                        col_idx = re_colnames.index(lc)
                        subj_fixed[k] += re_df.rx(row_idx + 1, col_idx + 1)[0]
            # Extract random intercept
            if "(Intercept)" in re_colnames:
                icpt_idx = re_colnames.index("(Intercept)")
                re_intercept = re_df.rx(row_idx + 1, icpt_idx + 1)[0]

        subject_coefs[subj_id] = subj_fixed
        subject_intercepts[subj_id] = fixed_intercept + re_intercept

    # Residuals
    resid_r = stats_pkg.residuals(r_model)
    residuals = pd.Series(np.array(resid_r), index=model_data.index)

    # BIC and AIC
    bic = float(ro.r['BIC'](r_model)[0])
    aic = float(ro.r['AIC'](r_model)[0])

    return {
        "j": j,
        "fixed": fixed,
        "pvals": pvals,
        "subject_coefs": subject_coefs,
        "subject_intercepts": subject_intercepts,
        "residuals": residuals,
        "bic": bic,
        "aic": aic,
        "warnings": warn_messages,
        "actual_re": actual_re,
    }


# ---------------------------------------------------------------------------
# RE fallback chain (same as statsmodels path)
# ---------------------------------------------------------------------------

_RE_FALLBACK_CHAIN = ("correlated", "orthogonal", "fixed")


# ---------------------------------------------------------------------------
# Main per-DV fitting function
# ---------------------------------------------------------------------------

def _fit_one_dv_lme4(
    j: int,
    var_cols: list[str],
    lag_data: pd.DataFrame,
    subject: str,
    temporal_re: str = "correlated",
) -> dict:
    """Fit a single mixed-effects model for DV var_cols[j] using R's lme4.

    Uses the same interface and return format as ``_fit_one_dv`` in
    ``_temporal.py`` so results can be assembled identically.
    """
    dv = var_cols[j]
    lag_cols = [f"{c}_lag" for c in var_cols]

    # Per-model listwise deletion (matching statsmodels path)
    needed_cols = [dv] + lag_cols
    model_data = lag_data.dropna(subset=needed_cols).copy()

    # Ensure subject column is string for R
    model_data[subject] = model_data[subject].astype(str)

    ro, pandas2ri, lmerTest_pkg, base_pkg = _setup_r_env()

    # Convert to R data.frame
    from rpy2.robjects import pandas2ri as p2r
    data_r = p2r.py2rpy(model_data)

    # Validate temporal_re
    if temporal_re not in _RE_FALLBACK_CHAIN:
        raise ValueError(
            f"temporal must be 'correlated', 'orthogonal', or 'fixed', "
            f"got {temporal_re!r}"
        )

    start_idx = _RE_FALLBACK_CHAIN.index(temporal_re)
    re_chain = _RE_FALLBACK_CHAIN[start_idx:]

    result = None
    warn_messages: list[str] = []
    actual_re = temporal_re

    for re_idx, re_structure in enumerate(re_chain):
        formula_str = _build_lmer_formula(dv, lag_cols, subject, re_structure)

        try:
            r_model, r_warns = _fit_lmer_with_warnings(
                formula_str, data_r, ro, lmerTest_pkg,
            )
            warn_messages = [str(w) for w in r_warns]
            actual_re = re_structure

            if not _has_severe_r_warnings(warn_messages):
                result = _extract_lme4_results(
                    r_model, j, dv, lag_cols, model_data, subject,
                    actual_re, warn_messages, ro,
                )
                return result

            # Severe warnings — try simpler RE if available
            if re_idx < len(re_chain) - 1:
                _warnings.warn(
                    f"mlVAR DV={dv!r}: R lme4 convergence warnings with "
                    f"temporal={re_structure!r}, falling back to simpler "
                    f"random-effects structure.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Last RE structure — accept with warnings
            result = _extract_lme4_results(
                r_model, j, dv, lag_cols, model_data, subject,
                actual_re, warn_messages, ro,
            )
            return result

        except RuntimeError:
            # R fit error — try simpler RE
            if re_structure == re_chain[-1]:
                raise
            _warnings.warn(
                f"mlVAR DV={dv!r}: R lme4 fit failed with "
                f"temporal={re_structure!r}, falling back to simpler "
                f"random-effects structure.",
                UserWarning,
                stacklevel=2,
            )
            continue

    raise RuntimeError(f"All RE structures failed for DV={dv!r} (lme4)")
