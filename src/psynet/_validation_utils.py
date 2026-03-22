"""Shared validation helpers for time-series and multilevel VAR estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_var_columns(
    data: pd.DataFrame,
    exclude_cols: set[str],
) -> list[str]:
    """Filter columns, check >=2 vars, check numeric, check NaN.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    exclude_cols : set[str]
        Column names to exclude (e.g. subject, beep, day).

    Returns
    -------
    list[str]
        Variable column names.

    Raises
    ------
    ValueError
        If fewer than 2 variables, non-numeric columns, or NaN values.
    """
    var_cols = [c for c in data.columns if c not in exclude_cols]

    if len(var_cols) < 2:
        raise ValueError(
            f"Need at least 2 variable columns, got {len(var_cols)}"
        )

    var_data = data[var_cols]
    if not var_data.apply(pd.api.types.is_numeric_dtype).all():
        raise ValueError("All variable columns must be numeric")

    if var_data.isna().any().any():
        raise ValueError("Data contains NaN values; remove or impute before estimation")

    return var_cols


def _find_valid_lag_indices(
    n_rows: int,
    beep_vals: np.ndarray | None = None,
    day_vals: np.ndarray | None = None,
) -> np.ndarray:
    """Return indices where (i, i+1) is a valid lag pair.

    Parameters
    ----------
    n_rows : int
        Number of rows in the data.
    beep_vals : ndarray, optional
        Beep/measurement index values.
    day_vals : ndarray, optional
        Day identifier values.

    Returns
    -------
    ndarray
        Array of indices i such that (i, i+1) is a valid consecutive pair.
    """
    if beep_vals is not None and day_vals is not None:
        valid = []
        for t in range(n_rows - 1):
            if day_vals[t] == day_vals[t + 1] and beep_vals[t + 1] - beep_vals[t] == 1:
                valid.append(t)
        return np.array(valid, dtype=int) if valid else np.array([], dtype=int)
    elif beep_vals is not None:
        valid = []
        for t in range(n_rows - 1):
            if beep_vals[t + 1] - beep_vals[t] == 1:
                valid.append(t)
        return np.array(valid, dtype=int) if valid else np.array([], dtype=int)
    else:
        return np.arange(n_rows - 1)
