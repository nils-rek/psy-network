"""Data validation and lag matrix construction for time-series networks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_ts_data(
    data: pd.DataFrame,
    beep: str | None = None,
    day: str | None = None,
) -> list[str]:
    """Validate time-series data and return variable column names.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data. Must have at least 2 rows and all-numeric
        variable columns (excluding beep/day).
    beep : str, optional
        Column name for beep/measurement index within a day.
    day : str, optional
        Column name for day identifier.

    Returns
    -------
    list[str]
        Variable column names (excluding beep/day).

    Raises
    ------
    ValueError
        If data has too few rows, contains NaN, or has non-numeric columns.
    """
    exclude = set()
    if beep is not None:
        if beep not in data.columns:
            raise ValueError(f"beep column {beep!r} not found in data")
        exclude.add(beep)
    if day is not None:
        if day not in data.columns:
            raise ValueError(f"day column {day!r} not found in data")
        exclude.add(day)

    var_cols = [c for c in data.columns if c not in exclude]

    if len(var_cols) < 2:
        raise ValueError(
            f"Need at least 2 variable columns for VAR estimation, got {len(var_cols)}"
        )

    if len(data) < 2:
        raise ValueError(
            f"Need at least 2 timepoints, got {len(data)}"
        )

    # Check numeric
    var_data = data[var_cols]
    if not var_data.apply(pd.api.types.is_numeric_dtype).all():
        raise ValueError("All variable columns must be numeric")

    # Check NaN
    if var_data.isna().any().any():
        raise ValueError("Data contains NaN values; remove or impute before estimation")

    return var_cols


def make_lag_matrix(
    data: pd.DataFrame,
    var_cols: list[str],
    beep: str | None = None,
    day: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct lag-1 matrices X (t-1) and Y (t) from time-series data.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data.
    var_cols : list[str]
        Variable column names.
    beep : str, optional
        Column for beep index. If provided with ``day``, only consecutive
        observations within the same day are included.
    day : str, optional
        Column for day identifier.

    Returns
    -------
    X : ndarray, shape (T_eff, p)
        Lagged predictors (observations at t-1).
    Y : ndarray, shape (T_eff, p)
        Outcomes (observations at t).
    """
    values = data[var_cols].values

    if beep is not None and day is not None:
        # Only include pairs where day is the same and beep is consecutive
        beep_vals = data[beep].values
        day_vals = data[day].values
        valid = []
        for t in range(len(data) - 1):
            if day_vals[t] == day_vals[t + 1] and beep_vals[t + 1] - beep_vals[t] == 1:
                valid.append(t)
        valid = np.array(valid)
        if len(valid) == 0:
            raise ValueError("No valid consecutive observation pairs found")
        X = values[valid]
        Y = values[valid + 1]
    elif beep is not None:
        # Use beep only: consecutive beeps
        beep_vals = data[beep].values
        valid = []
        for t in range(len(data) - 1):
            if beep_vals[t + 1] - beep_vals[t] == 1:
                valid.append(t)
        valid = np.array(valid)
        if len(valid) == 0:
            raise ValueError("No valid consecutive observation pairs found")
        X = values[valid]
        Y = values[valid + 1]
    else:
        # Simple: all consecutive pairs
        X = values[:-1]
        Y = values[1:]

    return X, Y
