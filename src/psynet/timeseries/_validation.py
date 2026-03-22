"""Data validation and lag matrix construction for time-series networks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._validation_utils import _find_valid_lag_indices, _validate_var_columns


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

    var_cols = _validate_var_columns(data, exclude)

    if len(data) < 2:
        raise ValueError(
            f"Need at least 2 timepoints, got {len(data)}"
        )

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

    beep_vals = data[beep].values if beep is not None else None
    day_vals = data[day].values if day is not None else None

    valid = _find_valid_lag_indices(len(data), beep_vals, day_vals)

    if beep_vals is not None and len(valid) == 0:
        raise ValueError("No valid consecutive observation pairs found")

    X = values[valid]
    Y = values[valid + 1]

    return X, Y
