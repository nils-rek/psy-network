"""Data validation and lag construction for multilevel VAR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._validation_utils import _find_valid_lag_indices, _validate_var_columns


def validate_mlvar_data(
    data: pd.DataFrame,
    subject: str,
    *,
    beep: str | None = None,
    day: str | None = None,
) -> list[str]:
    """Validate multilevel VAR data and return variable column names.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format ESM data with a subject identifier column.
    subject : str
        Name of the subject identifier column.
    beep : str, optional
        Column for beep/measurement index within a day.
    day : str, optional
        Column for day identifier.

    Returns
    -------
    list[str]
        Variable column names (excluding subject/beep/day).

    Raises
    ------
    ValueError
        If validation fails.
    """
    if subject not in data.columns:
        raise ValueError(f"subject column {subject!r} not found in data")

    exclude = {subject}
    if beep is not None:
        if beep not in data.columns:
            raise ValueError(f"beep column {beep!r} not found in data")
        exclude.add(beep)
    if day is not None:
        if day not in data.columns:
            raise ValueError(f"day column {day!r} not found in data")
        exclude.add(day)

    var_cols = _validate_var_columns(data, exclude)

    unique_subjects = data[subject].unique()
    if len(unique_subjects) < 2:
        raise ValueError(
            f"Need at least 2 subjects, got {len(unique_subjects)}"
        )

    # Check minimum observations per subject
    obs_per_subject = data.groupby(subject).size()
    too_few = obs_per_subject[obs_per_subject < 3]
    if len(too_few) > 0:
        raise ValueError(
            f"Subjects with fewer than 3 observations: {list(too_few.index)}"
        )

    return var_cols


def make_mlvar_lag_data(
    data: pd.DataFrame,
    var_cols: list[str],
    subject: str,
    *,
    beep: str | None = None,
    day: str | None = None,
) -> pd.DataFrame:
    """Construct lagged long-format DataFrame respecting subject/day boundaries.

    For each subject, creates lag-1 pairs where only consecutive observations
    within the same subject (and same day, if provided) are included.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format ESM data.
    var_cols : list[str]
        Variable column names.
    subject : str
        Subject identifier column.
    beep : str, optional
        Beep/measurement index column.
    day : str, optional
        Day identifier column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``subject``, ``{var}`` (outcome at t),
        ``{var}_lag`` (predictor at t-1) for each variable.
    """
    lag_frames = []

    for subj_id, subj_data in data.groupby(subject):
        subj_data = subj_data.reset_index(drop=True)
        values = subj_data[var_cols].values

        beep_vals = subj_data[beep].values if beep is not None else None
        day_vals = subj_data[day].values if day is not None else None

        valid = _find_valid_lag_indices(len(subj_data), beep_vals, day_vals)

        if len(valid) == 0:
            continue

        X = values[valid]       # t-1
        Y = values[valid + 1]   # t

        lag_dict = {subject: [subj_id] * len(valid)}
        for i, col in enumerate(var_cols):
            lag_dict[col] = Y[:, i]
            lag_dict[f"{col}_lag"] = X[:, i]

        lag_frames.append(pd.DataFrame(lag_dict))

    if not lag_frames:
        raise ValueError("No valid consecutive observation pairs found")

    return pd.concat(lag_frames, ignore_index=True)
