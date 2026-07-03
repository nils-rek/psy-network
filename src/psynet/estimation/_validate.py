"""Input validation shared by the cross-sectional estimators."""

from __future__ import annotations

import warnings

import pandas as pd


def _validate_estimation_data(data: pd.DataFrame) -> int:
    """Validate a wide-format DataFrame for network estimation.

    Raises on non-numeric columns (``DataFrame.corr`` would silently drop
    them, producing an adjacency matrix that no longer matches the label
    list) and warns on missing values (correlations then use pairwise-
    complete observations).

    Returns
    -------
    int
        Effective sample size: the number of complete rows, used for
        ``n_observations`` and EBIC computations instead of ``len(data)``
        so missing data does not overstate n.
    """
    if data.shape[1] < 2:
        raise ValueError("Network estimation requires at least 2 variables")

    non_numeric = [
        col for col in data.columns
        if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if non_numeric:
        raise ValueError(
            f"All columns must be numeric; non-numeric columns: "
            f"{non_numeric}. Drop or encode them before estimation."
        )

    n_complete = int(data.notna().all(axis=1).sum())
    if n_complete < len(data):
        warnings.warn(
            f"{len(data) - n_complete} of {len(data)} rows contain missing "
            f"values; correlations use pairwise-complete observations and "
            f"the effective sample size is set to the number of complete "
            f"rows ({n_complete}).",
            UserWarning,
            stacklevel=3,
        )
    if n_complete < 3:
        raise ValueError(
            "Network estimation requires at least 3 complete observations"
        )
    return n_complete
