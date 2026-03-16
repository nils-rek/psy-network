"""Shared utilities for the group subpackage."""

from __future__ import annotations

import pandas as pd


def parse_group_data(
    data: pd.DataFrame | list[pd.DataFrame],
    group_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Parse input data into a dict of per-group DataFrames.

    Parameters
    ----------
    data : pd.DataFrame or list[pd.DataFrame]
        Either a single DataFrame with a ``group_col`` column, or a list
        of DataFrames (one per group).
    group_col : str, optional
        Column name identifying groups (required if data is a single DataFrame).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from group label to numeric-only DataFrame (sorted by key).

    Raises
    ------
    ValueError
        If group_col is missing for DataFrame input, or if groups have
        different variable names.
    """
    if isinstance(data, list):
        group_dfs = {f"Group{i+1}": df.reset_index(drop=True) for i, df in enumerate(data)}
    else:
        if group_col is None:
            raise ValueError("group_col is required when data is a single DataFrame")
        group_dfs = {
            str(name): grp.drop(columns=[group_col]).reset_index(drop=True)
            for name, grp in data.groupby(group_col)
        }

    # Validate all groups have the same columns
    labels_list = [list(df.columns) for df in group_dfs.values()]
    if len(set(tuple(c) for c in labels_list)) > 1:
        raise ValueError(
            "All groups must have the same variable names. "
            f"Got: {dict(zip(group_dfs.keys(), labels_list))}"
        )

    return dict(sorted(group_dfs.items()))
