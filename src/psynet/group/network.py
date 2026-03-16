"""GroupNetwork dataclass — wraps multiple Network objects from JGL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


@dataclass(frozen=True)
class GroupNetwork:
    """Immutable container for jointly estimated group networks.

    Parameters
    ----------
    networks : dict[str, Network]
        Mapping from group label to estimated Network.
    group_labels : list[str]
        Ordered group labels.
    lambda1 : float
        Sparsity penalty used.
    lambda2 : float
        Similarity penalty used.
    penalty : str
        Penalty type (``"fused"`` or ``"group"``).
    criterion : str
        Information criterion used for selection.
    """

    networks: dict[str, Network]
    group_labels: list[str]
    lambda1: float
    lambda2: float
    penalty: str
    criterion: str
    # Derived
    n_groups: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_groups", len(self.group_labels))

    def __getitem__(self, group: str) -> Network:
        return self.networks[group]

    def centrality(self) -> pd.DataFrame:
        """Compute centrality for all groups, with a 'group' column."""
        frames = []
        for label in self.group_labels:
            df = self.networks[label].centrality()
            df["group"] = label
            df["node"] = df.index
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def compare_edges(self) -> pd.DataFrame:
        """Long-form edge comparison across groups.

        Returns DataFrame with columns: group, node1, node2, weight.
        """
        frames = []
        for label in self.group_labels:
            net = self.networks[label]
            edges = net.edges_df.copy()
            edges["group"] = label
            frames.append(edges)

        if not frames:
            return pd.DataFrame(columns=["group", "node1", "node2", "weight"])

        return pd.concat(frames, ignore_index=True)

    def plot(self, **kwargs) -> Figure:
        """Plot all group networks side by side."""
        from ..plotting.group_plot import plot_group_networks
        return plot_group_networks(self, **kwargs)
