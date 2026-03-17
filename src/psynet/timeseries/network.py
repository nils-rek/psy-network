"""TSNetwork dataclass — wraps temporal and contemporaneous Network objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


@dataclass(frozen=True)
class TSNetwork:
    """Immutable container for a time-series (graphicalVAR) network.

    Parameters
    ----------
    temporal : Network
        Directed VAR(1) coefficient network (B matrix).
    contemporaneous : Network
        Undirected partial correlation network of VAR residuals.
    labels : list[str]
        Variable names.
    method : str
        Estimation method (``"graphicalVAR"``).
    n_observations : int
        Effective number of observations after lagging (T_eff).
    n_timepoints : int
        Original number of timepoints (T).
    """

    temporal: Network
    contemporaneous: Network
    labels: list[str]
    method: str
    n_observations: int
    n_timepoints: int
    # Derived
    n_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_nodes", len(self.labels))

    def centrality(self) -> pd.DataFrame:
        """Compute centrality for both temporal and contemporaneous networks.

        Returns
        -------
        pd.DataFrame
            Centrality measures with a ``'network'`` column indicating
            ``'temporal'`` or ``'contemporaneous'``.
        """
        frames = []
        for name, net in [("temporal", self.temporal),
                          ("contemporaneous", self.contemporaneous)]:
            df = net.centrality()
            df["network"] = name
            df["node"] = df.index
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def plot(self, **kwargs) -> Figure:
        """Plot temporal and contemporaneous networks side by side."""
        from ..plotting.ts_plot import plot_ts_networks
        return plot_ts_networks(self, **kwargs)
