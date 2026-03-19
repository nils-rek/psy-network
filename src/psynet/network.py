"""Core Network dataclass — the primary result object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from ._types import AdjacencyMatrix
from .estimation_info import EstimationInfo

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class Network:
    """Immutable container for an estimated psychometric network.

    Parameters
    ----------
    adjacency : AdjacencyMatrix
        Square weighted adjacency matrix (p × p).
    labels : list[str]
        Node labels, length p.
    method : str
        Estimation method used (e.g. ``"EBICglasso"``).
    n_observations : int
        Number of observations in the data used for estimation.
    weighted : bool
        Whether edges carry continuous weights.
    signed : bool
        Whether negative edges are allowed.
    directed : bool
        Whether the network is directed.
    estimation_info : EstimationInfo | None
        Estimation diagnostics and metadata (lambda, EBIC curve, etc.).
    """

    adjacency: AdjacencyMatrix
    labels: list[str]
    method: str
    n_observations: int
    weighted: bool = True
    signed: bool = True
    directed: bool = False
    estimation_info: EstimationInfo | None = None
    # Derived (computed post-init)
    n_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_nodes", len(self.labels))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def adjacency_df(self) -> pd.DataFrame:
        """Labeled adjacency matrix as a DataFrame."""
        return pd.DataFrame(
            self.adjacency, index=self.labels, columns=self.labels,
        )

    @property
    def edges_df(self) -> pd.DataFrame:
        """Long-form edge list (node1, node2, weight) for unique edges."""
        rows: list[dict] = []
        p = self.n_nodes
        for i in range(p):
            start = 0 if self.directed else i + 1
            for j in range(start, p):
                if i == j:
                    continue
                w = self.adjacency[i, j]
                if w != 0:
                    rows.append({
                        "node1": self.labels[i],
                        "node2": self.labels[j],
                        "weight": w,
                    })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def to_networkx(self) -> nx.Graph:
        """Convert to a NetworkX graph."""
        cls = nx.DiGraph if self.directed else nx.Graph
        G = cls()
        G.add_nodes_from(self.labels)
        for _, row in self.edges_df.iterrows():
            G.add_edge(row["node1"], row["node2"], weight=row["weight"])
        return G

    def centrality(self) -> pd.DataFrame:
        """Compute centrality measures; delegates to :mod:`psynet.centrality`."""
        from .centrality import centrality
        return centrality(self)

    def communities(self, method: str = "walktrap", **kwargs) -> pd.Series:
        """Detect communities; delegates to :mod:`psynet.community`."""
        from .community import communities
        return communities(self, method=method, **kwargs)

    def plot_communities(self, method: str = "walktrap", **kwargs) -> Figure:
        """Plot network colored by community membership."""
        from .community import communities as detect
        from .plotting import plot_community
        comm = detect(self, method=method)
        return plot_community(self, comm, **kwargs)

    def plot(self, **kwargs) -> Figure:
        """Plot the network; delegates to :func:`psynet.plotting.plot_network`."""
        from .plotting import plot_network
        return plot_network(self, **kwargs)
