"""Centrality measures for psychometric networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .network import Network


def strength(net: Network) -> pd.Series:
    """Sum of absolute edge weights per node.

    For directed networks, returns total strength (in-strength + out-strength).
    For undirected networks (symmetric adjacency), this equals the row sum.
    """
    A = np.abs(net.adjacency)
    if net.directed:
        vals = np.sum(A, axis=0) + np.sum(A, axis=1)  # in + out
    else:
        vals = np.sum(A, axis=1)
    return pd.Series(vals, index=net.labels, name="strength")


def expected_influence(net: Network) -> pd.Series:
    """Sum of signed edge weights per node.

    For directed networks, returns total expected influence (in + out).
    For undirected networks (symmetric adjacency), this equals the row sum.
    """
    A = net.adjacency
    if net.directed:
        vals = np.sum(A, axis=0) + np.sum(A, axis=1)  # in + out
    else:
        vals = np.sum(A, axis=1)
    return pd.Series(vals, index=net.labels, name="expectedInfluence")


def closeness(net: Network, *, normalized: bool = True) -> pd.Series:
    """Closeness centrality via inverse absolute weight distances.

    Parameters
    ----------
    net : Network
        Estimated network.
    normalized : bool
        If ``True`` (default), closeness values are multiplied by ``n - 1``
        (NetworkX convention).  Set to ``False`` for unnormalized values
        matching R's ``centrality_auto()`` output.
    """
    G = net.to_networkx()
    # Set distance = 1/|weight| for shortest-path computation
    for u, v, d in G.edges(data=True):
        w = abs(d.get("weight", 1.0))
        d["distance"] = 1.0 / w if w > 0 else np.inf
    vals = nx.closeness_centrality(G, distance="distance")
    s = pd.Series(vals, name="closeness").reindex(net.labels)
    if not normalized:
        # NetworkX returns (n-1)/sum(d); unnormalized is 1/sum(d)
        n = net.n_nodes
        s = s / (n - 1) if n > 1 else s
    return s


def betweenness(net: Network, *, normalized: bool = True) -> pd.Series:
    """Betweenness centrality via inverse absolute weight distances.

    Parameters
    ----------
    net : Network
        Estimated network.
    normalized : bool
        If ``True`` (default), raw betweenness counts are scaled by
        ``2 / ((n-1)(n-2))`` for undirected graphs (NetworkX convention).
        Set to ``False`` for raw path counts matching R's
        ``centrality_auto()`` output.
    """
    G = net.to_networkx()
    for u, v, d in G.edges(data=True):
        w = abs(d.get("weight", 1.0))
        d["distance"] = 1.0 / w if w > 0 else np.inf
    vals = nx.betweenness_centrality(G, weight="distance", normalized=normalized)
    return pd.Series(vals, name="betweenness").reindex(net.labels)


def centrality(net: Network, *, normalized: bool = True) -> pd.DataFrame:
    """Compute all centrality measures; return a labeled DataFrame.

    Parameters
    ----------
    net : Network
        Estimated network.
    normalized : bool
        Passed to ``closeness()`` and ``betweenness()``.  See their
        docstrings for details.
    """
    return pd.DataFrame({
        "strength": strength(net),
        "closeness": closeness(net, normalized=normalized),
        "betweenness": betweenness(net, normalized=normalized),
        "expectedInfluence": expected_influence(net),
    })
