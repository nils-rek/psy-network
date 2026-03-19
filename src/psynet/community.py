"""Community detection for psychometric networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from .network import Network


def _prepare_graph(net: Network, absolute_weights: bool) -> nx.Graph:
    """Convert a Network to a NetworkX Graph for community detection.

    Optionally takes absolute values of weights and removes zero edges.
    """
    G = net.to_networkx()
    if absolute_weights:
        for u, v, d in G.edges(data=True):
            d["weight"] = abs(d["weight"])
    # Remove zero-weight edges
    zeros = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0]
    G.remove_edges_from(zeros)
    return G


def _renumber_communities(comm: pd.Series) -> pd.Series:
    """Renumber community IDs so community 0 contains the lexicographically first node."""
    if len(comm) == 0:
        return comm
    # Find the community of the lexicographically first node
    first_node = sorted(comm.index)[0]
    first_comm = comm[first_node]

    # Build mapping: first_comm -> 0, then remaining in order of first appearance
    old_ids = comm.unique()
    mapping = {}
    next_id = 0
    # Assign 0 to the community of the first node
    mapping[first_comm] = 0
    next_id = 1
    # Assign remaining IDs in order of first appearance (by sorted node name)
    for node in sorted(comm.index):
        cid = comm[node]
        if cid not in mapping:
            mapping[cid] = next_id
            next_id += 1
    return comm.map(mapping)


def louvain(net: Network, *, resolution: float = 1.0, seed: int | None = None,
            absolute_weights: bool = True) -> pd.Series:
    """Louvain community detection.

    Parameters
    ----------
    net : Network
        Estimated network.
    resolution : float
        Resolution parameter for modularity optimization.
    seed : int | None
        Random seed for reproducibility.
    absolute_weights : bool
        If True, use absolute edge weights (recommended for signed networks).

    Returns
    -------
    pd.Series
        Community assignments (index=node labels, values=community IDs).
    """
    G = _prepare_graph(net, absolute_weights)
    partition = nx.community.louvain_communities(G, resolution=resolution, seed=seed,
                                                  weight="weight")
    comm = pd.Series(index=net.labels, dtype=int, name="community")
    for cid, members in enumerate(partition):
        for node in members:
            comm[node] = cid
    return _renumber_communities(comm)


def greedy_modularity(net: Network, *, absolute_weights: bool = True) -> pd.Series:
    """Greedy modularity community detection.

    Parameters
    ----------
    net : Network
        Estimated network.
    absolute_weights : bool
        If True, use absolute edge weights (recommended for signed networks).

    Returns
    -------
    pd.Series
        Community assignments (index=node labels, values=community IDs).
    """
    G = _prepare_graph(net, absolute_weights)
    partition = nx.community.greedy_modularity_communities(G, weight="weight")
    comm = pd.Series(index=net.labels, dtype=int, name="community")
    for cid, members in enumerate(partition):
        for node in members:
            comm[node] = cid
    return _renumber_communities(comm)


def _walktrap_component(nodes: list[str], adj: np.ndarray, steps: int) -> dict[str, int]:
    """Run walktrap on a single connected component.

    Parameters
    ----------
    nodes : list of str
        Node labels in this component.
    adj : np.ndarray
        Adjacency submatrix for this component (n x n, non-negative weights).
    steps : int
        Number of random walk steps.

    Returns
    -------
    dict mapping node label to community ID (0-based within this component).
    """
    n = len(nodes)
    if n <= 1:
        return {nodes[i]: 0 for i in range(n)}

    # Degree vector
    deg = adj.sum(axis=1)
    # Avoid division by zero for isolated nodes within the component
    deg_safe = np.where(deg > 0, deg, 1.0)

    # Transition matrix P = D^{-1} A
    P = adj / deg_safe[:, np.newaxis]

    # P^t
    Pt = np.linalg.matrix_power(P, steps)

    # Ward-like distance matrix
    # d(i,j) = sqrt(sum_k ((Pt[i,k] - Pt[j,k])^2 / deg[k]))
    inv_deg = 1.0 / deg_safe
    diff_sq_weighted = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = Pt[i] - Pt[j]
            diff_sq_weighted[i, j] = np.sqrt(np.sum(diff ** 2 * inv_deg))
            diff_sq_weighted[j, i] = diff_sq_weighted[i, j]

    # Hierarchical clustering
    condensed = squareform(diff_sq_weighted)
    Z = linkage(condensed, method="ward")

    # Find the cut that maximizes modularity
    # Build a NetworkX graph for modularity computation
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                G.add_edge(nodes[i], nodes[j], weight=adj[i, j])

    best_mod = -np.inf
    best_labels = np.zeros(n, dtype=int)

    for k in range(1, n + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        # Build community sets
        comm_sets = {}
        for idx, lab in enumerate(labels):
            comm_sets.setdefault(lab, set()).add(nodes[idx])
        partition = list(comm_sets.values())
        try:
            mod = nx.community.modularity(G, partition, weight="weight")
        except nx.NetworkXError:
            mod = 0.0
        if mod >= best_mod:
            best_mod = mod
            best_labels = labels.copy()

    # Convert to 0-based community IDs
    unique_labels = sorted(set(best_labels))
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    return {nodes[i]: label_map[best_labels[i]] for i in range(n)}


def walktrap(net: Network, *, steps: int = 4, absolute_weights: bool = True) -> pd.Series:
    """Walktrap community detection using random walks.

    Parameters
    ----------
    net : Network
        Estimated network.
    steps : int
        Number of random walk steps (default 4).
    absolute_weights : bool
        If True, use absolute edge weights (recommended for signed networks).

    Returns
    -------
    pd.Series
        Community assignments (index=node labels, values=community IDs).
    """
    G = _prepare_graph(net, absolute_weights)
    labels = net.labels

    # Find connected components
    components = list(nx.connected_components(G))

    comm = pd.Series(index=labels, dtype=int, name="community")
    next_comm_id = 0

    for component in components:
        comp_nodes = sorted(component)
        if len(comp_nodes) == 1:
            # Isolated node gets its own community
            comm[comp_nodes[0]] = next_comm_id
            next_comm_id += 1
            continue

        # Build adjacency submatrix for this component
        node_idx = {node: i for i, node in enumerate(comp_nodes)}
        n_comp = len(comp_nodes)
        adj = np.zeros((n_comp, n_comp))
        for u, v, d in G.subgraph(comp_nodes).edges(data=True):
            i, j = node_idx[u], node_idx[v]
            adj[i, j] = d["weight"]
            adj[j, i] = d["weight"]

        comp_communities = _walktrap_component(comp_nodes, adj, steps)

        # Offset community IDs
        for node, cid in comp_communities.items():
            comm[node] = cid + next_comm_id
        max_cid = max(comp_communities.values()) if comp_communities else -1
        next_comm_id += max_cid + 1

    # Handle nodes not in any edge (truly isolated)
    for node in labels:
        if node not in G.nodes() or G.degree(node) == 0:
            if pd.isna(comm.get(node, np.nan)):
                comm[node] = next_comm_id
                next_comm_id += 1

    return _renumber_communities(comm)


def communities(net: Network, *, method: str = "walktrap", **kwargs) -> pd.Series:
    """Detect communities in a psychometric network.

    Parameters
    ----------
    net : Network
        Estimated network.
    method : str
        Algorithm: ``"walktrap"``, ``"louvain"``, or ``"greedy_modularity"``.
    **kwargs
        Passed to the chosen algorithm.

    Returns
    -------
    pd.Series
        Community assignments (index=node labels, values=community IDs).

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    methods = {
        "walktrap": walktrap,
        "louvain": louvain,
        "greedy_modularity": greedy_modularity,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown community detection method {method!r}. "
            f"Choose from: {', '.join(sorted(methods))}"
        )
    return methods[method](net, **kwargs)
