"""Network visualization — spring-layout graph plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


def plot_network(
    net: Network,
    *,
    layout: str = "spring",
    node_size: float | str = 300,
    node_color: str = "#87CEEB",
    edge_color_pos: str = "#2166AC",
    edge_color_neg: str = "#B2182B",
    max_edge_width: float = 3.0,
    min_edge_width: float = 0.3,
    font_size: int = 8,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 8),
    seed: int = 42,
) -> Figure:
    """Plot a psychometric network.

    Parameters
    ----------
    net : Network
        The network to plot.
    layout : str
        Layout algorithm: ``"spring"``, ``"circular"``, ``"kamada_kawai"``.
    node_size : float or str
        Fixed size or a centrality measure name (e.g. ``"strength"``).
    node_color : str
        Node fill color.
    edge_color_pos, edge_color_neg : str
        Colors for positive / negative edges.
    max_edge_width, min_edge_width : float
        Edge width range.
    font_size : int
        Node label font size.
    title : str | None
        Plot title.
    ax : Axes | None
        Matplotlib axes to draw on. If None, a new figure is created.
    figsize : tuple
        Figure size (only used if ax is None).
    seed : int
        Random seed for spring layout.

    Returns
    -------
    Figure
    """
    G = net.to_networkx()

    # Layout
    layout_funcs = {
        "spring": lambda: nx.spring_layout(G, seed=seed, weight="weight"),
        "circular": lambda: nx.circular_layout(G),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(
            G, dist={n: {m: 1.0 / abs(G[n][m]["weight"]) if abs(G[n][m]["weight"]) > 0 else 10.0
                         for m in G[n]} for n in G}
        ),
    }
    pos = layout_funcs.get(layout, layout_funcs["spring"])()

    # Figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Node sizes
    if isinstance(node_size, str):
        from ..centrality import centrality
        cent = centrality(net)
        if node_size in cent.columns:
            vals = cent[node_size].values
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
            sizes = 200 + vals * 800
        else:
            sizes = 300
    else:
        sizes = node_size

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_color,
                           edgecolors="#333333", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight="bold")

    # Draw edges
    edges = G.edges(data=True)
    if edges:
        weights = np.array([abs(d["weight"]) for _, _, d in edges])
        max_w = weights.max() if weights.max() > 0 else 1.0
        widths = min_edge_width + (weights / max_w) * (max_edge_width - min_edge_width)

        for (u, v, d), w in zip(edges, widths):
            color = edge_color_pos if d["weight"] >= 0 else edge_color_neg
            style = "solid" if d["weight"] >= 0 else "dashed"
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                width=float(w), edge_color=color, style=style, alpha=0.7,
            )

    ax.set_title(title or f"Network ({net.method})", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig
