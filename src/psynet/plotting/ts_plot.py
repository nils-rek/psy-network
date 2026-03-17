"""Time-series network visualizations — temporal and contemporaneous side by side."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..timeseries.network import TSNetwork


def plot_ts_networks(
    ts_net: TSNetwork,
    *,
    layout: str = "spring",
    shared_layout: bool = True,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    **kwargs,
) -> Figure:
    """Plot temporal and contemporaneous networks side by side.

    Parameters
    ----------
    ts_net : TSNetwork
        Time-series network result.
    layout : str
        Layout algorithm (``"spring"``, ``"circular"``, ``"kamada_kawai"``).
    shared_layout : bool
        If True, use the same node positions for both panels.
    figsize : tuple, optional
        Figure size.
    seed : int
        Random seed for layout.
    **kwargs
        Additional drawing keyword arguments.

    Returns
    -------
    Figure
    """
    if figsize is None:
        figsize = (12, 6)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Compute layout from contemporaneous network (undirected, better for layout)
    G_layout = ts_net.contemporaneous.to_networkx()
    layout_funcs = {
        "spring": lambda: nx.spring_layout(G_layout, seed=seed, weight="weight"),
        "circular": lambda: nx.circular_layout(G_layout),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G_layout),
    }
    pos = layout_funcs.get(layout, layout_funcs["spring"])()

    # Draw temporal (directed)
    _draw_network_on_ax(ts_net.temporal, axes[0], pos, directed=True, **kwargs)
    axes[0].set_title(f"Temporal (n={ts_net.n_observations})", fontsize=11)

    # Draw contemporaneous (undirected)
    _draw_network_on_ax(ts_net.contemporaneous, axes[1], pos, directed=False, **kwargs)
    axes[1].set_title(f"Contemporaneous (n={ts_net.n_observations})", fontsize=11)

    fig.suptitle("Time-Series Network (graphicalVAR)", fontsize=13)
    fig.tight_layout()
    return fig


def _draw_network_on_ax(net, ax, pos, *, directed=False, **kwargs):
    """Draw a network on given axes with pre-computed positions."""
    node_color = kwargs.get("node_color", "#87CEEB")
    edge_color_pos = kwargs.get("edge_color_pos", "#2166AC")
    edge_color_neg = kwargs.get("edge_color_neg", "#B2182B")
    max_edge_width = kwargs.get("max_edge_width", 3.0)
    min_edge_width = kwargs.get("min_edge_width", 0.3)
    font_size = kwargs.get("font_size", 8)
    node_size = kwargs.get("node_size", 300)

    G = net.to_networkx()

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size, node_color=node_color,
        edgecolors="#333333", linewidths=1.0,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight="bold")

    edges = list(G.edges(data=True))
    if edges:
        weights = np.array([abs(d["weight"]) for _, _, d in edges])
        max_w = weights.max() if weights.max() > 0 else 1.0
        widths = min_edge_width + (weights / max_w) * (max_edge_width - min_edge_width)

        for (u, v, d), w in zip(edges, widths):
            color = edge_color_pos if d["weight"] >= 0 else edge_color_neg
            style = "solid" if d["weight"] >= 0 else "dashed"
            edge_kwargs = dict(
                edgelist=[(u, v)], ax=ax,
                width=float(w), edge_color=color, style=style, alpha=0.7,
            )
            if directed:
                edge_kwargs.update(
                    arrows=True,
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.1",
                )
            nx.draw_networkx_edges(G, pos, **edge_kwargs)

    ax.axis("off")
