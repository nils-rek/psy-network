"""Shared network drawing helper for matplotlib axes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _compute_layout(network, layout: str = "spring", seed: int = 42) -> dict:
    """Compute node positions from a Network using a networkx layout algorithm.

    Parameters
    ----------
    network : Network
        Network to compute layout for (typically undirected for best results).
    layout : str
        Layout algorithm (``"spring"``, ``"circular"``, ``"kamada_kawai"``).
    seed : int
        Random seed for stochastic layouts.

    Returns
    -------
    dict
        Mapping of node labels to (x, y) positions.
    """
    G = network.to_networkx()
    layout_funcs = {
        "spring": lambda: nx.spring_layout(G, seed=seed, weight="weight"),
        "circular": lambda: nx.circular_layout(G),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G),
    }
    return layout_funcs.get(layout, layout_funcs["spring"])()


def _plot_network_panels(
    panels: list[tuple[str, object, bool]],
    *,
    layout: str = "spring",
    layout_network=None,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    suptitle: str = "",
    **kwargs,
) -> Figure:
    """Plot multiple network panels side by side with a shared layout.

    Parameters
    ----------
    panels : list of (title, Network, directed)
        Each tuple is (panel title, Network object, whether to draw directed).
    layout : str
        Layout algorithm.
    layout_network : Network, optional
        Network to compute layout from. If None, uses the first undirected
        network in panels, or the first network.
    figsize : tuple, optional
        Figure size. Defaults to (6*n_panels, 6).
    seed : int
        Random seed for layout.
    suptitle : str
        Figure super-title.
    **kwargs
        Additional drawing keyword arguments passed to ``_draw_network_on_ax``.

    Returns
    -------
    Figure
    """
    n = len(panels)
    if figsize is None:
        figsize = (6 * n, 6)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    # Determine layout source network
    if layout_network is None:
        # Prefer first undirected network
        for _, net, directed in panels:
            if not directed:
                layout_network = net
                break
        if layout_network is None:
            layout_network = panels[0][1]

    pos = _compute_layout(layout_network, layout, seed)

    for ax, (title, net, directed) in zip(axes, panels):
        _draw_network_on_ax(net, ax, pos, directed=directed, **kwargs)
        ax.set_title(f"{title} (n={net.n_observations})", fontsize=11)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
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
