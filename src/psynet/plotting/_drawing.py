"""Shared network drawing helper for matplotlib axes."""

from __future__ import annotations

import networkx as nx
import numpy as np


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
