"""PsyNet — Psychometric network analysis toolkit for Python."""

from .network import Network
from .estimation import estimate_network, available_methods
from .centrality import centrality, strength, closeness, betweenness, expected_influence
from .bootstrap import bootnet, BootstrapResult
from .plotting import (
    plot_network,
    plot_centrality,
    plot_edge_accuracy,
    plot_centrality_stability,
    plot_difference,
    plot_group_networks,
    plot_group_edge_accuracy,
    plot_group_centrality_comparison,
)
from .datasets import make_bfi25, make_depression9, make_multigroup
from .group import estimate_group_network, GroupNetwork, GroupBootstrapResult, bootnet_group

__version__ = "0.1.0"

__all__ = [
    # Core
    "Network",
    "estimate_network",
    "available_methods",
    # Centrality
    "centrality",
    "strength",
    "closeness",
    "betweenness",
    "expected_influence",
    # Bootstrap
    "bootnet",
    "BootstrapResult",
    # Plotting
    "plot_network",
    "plot_centrality",
    "plot_edge_accuracy",
    "plot_centrality_stability",
    "plot_difference",
    "plot_group_networks",
    "plot_group_edge_accuracy",
    "plot_group_centrality_comparison",
    # Group
    "estimate_group_network",
    "GroupNetwork",
    "GroupBootstrapResult",
    "bootnet_group",
    # Datasets
    "make_bfi25",
    "make_depression9",
    "make_multigroup",
]
