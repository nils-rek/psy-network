"""PsyNet — Psychometric network analysis toolkit for Python."""

from .network import Network
from .estimation_info import EstimationInfo
from .estimation import estimate_network, available_methods
from .centrality import centrality, strength, closeness, betweenness, expected_influence
from .community import communities, louvain, greedy_modularity, walktrap
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
    plot_ts_networks,
    plot_community,
    plot_mlvar_networks,
)
from .datasets import make_bfi25, make_depression9, make_multigroup, make_var_data, make_mlvar_data
from .group import estimate_group_network, GroupNetwork, GroupBootstrapResult, bootnet_group
from .timeseries import estimate_var_network, TSNetwork
from .mlvar import estimate_mlvar_network, MLVARNetwork

__version__ = "0.1.0"

__all__ = [
    # Core
    "Network",
    "EstimationInfo",
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
    "plot_community",
    # Community
    "communities",
    "louvain",
    "greedy_modularity",
    "walktrap",
    # Group
    "estimate_group_network",
    "GroupNetwork",
    "GroupBootstrapResult",
    "bootnet_group",
    # Time-series
    "estimate_var_network",
    "TSNetwork",
    "plot_ts_networks",
    # mlVAR
    "estimate_mlvar_network",
    "MLVARNetwork",
    "plot_mlvar_networks",
    # Datasets
    "make_bfi25",
    "make_depression9",
    "make_multigroup",
    "make_var_data",
    "make_mlvar_data",
]
