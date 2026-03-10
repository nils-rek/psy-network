"""Enums, type aliases, and shared types for psynet."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

# Core type alias
AdjacencyMatrix = NDArray[np.floating]


class CorMethod(str, Enum):
    """Correlation method."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"


class BootstrapType(str, Enum):
    """Bootstrap type."""
    NONPARAMETRIC = "nonparametric"
    CASE = "case"


class Statistic(str, Enum):
    """Network statistic to track across bootstraps."""
    EDGE = "edge"
    STRENGTH = "strength"
    CLOSENESS = "closeness"
    BETWEENNESS = "betweenness"
    EXPECTED_INFLUENCE = "expectedInfluence"
