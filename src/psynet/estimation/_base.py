"""Protocol for network estimators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..network import Network


@runtime_checkable
class NetworkEstimator(Protocol):
    """Interface every estimator must satisfy."""

    name: str

    def estimate(self, data: pd.DataFrame, **kwargs) -> Network:
        """Estimate a network from *data* (observations × variables)."""
        ...
