"""Network estimation methods."""

from __future__ import annotations

import pandas as pd

from ..network import Network
from ._registry import get_estimator, available_methods

# Import estimators to trigger registration
from . import cor  # noqa: F401
from . import pcor  # noqa: F401
from . import ebicglasso  # noqa: F401


def estimate_network(
    data: pd.DataFrame,
    *,
    method: str = "EBICglasso",
    **kwargs,
) -> Network:
    """Estimate a psychometric network from data.

    Parameters
    ----------
    data : pd.DataFrame
        Observations × variables data matrix.
    method : str
        Estimation method. Use ``available_methods()`` to list options.
    **kwargs
        Additional arguments passed to the estimator.

    Returns
    -------
    Network
    """
    estimator = get_estimator(method)
    return estimator.estimate(data, **kwargs)


__all__ = ["estimate_network", "get_estimator", "available_methods"]
