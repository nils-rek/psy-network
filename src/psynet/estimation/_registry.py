"""Estimator registry — ``@register`` decorator and factory lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import NetworkEstimator

_REGISTRY: dict[str, type[NetworkEstimator]] = {}


def register(name: str):
    """Class decorator that registers an estimator under *name* (case-insensitive)."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        cls.name = name
        return cls
    return decorator


def get_estimator(name: str) -> NetworkEstimator:
    """Instantiate a registered estimator by name (case-insensitive)."""
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown estimation method {name!r}. Available: {available}"
        )
    return _REGISTRY[key]()


def available_methods() -> list[str]:
    """Return sorted list of registered method names."""
    return sorted(_REGISTRY)
