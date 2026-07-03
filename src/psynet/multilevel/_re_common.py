"""Shared random-effects fallback logic for the temporal backends.

Both the statsmodels backend (:mod:`._temporal`) and the lme4 backend
(:mod:`._lme4_backend`) walk the same fallback chain and downgrade the
random-effects structure on severe convergence warnings; only the
warning vocabularies differ between statsmodels and R.
"""

from __future__ import annotations

# RE structures from most complex to simplest
RE_FALLBACK_CHAIN = ("correlated", "orthogonal", "fixed")

# statsmodels warning fragments indicating severe convergence issues
STATSMODELS_SEVERE_PATTERNS = (
    "singular",
    "not positive definite",
    "optimization failed",
    "on the boundary",
)

# R/lme4 warning fragments indicating severe convergence issues
R_SEVERE_PATTERNS = (
    "failed to converge",
    "singular",
    "unable to evaluate",
    "boundary",
)


def has_severe_warnings(
    warn_messages: list[str], patterns: tuple[str, ...],
) -> bool:
    """Check whether any warning message contains a severe pattern."""
    for msg in warn_messages:
        msg_lower = msg.lower()
        if any(pat in msg_lower for pat in patterns):
            return True
    return False


def validate_temporal_re(temporal_re: str) -> tuple[str, ...]:
    """Validate an RE structure name and return its fallback chain."""
    if temporal_re not in RE_FALLBACK_CHAIN:
        raise ValueError(
            f"temporal must be 'correlated', 'orthogonal', or 'fixed', "
            f"got {temporal_re!r}"
        )
    start_idx = RE_FALLBACK_CHAIN.index(temporal_re)
    return RE_FALLBACK_CHAIN[start_idx:]
