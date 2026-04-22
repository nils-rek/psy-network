"""Centralized visual theme constants for psynet plots.

All light-theme colors use the Okabe-Ito palette — the gold standard for
colorblind-safe scientific visualization.  The dark theme uses the PX palette
(cool near-black surfaces, three community accents, rose-red negative edges).

Usage
-----
    import psynet.plotting as p
    p.set_theme("dark")   # switch to PX dark palette
    p.set_theme("light")  # revert to Okabe-Ito defaults
"""

from __future__ import annotations

import sys
from typing import Literal

# ── Nodes ──────────────────────────────────────────────────────────
NODE_FILL_COLOR = "#56B4E9"      # sky blue (Okabe-Ito)
NODE_BORDER_COLOR = "#444444"    # dark gray
NODE_BORDER_WIDTH = 1.5
NODE_SIZE_DEFAULT = 900          # fits 1-2 digit numeric labels comfortably
NODE_SIZE_RANGE = (700, 1800)    # min/max for centrality-scaled sizing

# ── Edges ──────────────────────────────────────────────────────────
EDGE_COLOR_POS = "#0072B2"       # blue (Okabe-Ito)
EDGE_COLOR_NEG = "#D55E00"       # vermillion/orange (Okabe-Ito)
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 4.5
EDGE_ALPHA = 0.75
EDGE_ALPHA_MIN = 0.25
EDGE_ALPHA_MAX = 0.85
EDGE_STYLE_POS = "solid"
EDGE_STYLE_NEG = (0, (5, 3))    # dash pattern (offset, (on, off))

# ── Centrality aura ───────────────────────────────────────────────
AURA_N_SEGMENTS = 40            # arc segments for smooth curve
AURA_ARC_MAX_ANGLE = 340        # degrees at max centrality
AURA_ALPHA_START = 0.45         # opacity at arc start
AURA_WIDTH_FACTOR = 0.30        # arc half-width as fraction of node radius
AURA_GAP_FACTOR = 1.15          # inner edge distance from center (× node radius)

# ── Fonts ──────────────────────────────────────────────────────────
FONT_SIZE_NODE = 9
FONT_SIZE_LEGEND = 8.5
LEGEND_STEP = 0.04              # vertical spacing between legend entries (axes frac)
LEGEND_HEADER_FONT_SIZE = 9.5
LEGEND_SUBHEADING_FONT_SIZE = 9
LEGEND_TITLE_DEFAULT = "Legend"
FONT_WEIGHT_NODE = "bold"

# ── Figure defaults ────────────────────────────────────────────────
FIGSIZE_SINGLE = (10, 8)        # wider than (8,8) to fit legend panel
FIGSIZE_PANEL_WIDTH = 7         # per-panel width for multi-panel plots
FIGSIZE_PANEL_HEIGHT = 6
TITLE_FONT_SIZE = 12
PANEL_TITLE_FONT_SIZE = 11
SUPTITLE_FONT_SIZE = 13

# ── Surface / background ──────────────────────────────────────────
BACKGROUND_COLOR = "white"
SURFACE_COLOR = "white"
AXIS_COLOR = "#cccccc"

# ── Text ──────────────────────────────────────────────────────────
TEXT_PRIMARY = "#222222"
TEXT_SECONDARY = "#555555"
TEXT_TERTIARY = "#888888"

# ── Plot element colors ────────────────────────────────────────────
CI_BAR_COLOR = "#CCCCCC"
LOLLIPOP_STEM_COLOR = "#666666"
ZERO_LINE_COLOR = "black"
THRESHOLD_LINE_COLOR = "gray"
DIFFERENCE_CMAP = "gray"

# ── Accent palette (bootstrap, centrality, groups) ─────────────────
ACCENT_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # amber/orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

# ── Community palette ──────────────────────────────────────────────
COMMUNITY_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # amber
    "#009E73",  # green
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]


# ── Theme definitions ─────────────────────────────────────────────
# Keys that are swapped when the theme changes.  All other constants
# (geometry, fonts, alpha ranges) are theme-invariant.
_DARK_THEME: dict[str, object] = {
    "NODE_FILL_COLOR": "#6b9fd4",   # community-blue, default single-node fill
    "NODE_BORDER_COLOR": "#2c2f39", # Surface 600
    "EDGE_COLOR_POS": "#6b9fd4",    # blue (distinct positive hue)
    "EDGE_COLOR_NEG": "#c96b62",    # rose-red — oklch(0.65 0.13 10)
    "BACKGROUND_COLOR": "#0e0f14",  # oklch(0.10 0.01 260)
    "SURFACE_COLOR": "#151619",     # Surface 900
    "AXIS_COLOR": "#2c2f39",        # Surface 600
    "TEXT_PRIMARY": "#ebecef",      # oklch(0.95 0.005 260)
    "TEXT_SECONDARY": "#93969d",    # oklch(0.65 0.01 260)
    "TEXT_TERTIARY": "#6b6d74",     # oklch(0.45 0.01 260)
    "CI_BAR_COLOR": "#2c2f39",
    "LOLLIPOP_STEM_COLOR": "#6b6d74",
    "ZERO_LINE_COLOR": "#93969d",
    "THRESHOLD_LINE_COLOR": "#6b6d74",
    "DIFFERENCE_CMAP": "gray_r",    # inverted: 0=white (significant), 0.85=dark
    "ACCENT_COLORS": [
        "#5bb98b",  # green
        "#d4915e",  # orange
        "#6b9fd4",  # blue
        "#c96b62",  # rose-red
        "#8bcfb0",  # light green
        "#e8b89a",  # light orange
        "#a4c4e3",  # light blue
    ],
    "COMMUNITY_PALETTE": [
        "#5bb98b",  # community 0 — green  (oklch 0.68 0.14 155)
        "#d4915e",  # community 1 — orange (oklch 0.72 0.14  55)
        "#6b9fd4",  # community 2 — blue   (oklch 0.65 0.12 240)
        "#c96b62",  # community 3 — rose-red
        "#8bcfb0",  # community 4 — light green
        "#e8b89a",  # community 5 — light orange
        "#a4c4e3",  # community 6 — light blue
        "#e8a0a4",  # community 7 — light rose
    ],
}

# Captured at module-load (light defaults) so set_theme("light") can restore them.
_LIGHT_THEME: dict[str, object] = {k: globals()[k] for k in _DARK_THEME}

_current_theme: str = "light"


def set_theme(name: Literal["light", "dark"]) -> None:
    """Switch the psynet plotting theme globally.

    Call once before creating plots; the chosen palette applies to every
    subsequent ``plot_*`` call.

    Parameters
    ----------
    name : "light" or "dark"
        ``"light"`` restores the default Okabe-Ito palette on a white
        background. ``"dark"`` applies the PX palette on a near-black
        background (cool near-black surfaces, three community accents,
        rose-red negative edges).
    """
    if name not in ("light", "dark"):
        raise ValueError(f"Unknown theme {name!r}. Choose 'light' or 'dark'.")
    global _current_theme
    _current_theme = name
    palette = _LIGHT_THEME if name == "light" else _DARK_THEME
    module = sys.modules[__name__]
    for k, v in palette.items():
        setattr(module, k, v)


def get_theme() -> str:
    """Return the name of the currently active theme (``"light"`` or ``"dark"``)."""
    return _current_theme


def apply_theme_to_axes(ax) -> None:
    """Stamp the active theme's colours onto a matplotlib axes and its figure.

    Sets figure and axes facecolors, tick-label colours, and spine colours
    from the active theme constants.  Safe to call on both axes-on (centrality,
    bootstrap) and axes-off (network) axes.
    """
    fig = ax.get_figure()
    fig.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, which="both")
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
