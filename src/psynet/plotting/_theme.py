"""Centralized visual theme constants for psynet plots.

All colors use the Okabe-Ito palette — the gold standard for
colorblind-safe scientific visualization.
"""

# ── Nodes ──────────────────────────────────────────────────────────
NODE_FILL_COLOR = "#56B4E9"      # sky blue (Okabe-Ito)
NODE_BORDER_COLOR = "#444444"    # dark gray
NODE_BORDER_WIDTH = 1.5
NODE_SIZE_DEFAULT = 600          # fits 1-2 digit numeric labels comfortably
NODE_SIZE_RANGE = (400, 1200)    # min/max for centrality-scaled sizing

# ── Edges ──────────────────────────────────────────────────────────
EDGE_COLOR_POS = "#0072B2"       # blue (Okabe-Ito)
EDGE_COLOR_NEG = "#D55E00"       # vermillion/orange (Okabe-Ito)
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 4.5
EDGE_ALPHA = 0.75
EDGE_STYLE_POS = "solid"
EDGE_STYLE_NEG = (0, (5, 3))    # dash pattern (offset, (on, off))

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
