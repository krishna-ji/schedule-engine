"""
Centralized thesis-ready styling configuration for all plots.
Applies Seaborn-inspired theme with Times New Roman font.
"""

# CRITICAL: Set non-interactive backend BEFORE any other matplotlib imports
# This prevents tkinter-related errors in CLI environments
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for file generation

import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn-inspired color palette
COLORS = {
    "blue": "#4C72B0",  # Primary blue
    "orange": "#DD8452",  # Orange
    "green": "#55A868",  # Green
    "red": "#C44E52",  # Red
    "purple": "#8172B3",  # Purple
    "brown": "#937860",  # Brown
    "pink": "#DA8BC3",  # Pink
    "gray": "#8C8C8C",  # Gray
    "yellow": "#CCB974",  # Yellow
    "cyan": "#64B5CD",  # Cyan
}

# Extended palette for multiple lines
PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
    "#4C72B0",
    "#DD8452",  # Repeat for more lines
]

# Line styles for distinguishability
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "X", "P", "h", "+", "x"]


def apply_thesis_style():
    """
    Apply thesis-ready styling to matplotlib globally.
    Uses Seaborn style with Times New Roman font.
    """
    # Set Seaborn style as base
    sns.set_style(
        "whitegrid",
        {
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "axes.edgecolor": ".15",
            "axes.linewidth": 1.25,
        },
    )
    sns.set_context(
        "paper",
        font_scale=1.4,
        rc={
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        },
    )

    # Configure Times New Roman font
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            # Figure settings
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # Grid settings
            "grid.color": "#CCCCCC",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.5,
            # Axes settings
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.labelcolor": "#333333",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Tick settings
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend settings
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": True,
            "legend.shadow": False,
            # Line settings
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "lines.markeredgewidth": 0.5,
            # PDF settings
            "pdf.fonttype": 42,  # TrueType fonts (editable in PDF)
            "ps.fonttype": 42,
        }
    )

    # Set the color cycle to Seaborn palette
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", PALETTE)


def get_color(name):
    """Get a color from the thesis palette by name."""
    return COLORS.get(name, COLORS["blue"])


def get_constraint_colors():
    """
    Get distinct colors for constraint types.
    Returns dict with 'hard' and 'soft' keys.
    """
    return {
        "hard": COLORS["red"],
        "soft": COLORS["green"],
    }


def save_figure(fig, filepath, **kwargs):
    """
    Save figure with thesis-ready settings.

    Args:
        fig: matplotlib figure object
        filepath: path to save the figure
        **kwargs: additional arguments for savefig
    """
    default_kwargs = {
        "dpi": 300,
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
        "format": "pdf",
    }
    default_kwargs.update(kwargs)
    fig.savefig(filepath, **default_kwargs)
    plt.close(fig)


def create_thesis_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create a new figure with thesis-ready styling.

    Args:
        nrows: number of subplot rows
        ncols: number of subplot columns
        figsize: tuple of (width, height) in inches
        **kwargs: additional arguments for plt.subplots

    Returns:
        fig, ax (or axes array if multiple subplots)
    """
    if figsize is None:
        # Default sizes based on subplot layout
        if nrows == 1 and ncols == 1:
            figsize = (8, 5)
        elif nrows == 2 and ncols == 2:
            figsize = (12, 10)
        else:
            figsize = (10, 6 * nrows)

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, ax


def format_axis(ax, xlabel=None, ylabel=None, title=None, legend=True):
    """
    Apply consistent formatting to an axis.

    Args:
        ax: matplotlib axis object
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title
        legend: whether to show legend
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="normal")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="normal")
    if title:
        ax.set_title(title, fontweight="bold", pad=15)

    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(framealpha=0.95, edgecolor="#CCCCCC")

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Initialize styling when module is imported
apply_thesis_style()
