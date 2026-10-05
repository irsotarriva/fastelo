"""Shared matplotlib style (validated reference palette from the dataviz skill)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ramp(n, lo=3):
    """n ordered colors from the blue ramp, skipping the lightest `lo` steps."""
    steps = BLUE_RAMP[lo:]
    if n == 1:
        return [steps[-1]]
    return [steps[round(i * (len(steps) - 1) / (n - 1))] for i in range(n)]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 5, "font.size": 10, "legend.frameon": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
