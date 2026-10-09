# Shared plot settings for the figure scripts.
import matplotlib.pyplot as plt

# ColorBrewer "Blues", light to dark
B1 = "#bdd7e7"
B2 = "#8bbddb"
B3 = "#5a9bcf"
B4 = "#2b7bba"
B5 = "#08306b"

BAND = B2          # shaded band in Figure 5
MED = B4           # median line
NAVY = B5          # markers and main series
INK = "#1b1b1b"    # text
CHARCOAL = "#3a3a3a"

# Font sizes in points. Figures are drawn at the printed text width, so these are the
# sizes that appear in the paper.
FS_LABEL = 9.5
FS_TICK = 8.5
FS_LEGEND = 8.5
FS_ANNOT = 8.5

WIDTH = 6.5        # text width in inches (11pt article, 1in margins)

SOLID = (0, ())
PORT3 = [(B2, SOLID), (B4, SOLID), (B5, SOLID)]       # conservative, balanced, aggressive
HORIZON5 = [(B1, SOLID), (B2, SOLID), (B3, SOLID), (B4, SOLID), (B5, SOLID)]  # T = 10..50


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": FS_LABEL,
        "axes.linewidth": 0.6, "axes.edgecolor": CHARCOAL,
        "axes.labelsize": FS_LABEL, "axes.titlesize": FS_LABEL,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": CHARCOAL, "ytick.color": CHARCOAL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": False,
        "legend.frameon": False, "lines.solid_capstyle": "round",
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def ygrid(ax):
    ax.set_axisbelow(True)
