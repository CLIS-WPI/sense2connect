"""Shared matplotlib style for IEEE single-column paper figures (vector PDF)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figs"
COLUMN_IN = 3.5  # IEEE single-column width [in]

COLORS = {
    "oracle": "#000000",
    "a3": "#1f77b4",
    "hybrid": "#d62728",
    "genie": "#ff7f0e",
    "xapp": "#9467bd",
}


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
            "grid.linewidth": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    return path
