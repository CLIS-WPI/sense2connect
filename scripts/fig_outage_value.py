"""Full-width outage and value-of-foresight figure (paper/figs/fig_outage_value.pdf); figure-only merge.

(a) the content of scripts/fig_outage_margin.py (outage at the service rate
vs link margin, mean and 95 % CI over the 40 runs, log y, 3GPP reference
marked) and (b) the content of scripts/fig_value.py (share of the
A5-to-instantaneous-oracle gap: closable by class of the dominant LoS
blocker, not closable, genie-planner and sensing-planner markers, values
below the axis printed at the bottom edge, absolute gap above each bar).
Both panels use the same numeric margin axis and tick labels; the planners
have the same colors and markers in both panels. Data come from the same
functions as the two single figures (whose PDFs are kept). Exact size
7.0 x 2.5 in; ticks / labels 8.5 pt, legends 7.5 pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fig_outage_margin as FO  # noqa: E402
import fig_value as FV  # noqa: E402
from figstyle import save, setup  # noqa: E402

STYLE = {  # name in fig_outage_margin -> (color, marker, line style, legend label)
    "A3": ("#1f77b4", "o", "-", "A3"),
    "A5 (best reactive)": ("#2ca02c", "v", "-", "A5 (best reactive)"),
    "Sensing-planner": ("#d62728", "s", "--", "Sensing-planner"),
    "Genie-planner (H = 0.5 s)": ("#ff7f0e", "^", "-.", "Genie-planner"),
    "Cost-aware oracle": ("#7f7f7f", "D", ":", "Cost-aware oracle"),
    "Instantaneous oracle": ("#000000", "x", ":", "Inst. oracle"),
}
CLASS = (("ped", "#c5b0d5", "Closable: pedestrian"), ("bus", "#aec7e8", "Closable: bus/truck"), ("car", "#dbdb8d", "Closable: car"))


def main() -> None:
    plt = setup()
    plt.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "savefig.bbox": "standard"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.975, bottom=0.445, wspace=0.2)
    # ---- (a) outage vs margin
    data = FO.series()
    x = data["x"]
    n = len(data["series"])
    for k, (name, rows) in enumerate(data["series"].items()):
        mean = np.array([r["mean"] for r in rows])
        ci = np.array([r["ci"] for r in rows])
        color, marker, ls, lab = STYLE[name]
        dx = (k - (n - 1) / 2) * 0.25
        lo = np.maximum(mean - ci, 1e-3)
        ax.errorbar(x + dx, mean, yerr=[mean - lo, mean + ci - mean], color=color, marker=marker, ls=ls, capsize=1.2, elinewidth=0.5,
                    lw=0.9, markersize=3.2, markerfacecolor="white" if marker != "x" else color, label=lab)
    ref = data["ref_x"]
    ax.axvline(ref, color="gray", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("Outage [s/UE-min]")
    # ---- (b) share of the A5-to-oracle gap
    d = FV.data()
    xb = np.array(d["x"])
    w = 3.0
    bottom = np.zeros(len(xb))
    for key, color, lab in CLASS:
        v = np.array(d[key])
        bx.bar(xb, v, width=w, bottom=bottom, color=color, edgecolor="black", lw=0.3, label=lab)
        bottom = bottom + v
    bx.bar(xb, 1.0 - bottom, width=w, bottom=bottom, color="#f2f2f2", edgecolor="black", lw=0.3, label="Not closable")
    for xi, g in zip(xb, d["gap"]):
        bx.text(xi, 1.02, f"{g:.2f}", ha="center", va="bottom", fontsize=6.5)
    for key, name in (("genie", "Genie-planner (H = 0.5 s)"), ("sensing", "Sensing-planner")):
        color, marker, _ls, lab = STYLE[name]
        y = np.array(d[key])
        clipped = y < FV.YMIN
        bx.plot(xb[~clipped], y[~clipped], ls="none", marker=marker, color=color, markerfacecolor="white", markersize=4.2, label=lab)
        if clipped.any():
            yb = FV.YMIN + 0.07
            bx.plot(xb[clipped], np.full(clipped.sum(), yb), ls="none", marker=marker, color=color, markerfacecolor="white", markersize=4.2)
            for i in np.flatnonzero(clipped):
                bx.annotate(f"{y[i]:.1f}", (xb[i], yb), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=color,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3}, zorder=5)
    bx.axhline(0.0, color="black", lw=0.5)
    bx.axvline(ref, color="gray", lw=0.6, ls=":", zorder=0)
    bx.set_ylim(FV.YMIN, 1.13)
    bx.set_ylabel("Share of A5-to-oracle gap")
    # ---- shared x axis layout
    ticks = [0, 5, 10, 15, 20, 25, 30, ref]
    labels = ["0", "5", "10", "15", "20", "25", "30", f"{ref:.1f}\n(3GPP)"]
    for a, tag in ((ax, "(a)"), (bx, "(b)")):
        a.set_xticks(ticks)
        a.set_xticklabels(labels)
        a.set_xlim(-5.5, ref + 2.2)
        a.set_xlabel("Link margin above SNR$_\\mathrm{req}$ [dB]")
        a.grid(True, which="major", axis="y" if a is bx else "both", alpha=0.6)
        a.text(0.012, 0.975, tag, transform=a.transAxes, ha="left", va="top", fontsize=8, zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.47), ncol=3, frameon=False, fontsize=7.5, handlelength=2.0, columnspacing=0.9, handletextpad=0.4)
    h, lab = bx.get_legend_handles_labels()
    order = [lab.index(s) for s in ("Closable: pedestrian", "Closable: bus/truck", "Closable: car", "Not closable", "Genie-planner", "Sensing-planner")]
    bx.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.47), ncol=3, frameon=False, fontsize=7.5,
              handlelength=1.4, columnspacing=0.9, handletextpad=0.4)
    print(f"wrote {save(fig, 'fig_outage_value.pdf')}")


if __name__ == "__main__":
    main()
