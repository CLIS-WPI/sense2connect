"""Accuracy break-even (paper/figs/fig_error_sweep.pdf), added after the second external review.

Paired outage difference planner - A5 [s/UE-min] vs the error sigma
(standard deviation per horizontal axis; for the realistic blocker model the
along-line axis), at 20 dB and at the 3GPP reference. Realistic error model
(time-correlated, along the lane / sidewalk): blocker position only,
blocker velocity only (solid: UE position exact; dashed: UE error 1 m), and
UE position only (perfect tracks). sigma = 0 is perfect tracks. Below zero
the planner beats A5.
After the third external review (figure-only change): the statistical unit
is the seed -- points are the mean over the 10 seeds of the per-seed mean
difference, bands the 95 % cluster-bootstrap CI over seeds (10,000
resamples), and the UE and blocker-position curves include the dense
points near the crossings (scripts/review3_eval.py); all from
results/M5/review3/seedlevel.json.
Fourth review (figure-only, no number changes): larger text for two-column
print -- axis and tick labels 8.5 pt, panel labels 8 pt, legend 7.5 pt in
two columns; saved at exactly 3.5 x 3.6 in (no tight cropping) so it prints
1:1 at the column width.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import COLUMN_IN, save, setup  # noqa: E402

PANELS = (("20 dB", "(a) 20 dB margin"), ("3GPP short-range reference", "(b) 3GPP reference"))


def _name(s: float) -> str:
    t = f"{s:g}"
    return "1.0" if t == "1" else t


def curve(comp: dict, sigmas: list[float], name_of, base_name: str, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = [0.0] + sorted(set(sigmas))
    names = [base_name] + [name_of(_name(s)) for s in xs[1:]]
    pts = [comp[f"sweep: {n} - A5"][label]["seed"] for n in names]
    return np.array(xs), np.array([p["mean_diff"] for p in pts]), np.array([p["ci95_boot"][0] for p in pts]), np.array([p["ci95_boot"][1] for p in pts])


def main() -> None:
    plt = setup()
    plt.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "savefig.bbox": "standard"})
    sw = json.loads((ROOT / "results" / "M5" / "review2" / "sweeps.json").read_text())
    ev = json.loads((ROOT / "results" / "M5" / "review3" / "eval.json").read_text())
    comp = json.loads((ROOT / "results" / "M5" / "review3" / "seedlevel.json").read_text())["comparisons"]
    sig = [float(x) for x in sw["sigmas"]]
    dense = {"pos": sig + [float(x) for x in ev["pos_sigmas"]], "ue": sig + [float(x) for x in ev["ue_sigmas"]], "vel": sig}
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.6), sharex=True)
    fig.subplots_adjust(left=0.165, right=0.975, top=0.985, bottom=0.285, hspace=0.09)
    series = [
        ("Blocker pos. [m]", "#1f77b4", "-", lambda s: f"R pos {s} | ue 0.0", "perfect | ue 0.0", "pos"),
        ("Blocker vel. [m/s]", "#d62728", "-", lambda s: f"R vel {s} | ue 0.0", "perfect | ue 0.0", "vel"),
        ("Blocker pos., UE err. 1 m", "#1f77b4", "--", lambda s: f"R pos {s} | ue 1.0", "perfect | ue 1.0", "vel"),
        ("Blocker vel., UE err. 1 m", "#d62728", "--", lambda s: f"R vel {s} | ue 1.0", "perfect | ue 1.0", "vel"),
        ("UE pos. [m]", "#555555", ":", lambda s: f"perfect | ue {s}", "perfect | ue 0.0", "ue"),
    ]
    for ax, (label, title) in zip(axes, PANELS):
        for name, color, ls, fn, base, grid in series:
            x, m, lo, hi = curve(comp, dense[grid], fn, base, label)
            ax.plot(x, m, color=color, ls=ls, marker="o", markersize=2.5, label=name)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.grid(True, alpha=0.5)
        ax.text(0.015, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    fig.supylabel("Planner $-$ A5 [s/UE-min]", fontsize=8.5, x=0.01, y=0.65)
    axes[-1].set_xlabel(r"Error standard deviation $\sigma$ (m or m/s)")
    axes[-1].set_xticks([0.0] + sig)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, 0.0),
               handlelength=2.2, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2)
    path = save(fig, "fig_error_sweep.pdf")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
