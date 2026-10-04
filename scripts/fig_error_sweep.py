"""Accuracy break-even (paper/figs/fig_error_sweep.pdf), added after the second external review.

From results/M5/review2/sweeps.json (scripts/review2_sweeps.py): paired
outage difference planner - A5 [s/UE-min] (mean over jobs, t-based 95 % CI)
vs the error sigma, at 20 dB and at the 3GPP reference. Realistic error
model (time-correlated, along the lane / sidewalk): blocker position only,
blocker velocity only (solid: UE position exact; dashed: UE error 1 m), and
UE position only (perfect tracks). sigma = 0 is perfect tracks.
Below zero the planner beats A5.
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


def curve(cond: dict, sigmas: list[float], name_of, base_name: str, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, mean, half = [0.0], [], []
    names = [base_name] + [name_of(s) for s in sigmas]
    xs += list(sigmas)
    vals = []
    for n in names:
        m = {r["label"]: r for r in cond[n]["margins"]}[label]["vs_a5"]
        vals.append((m["mean_diff"], m["ci95_half_t"]))
    mean = np.array([v[0] for v in vals])
    half = np.array([v[1] for v in vals])
    return np.array(xs), mean, half


def main() -> None:
    plt = setup()
    sw = json.loads((ROOT / "results" / "M5" / "review2" / "sweeps.json").read_text())
    cond, sig = sw["conditions"], list(sw["sigmas"])
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.6), sharex=True)
    series = [
        ("Blocker position [m]", "#1f77b4", "-", lambda s: f"R pos {s} | ue 0.0", "perfect | ue 0.0"),
        ("Blocker velocity [m/s]", "#d62728", "-", lambda s: f"R vel {s} | ue 0.0", "perfect | ue 0.0"),
        ("Blocker position, UE error 1 m", "#1f77b4", "--", lambda s: f"R pos {s} | ue 1.0", "perfect | ue 1.0"),
        ("Blocker velocity, UE error 1 m", "#d62728", "--", lambda s: f"R vel {s} | ue 1.0", "perfect | ue 1.0"),
        ("UE position [m]", "#555555", ":", lambda s: f"perfect | ue {s}", "perfect | ue 0.0"),
    ]
    for ax, (label, title) in zip(axes, PANELS):
        for name, color, ls, fn, base in series:
            x, m, h = curve(cond, sig, fn, base, label)
            ax.plot(x, m, color=color, ls=ls, marker="o", label=name)
            ax.fill_between(x, m - h, m + h, color=color, alpha=0.12, lw=0)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_ylabel("Planner $-$ A5 [s/UE-min]")
        ax.grid(True, alpha=0.5)
        ax.text(0.01, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    axes[-1].set_xlabel(r"Error standard deviation $\sigma$ (m or m/s)")
    axes[-1].set_xticks([0.0] + sig)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    path = save(fig, "fig_error_sweep.pdf")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
