"""Model-B loss vs time for one typical bus/truck and one pedestrian event (paper/figs/fig_onset.pdf).

Restyled version of results/M3/onset/onset_events.pdf. The events, the
selection rule and the handover times come from
results/M3/onset/onset_events.json (written by scripts/run_m3_genie.py);
the loss curves from the m3_timeline cache of that job. Selection rule:
per class, the evaluation-seed 10 dB event whose duration is closest to
the class median (ties: lowest seed, mount, density, UE, start).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import COLORS, COLUMN_IN, save, setup  # noqa: E402

DT = 0.01
CAP_DB = 40.0
# handover markers: (color, line style, width, alpha, z-order); genie drawn wide and light under A3
MARKS = {
    "A3 @ 3GPP ref.": (COLORS["a3"], "-", 0.9, 1.0, 4),
    "A3 @ 10 dB": (COLORS["a3"], "--", 0.9, 1.0, 4),
    "genie+A3 @ 3GPP ref.": (COLORS["genie"], "-", 2.2, 0.5, 3),
    "genie+A3 @ 10 dB": (COLORS["genie"], "--", 2.2, 0.5, 3),
}


def main() -> None:
    plt = setup()
    payload = json.loads((ROOT / "results" / "M3" / "onset" / "onset_events.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.4))
    for ax, cls, tag in zip(axes, ("bus/truck", "pedestrian"), ("(a)", "(b)")):
        chosen = payload["chosen"][cls]
        ev = chosen["event"]
        rec = payload["events"][cls]
        seed, mount, density = chosen["job"]
        with np.load(ROOT / "results" / "cache" / mount / density / f"seed_{seed}" / "m3_timeline.npz") as f:
            loss = f["los_loss_db"]
        u, cell = int(ev["ue"]), int(ev["cell"])
        lo = max(0, ev["start_k"] - 300)
        hi = min(loss.shape[0] - 1, ev["end_k"] + 200)
        t = np.arange(lo, hi + 1) * DT - ev["start_s"]
        own = np.clip(np.nan_to_num(loss[lo : hi + 1, u, cell], posinf=CAP_DB), 0, CAP_DB)
        oth = np.clip(np.nan_to_num(loss[lo : hi + 1, u, 1 - cell], posinf=CAP_DB), 0, CAP_DB)
        ax.axvspan(0.0, ev["end_s"] - ev["start_s"], color="#fde0dd", lw=0, label="10 dB event")
        ax.plot(t, own, color="black", lw=1.1, label="Serving (blocked) cell")
        ax.plot(t, oth, color="gray", lw=0.9, ls="--", label="Other cell")
        ax.axhline(10.0, color="gray", lw=0.4, ls=":")
        for key, (color, ls, lw, alpha, z) in MARKS.items():
            for ho in rec["handovers"].get(key, []):
                ax.axvline(ho["switch_s"] - ev["start_s"], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
        ax.set_ylabel("LoS loss [dB]")
        ax.set_ylim(-1, CAP_DB if cls == "pedestrian" else 20)
        ax.grid(True, alpha=0.5)
        ax.text(0.01, 0.95, f"{tag} {cls}, 10–90 % onset {ev['onset_s']:.2f} s", transform=ax.transAxes, ha="left", va="top", fontsize=7, zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    axes[-1].set_xlabel("Time relative to event start [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    for key, (color, ls, lw, alpha, _) in MARKS.items():
        handles.append(plt.Line2D([], [], color=color, ls=ls, lw=lw, alpha=alpha))
        labels.append(key.replace("@", "HO,"))
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = save(fig, "fig_onset.pdf")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
