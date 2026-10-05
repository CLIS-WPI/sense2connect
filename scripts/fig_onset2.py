"""Onset figure for the right half of a two-column figure (paper/figs/fig_onset2.pdf); figure-only variant of fig_onset.py.

Same events, data and 1 ms loss evaluation as scripts/fig_onset.py (whose
fig_onset.pdf is kept): (b) bus/truck and (c) pedestrian 10 dB event from
results/M3/onset/onset_events.json, model-B LoS loss of the serving
(blocked) and the other cell at 1 ms resolution, the 10 dB event shaded,
and the handovers of A3 and genie+A3 at the 10 dB margin. Exact size
3.45 x 2.7 in; ticks / labels 8.5 pt, legend 7.5 pt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from fig_onset import CAP_DB, DT, loss_1ms  # noqa: E402
from figstyle import COLORS, save, setup  # noqa: E402

MARKS = {"A3 @ 10 dB": (COLORS["a3"], "--", 1.0, 1.0, 4, "A3 handover"),
         "genie+A3 @ 10 dB": (COLORS["genie"], "-", 2.2, 0.55, 3, "Genie+A3 handover")}


def main() -> None:
    plt = setup()
    plt.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "savefig.bbox": "standard"})
    payload = json.loads((ROOT / "results" / "M3" / "onset" / "onset_events.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.7), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.975, top=0.985, bottom=0.305, hspace=0.16)
    for ax, cls, tag in zip(axes, ("bus/truck", "pedestrian"), ("(b) bus/truck", "(c) pedestrian")):
        chosen = payload["chosen"][cls]
        ev = chosen["event"]
        rec = payload["events"][cls]
        seed, mount, density = chosen["job"]
        u, cell = int(ev["ue"]), int(ev["cell"])
        with np.load(ROOT / "results" / "cache" / mount / density / f"seed_{seed}" / "m3_timeline.npz") as f:
            n_k = f["los_loss_db"].shape[0]
        lo = max(0, ev["start_k"] - 300)
        hi = min(n_k - 1, ev["end_k"] + 200)
        ts, loss = loss_1ms((seed, mount, density), u, lo * DT, hi * DT)
        t = ts - ev["start_s"]
        own = np.clip(np.nan_to_num(loss[:, cell], posinf=CAP_DB), 0, CAP_DB)
        oth = np.clip(np.nan_to_num(loss[:, 1 - cell], posinf=CAP_DB), 0, CAP_DB)
        ax.axvspan(0.0, ev["end_s"] - ev["start_s"], color="#fde0dd", lw=0, label="10 dB event")
        ax.plot(t, own, color="black", lw=1.1, label="Serving cell")
        ax.plot(t, oth, color="gray", lw=0.9, ls="--", label="Other cell")
        ax.axhline(10.0, color="gray", lw=0.4, ls=":")
        for key, (color, ls, lw, alpha, z, lab) in MARKS.items():
            for ho in rec["handovers"].get(key, []):
                ax.axvline(ho["switch_s"] - ev["start_s"], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
        ax.set_ylim(-1, CAP_DB)
        ax.set_yticks([0, 20, 40])
        ax.set_xlim(t[0], t[-1])
        ax.grid(True, alpha=0.5)
        ax.text(0.015, 0.94, tag, transform=ax.transAxes, ha="left", va="top", fontsize=8, zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    fig.supylabel("LoS loss [dB]", fontsize=8.5, x=0.01, y=0.645)
    axes[-1].set_xlabel("Time relative to event start [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    for key, (color, ls, lw, alpha, _z, lab) in MARKS.items():
        handles.append(plt.Line2D([], [], color=color, ls=ls, lw=lw, alpha=alpha))
        labels.append(lab)
    order = [1, 2, 0, 3, 4]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.54, 0.0), handlelength=1.8, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
    print(f"wrote {save(fig, 'fig_onset2.pdf')}")


if __name__ == "__main__":
    main()
