"""Model-B loss vs time for one typical bus/truck and one pedestrian event (paper/figs/fig_onset.pdf).

Restyled version of results/M3/onset/onset_events.pdf. The events, the
selection rule and the handover times come from
results/M3/onset/onset_events.json (written by scripts/run_m3_genie.py).
The loss curves are evaluated at 1 ms resolution (changed after the second
external review, so the figure matches the 1 ms onset analysis of
scripts/review_a34.py): model B analytically on the LoS segment O-RU -> UE
of both cells from the exact analytic poses (sim.scenes.motion.states_at;
no re-trace), the same evaluation as review_a34; the printed 10-90 % onset
is the 1 ms value from results/M5/review_a34.json. Selection rule:
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
sys.path.insert(0, str(ROOT))
from figstyle import COLORS, COLUMN_IN, save, setup  # noqa: E402

DT = 0.01
DT1 = 0.001
CAP_DB = 40.0
# handover markers: (color, line style, width, alpha, z-order); genie drawn wide and light under A3
MARKS = {
    "A3 @ 3GPP ref.": (COLORS["a3"], "-", 0.9, 1.0, 4),
    "A3 @ 10 dB": (COLORS["a3"], "--", 0.9, 1.0, 4),
    "genie+A3 @ 3GPP ref.": (COLORS["genie"], "-", 2.2, 0.5, 3),
    "genie+A3 @ 10 dB": (COLORS["genie"], "--", 2.2, 0.5, 3),
}


def loss_1ms(job: tuple, ue_index: int, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Model-B LoS loss [dB] of both cells for one UE every 1 ms on [t0, t1] s: times [N], loss [N, 2]."""
    from sim.comm.blockage import path_blocker_loss, sum_blockage_db
    from sim.scenes.config import load_yaml
    from sim.scenes.motion import states_at
    from sim.scenes.traffic import prepare_scenario

    seed, mount, density = job
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    wavelength = 299792458.0 / float(raw["carrier_hz"])
    blockers = list(sc["vehicles"]) + list(sc["pedestrians"])
    oru = [np.asarray(o["position_m"], dtype=np.float64) for o in sc["orus"]]
    ue_name = sc["ues"][ue_index]["name"]
    ts = np.arange(int(round(t0 / DT1)), int(round(t1 / DT1)) + 1) * DT1
    loss = np.zeros((len(ts), 2))
    for i, t in enumerate(ts):
        st = states_at(sc, float(t))
        ue = st[ue_name]["position_m"]
        for c in range(2):
            per = [path_blocker_loss([(oru[c], ue)], st[b["name"]]["position_m"], float(b["length_m"]), float(b["width_m"]), float(b["height_m"]), wavelength)[0]
                   for b in blockers]
            loss[i, c] = sum_blockage_db(per)
    return ts, loss


def main() -> None:
    plt = setup()
    payload = json.loads((ROOT / "results" / "M3" / "onset" / "onset_events.json").read_text())
    a4 = json.loads((ROOT / "results" / "M5" / "review_a34.json").read_text())["a4_onset"]["events"]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.4))
    for ax, cls, tag in zip(axes, ("bus/truck", "pedestrian"), ("(a)", "(b)")):
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
        match = [r for r in a4 if r["job"] == [seed, mount, density] and r["ue"] == u and abs(r["start_s"] - ev["start_s"]) < 1e-9]
        if len(match) != 1:
            raise SystemExit(f"chosen {cls} event not found once in results/M5/review_a34.json")
        onset = match[0]["onset_1ms_s"]
        ax.axvspan(0.0, ev["end_s"] - ev["start_s"], color="#fde0dd", lw=0, label="10 dB event")
        ax.plot(t, own, color="black", lw=1.1, label="Serving (blocked) cell, 1 ms resolution")
        ax.plot(t, oth, color="gray", lw=0.9, ls="--", label="Other cell")
        ax.axhline(10.0, color="gray", lw=0.4, ls=":")
        for key, (color, ls, lw, alpha, z) in MARKS.items():
            for ho in rec["handovers"].get(key, []):
                ax.axvline(ho["switch_s"] - ev["start_s"], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
        ax.set_ylabel("LoS loss [dB]")
        ax.set_ylim(-1, CAP_DB)
        ax.grid(True, alpha=0.5)
        ax.text(0.01, 0.95, f"{tag} {cls}, 10–90 % onset {onset * 1e3:.0f} ms", transform=ax.transAxes, ha="left", va="top", fontsize=7, zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
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
