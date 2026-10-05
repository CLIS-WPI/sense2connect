"""Paper 2 (second review): definitions and counts behind the onset table (scripts/p2_diag_onset.py).

Recounts, for the held-out set and the 2 s start-up exclusion of p2_diag_onset.py, the events and
the epochs per window under the single-assignment rule actually used (each epoch gets exactly one
label; priority during > [-0.5, 0) s > [-1, -0.5) s > elsewhere), and how many epochs would qualify
for more than one window if windows of consecutive events were allowed to overlap (i.e. how often the
priority rule mattered). Serving cell of an event: the paper-1 reference association
xapp.metrics.fixed_cell, i.e. per UE and 10 ms step the cell with the larger UNBLOCKED (blockage-free)
received power (no handover policy); events = 10 dB LoS-loss events on that cell's LoS (paper-1
hysteresis: gaps < 0.5 s merged), the event's cell = fixed cell at its start.
Writes results/P2/onset_definitions_<set>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> None:
    from p2_seeds import eval_tag
    from p2_seeds import load as load_seeds

    tag = eval_tag()
    seeds = load_seeds()["evaluation"]
    windows = ("[-1,-0.5) s", "[-0.5,0) s", "during", "elsewhere")
    counts = {w: 0 for w in windows}
    multi = {"during & pre-onset": 0, "both pre-onset windows (different events)": 0, "any multiple": 0}
    n_events = 0
    cell_switch_during = 0
    for s in seeds:
        for mount in ("lamppost", "facade"):
            for dens in ("low", "high"):
                meta = json.loads((ROOT / "results" / "cache" / mount / dens / f"seed_{s}" / "m3_timeline_meta.json").read_text())
                T = 600
                t = np.arange(T) * 0.1
                for u in range(2):
                    evs = [e for e in meta["events"] if e["ue"] == u]
                    n_events += len(evs)
                    during = np.zeros(T, bool)
                    pre1 = np.zeros(T, int)
                    pre05 = np.zeros(T, int)
                    for e in evs:
                        st = e["start_s"]
                        during |= (t >= st) & (t <= e["end_s"])
                        pre1 += ((t >= st - 1.0) & (t < st - 0.5)).astype(int)
                        pre05 += ((t >= st - 0.5) & (t < st)).astype(int)
                    keep = t >= 2.0
                    lab = np.where(during, "during", np.where(pre05 > 0, "[-0.5,0) s", np.where(pre1 > 0, "[-1,-0.5) s", "elsewhere")))
                    for w in windows:
                        counts[w] += int(np.sum((lab == w) & keep))
                    nwin = during.astype(int) + (pre05 > 0) + (pre1 > 0)
                    multi["during & pre-onset"] += int(np.sum(during & ((pre05 + pre1) > 0) & keep))
                    multi["both pre-onset windows (different events)"] += int(np.sum((pre05 > 0) & (pre1 > 0) & ~during & keep))
                    multi["any multiple"] += int(np.sum((nwin > 1) & keep))
    total = sum(counts.values())
    out = {"definition": __doc__, "set": tag, "n_events": n_events, "epochs_per_window_total": counts, "epochs_total": total,
           "epochs_per_window_per_seed": {w: counts[w] / len(seeds) for w in windows}, "epochs_qualifying_for_several_windows": multi,
           "share_of_epochs_with_several_windows": multi["any multiple"] / total}
    (ROOT / "results" / "P2" / f"onset_definitions_{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "definition"}, indent=1))


if __name__ == "__main__":
    main()
