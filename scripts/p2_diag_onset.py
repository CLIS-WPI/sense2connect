"""Paper 2 DIAGNOSTIC (external review of the draft): estimator error around the onset of blockage events.

Held-out set (S2C_EVAL_SET), main configuration (400 MHz, TDoA, sync 1 ns, phi 2 deg,
blocked LoS diffracted), estimator variants v1 and A (EKF output, 0.1 s epochs; first 2 s
excluded). Events: the paper-1 10 dB LoS events of the SERVING (fixed) cell of each UE
(m3_timeline_meta.json of the paper-1 timeline build: model-B LoS loss >= 10 dB with the
paper-1 hysteresis / 0.5 s minimum gap, 10 ms resolution). Each epoch of a UE is put in
one window: [-1, -0.5) s or [-0.5, 0) s before the onset of the next event, during an
event ([start, end]), or elsewhere (pre-onset windows take precedence over "elsewhere";
"during" takes precedence over pre-onset windows of a later event). Seed level: per
seed the median, p90 and the shares of epochs with error > 0.1 m and > 0.25 m per
window (pooled over its 4 runs and 2 UEs); mean over the 10 seeds with 95 % bootstrap
CIs over seeds. Writes results/P2/diag_onset_<set>.json.
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

CFG = "bw400_tdoa_s1_p2_b"
WINDOWS = ("[-1,-0.5) s", "[-0.5,0) s", "during", "elsewhere")


def boot(v, n=10000, seed=20261005):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main() -> None:
    from p2_seeds import eval_tag
    from p2_seeds import load as load_seeds

    tag = eval_tag()
    seeds = load_seeds()["evaluation"]
    out = {"definition": __doc__, "set": tag, "config": CFG, "variants": {}}
    for variant in ("v1", "A"):
        d_ = "est" if variant == "v1" else f"est_{variant}"
        per_seed = {w: {k: [] for k in ("median_m", "p90_m", "share_gt_0.1", "share_gt_0.25", "n")} for w in WINDOWS}
        n_events = 0
        for s in seeds:
            pooled = {w: [] for w in WINDOWS}
            for mount in ("lamppost", "facade"):
                for dens in ("low", "high"):
                    tr = np.load(ROOT / "results" / "P2" / d_ / tag / CFG / f"{mount}_{dens}_{s}_track.npz")
                    ms = np.load(ROOT / "results" / "P2" / "est" / tag / CFG / f"{mount}_{dens}_{s}.npz")
                    meta = json.loads((ROOT / "results" / "cache" / mount / dens / f"seed_{s}" / "m3_timeline_meta.json").read_text())
                    err = np.linalg.norm(tr["xy_ekf"] - ms["ue"][..., :2], axis=-1)
                    err = np.where(np.isfinite(err), err, 1e3)
                    T, U = err.shape
                    t = np.arange(T) * 0.1
                    for u in range(U):
                        lab = np.full(T, "elsewhere", dtype=object)
                        evs = [e for e in meta["events"] if e["ue"] == u]
                        n_events += len(evs)
                        for e in evs:
                            st = e["start_s"]
                            lab[(t >= st - 1.0) & (t < st - 0.5) & (lab == "elsewhere")] = "[-1,-0.5) s"
                            lab[(t >= st - 0.5) & (t < st) & ((lab == "elsewhere") | (lab == "[-1,-0.5) s"))] = "[-0.5,0) s"
                        for e in evs:
                            lab[(t >= e["start_s"]) & (t <= e["end_s"])] = "during"
                        for w in WINDOWS:
                            sel = (lab == w) & (t >= 2.0)
                            pooled[w].append(err[sel, u])
            for w in WINDOWS:
                v = np.concatenate(pooled[w])
                per_seed[w]["n"].append(int(v.size))
                per_seed[w]["median_m"].append(float(np.median(v)) if v.size else np.nan)
                per_seed[w]["p90_m"].append(float(np.percentile(v, 90)) if v.size else np.nan)
                per_seed[w]["share_gt_0.1"].append(float(np.mean(v > 0.1)) if v.size else np.nan)
                per_seed[w]["share_gt_0.25"].append(float(np.mean(v > 0.25)) if v.size else np.nan)
        res = {"n_events": n_events}
        for w in WINDOWS:
            res[w] = {k: {"mean": float(np.nanmean(v)), "ci95_boot": boot(v)} for k, v in per_seed[w].items() if k != "n"}
            res[w]["epochs_per_seed"] = per_seed[w]["n"]
        out["variants"][variant] = res
        print(f"{variant}: {n_events} events")
        for w in WINDOWS:
            r = res[w]
            print(f"  {w:12s}: median {r['median_m']['mean']:.3f} m, p90 {r['p90_m']['mean']:.2f} m, >0.1 m {r['share_gt_0.1']['mean']:.3f}, "
                  f">0.25 m {r['share_gt_0.25']['mean']:.3f} (epochs/seed {int(np.mean(r['epochs_per_seed']))})")
    (ROOT / "results" / "P2" / f"diag_onset_{tag}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
