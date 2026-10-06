"""TVT T6: paper-2 estimator A (frozen, tuned results/P2/est_tuned_A.json) on the sensitivity variants of the canyon.

Uses the paper-2 dominant-path measurements that scripts/tvt_t4_measure.py stores with every job (dom_*;
identical to the frozen measurement) and the frozen track_v2 (variant A) with the variant's scenario
config (walkable map from it). Not applicable to the intersection (the paper-2 walkable map and curb rule
assume a single street). Writes results/TVT/T6/est_A/<cfg>/<mount>_<density>_<seed>_track.npz.
Run: python scripts/tvt_t6_est_variants.py --scenario configs/tvt_sweep_h3.yaml --mounts lamppost_h3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> None:
    import p2_estimate as P
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--mounts", nargs="+", required=True)
    ap.add_argument("--cfg", default="bw400_tdoa_s1_p2_b")
    ap.add_argument("--set", default="development")
    a = ap.parse_args()
    raw = load_yaml(ROOT / a.scenario)
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    c = P.configs()[a.cfg]
    params = json.loads((ROOT / "results" / "P2" / "est_tuned_A.json").read_text())["tuned"][f"{c['bw']}|{c['timing']}"]["params"]
    n_sc = int(p2cfg["bandwidths"][c["bw"]]["n_sc"])
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    out = ROOT / "results" / "TVT" / "T6" / "est_A" / a.cfg
    out.mkdir(parents=True, exist_ok=True)
    for s in check(load()[a.set]):
        for m in a.mounts:
            for d in ("low", "high"):
                job = (s, m, d)
                g = np.load(ROOT / "results" / "TVT" / "T4" / "meas" / a.set / a.cfg / f"{m}_{d}_{s}.npz")
                meas = {k: g[f"dom_{k}"] for k in ("tau_ns", "uy", "uz", "snr", "ratio_db")}
                tr = P.track_v2(meas, g["oru"], c["timing"], params, n_sc, df, job, raw, p2cfg, "A")
                np.savez(out / f"{m}_{d}_{s}_track.npz", **tr)
                e = np.linalg.norm(tr["xy_ekf"] - g["ue"][..., :2], axis=-1)
                print(job, f"median {np.nanmedian(e[20:]):.3f} m", flush=True)


if __name__ == "__main__":
    main()
