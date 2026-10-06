"""TVT T5: paper-2 estimator A (frozen, tuned parameters results/P2/est_tuned_A.json) on the TUNING seeds.

Paper 2 stored estimator-A tracks only for its evaluation sets; the T5 baseline "paper-1 planner with
the paper-2 estimator" is tuned on the tuning seeds and therefore needs them there. Uses the stored
paper-2 measurements results/P2/est/tuning/<cfg>/<job>.npz and the frozen track_v2 (variant A).
Writes results/TVT/T5/est_A/tuning/<cfg>/<mount>_<density>_<seed>_track.npz.
Run: python scripts/tvt_t5_est_tuning.py
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


def main() -> None:
    import p2_estimate as P
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    c = P.configs()[CFG]
    params = json.loads((ROOT / "results" / "P2" / "est_tuned_A.json").read_text())["tuned"][f"{c['bw']}|{c['timing']}"]["params"]
    n_sc = int(p2cfg["bandwidths"][c["bw"]]["n_sc"])
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    out = ROOT / "results" / "TVT" / "T5" / "est_A" / "tuning" / CFG
    out.mkdir(parents=True, exist_ok=True)
    for s in check(load()["tuning"]):
        for m in ("lamppost", "facade"):
            for d in ("low", "high"):
                job = (s, m, d)
                g = np.load(ROOT / "results" / "P2" / "est" / "tuning" / CFG / f"{m}_{d}_{s}.npz")
                meas = {k: g[k] for k in ("tau_ns", "uy", "uz", "snr", "ratio_db")}
                tr = P.track_v2(meas, g["oru"], c["timing"], params, n_sc, df, job, raw, p2cfg, "A")
                np.savez(out / f"{m}_{d}_{s}_track.npz", **tr)
                e = np.linalg.norm(tr["xy_ekf"] - g["ue"][..., :2], axis=-1)
                print(job, f"median {np.nanmedian(e[20:]):.3f} m", flush=True)


if __name__ == "__main__":
    main()
