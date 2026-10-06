"""TVT T4: tracker parameters on the TUNING seeds (101-105) only.

1. Constant visibility prior for the "none" ablation: the mean oracle visibility of the
   predicted LoS and NLoS sources (cond oracle, default parameters, tuning seeds).
2. Grid over (q_acc, floor_u, pd_nlos) for the method (cond pred_real); objective as in paper 2:
   median position error, then the 90th percentile (pooled over the tuning runs, first 2 s
   excluded). All ablations use the selected parameters unchanged ("identical parameters").
Writes results/TVT/T4/tune_rows_<k>.json per shard and, with --merge, results/TVT/T4/tuned.json.
Run: python scripts/tvt_t4_tune.py --shard k --nshards 4 (k = 0..3), then --merge
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

GRID = {"q_acc": [0.2, 1.0], "floor_u": [0.001, 0.002], "pd_nlos": [0.5, 0.8], "eps_robust": [0.1, 0.3]}


def errors(res) -> np.ndarray:
    return np.concatenate([np.linalg.norm(r["xy"][20:] - r["ue"][20:, :, :2], axis=-1).ravel() for r in res.values()])


def main() -> None:
    import tvt_t4_track as T4
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
    jobs = [(s, m, d) for s in check(load()["tuning"]) for m in ("lamppost", "facade") for d in ("low", "high")]
    clock = time.perf_counter()
    qp = ROOT / "results" / "TVT" / "T4" / "q_prior.json"
    if qp.exists():
        q_prior = json.loads(qp.read_text())
    else:
        q_prior = T4.run(jobs, "tuning", "oracle", 0.0, {}, raw, p2cfg, tcfg, save=False, log=False)["q_mean"]
        qp.write_text(json.dumps(q_prior) + "\n")
    print("q prior (oracle mean, tuning seeds):", q_prior, flush=True)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    tdir = ROOT / "results" / "TVT" / "T4"
    if a.merge:
        rows = [r for f in sorted(tdir.glob("tune_rows_*.json")) for r in json.loads(f.read_text())]
        best = min(rows, key=lambda r: (round(r["median_m"], 3), r["p90_m"]))
        out = {"definition": __doc__, "grid": GRID, "rows": rows, "best": best, "params": {**best["params"], "_q_prior": q_prior}}
        (tdir / "tuned.json").write_text(json.dumps(out, indent=1) + "\n")
        print("best", best)
        return
    rows = []
    for gi, vals in enumerate(itertools.product(*GRID.values())):
        if gi % a.nshards != a.shard:
            continue
        prm = dict(zip(GRID.keys(), vals))
        prm["_q_prior"] = q_prior
        t0 = time.perf_counter()
        out = T4.run(jobs, "tuning", "pred_real", 0.0, prm, raw, p2cfg, tcfg, save=False, log=False)
        e = errors(out["res"])
        rows.append({"params": {k: v for k, v in prm.items() if not k.startswith("_")}, "median_m": float(np.nanmedian(e)),
                     "p90_m": float(np.nanpercentile(e, 90)), "wall_s": time.perf_counter() - t0})
        print(rows[-1], flush=True)
    (tdir / f"tune_rows_{a.shard}.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"shard {a.shard} done in {time.perf_counter() - clock:.0f} s")


if __name__ == "__main__":
    main()
