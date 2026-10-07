"""TVT panels: paper-2 estimator A re-tuned and run on the back-to-back panel measurements (configs/tvt.yaml panels).

Measurements: the paper-2 dominant-path measurement of the panel with the larger snapshot energy per O-RU and epoch
(results/TVT/T4/meas/<set>/<cfg>/<job>.npz: dom_*, dom_sx; scripts/tvt_t4_measure.py).
tune  the paper-2 procedure on the TUNING seeds (scripts/p2_estimate.py tune): v1 grid (p2_estimate.GRID, 144 points,
      frozen v1 estimator) for the base parameters, then the variant-A grid (VGRID["A"], sel_gate, 3 points) with the
      panel-restricted estimator A (sim/tvt/est_panels.py); objective median EKF error, then p90, pooled.
      Writes results/TVT/est_A/est_tuned_A.json (same layout as results/P2/est_tuned_A.json).
run   estimator-A tracks with the tuned parameters -> results/TVT/est_A/<set>/<cfg>/<mount>_<density>_<seed>_track.npz.
Run: python scripts/tvt_est_tune.py tune --cfgs bw400_tdoa_s1_p2_b bw100_tdoa_s1_p2_b
     python scripts/tvt_est_tune.py run --set development [--cfg ...] [--scenario ... --mounts ...]
"""

from __future__ import annotations

import argparse
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

OUT = ROOT / "results" / "TVT" / "est_A"
MEAS = ROOT / "results" / "TVT" / "T4" / "meas"
KEYS = ("tau_ns", "uy", "uz", "snr", "ratio_db")


def _meas(set_name: str, cfg: str, job: tuple) -> dict:
    g = np.load(MEAS / set_name / cfg / f"{job[1]}_{job[2]}_{job[0]}.npz")
    d = {k: g[f"dom_{k}"] for k in KEYS} | {"sx": g["dom_sx"]}
    return {"meas": d, "oru": g["oru"], "ue": g["ue"]}


def tune(cfgs: list[str]) -> None:
    import p2_estimate as P
    from sim.scenes.config import load_yaml
    from sim.tvt.est_panels import track_v2
    from sim.tvt.seeds import check, load

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    jobs = [(s, m, d) for s in check(load()["tuning"]) for m in ("lamppost", "facade") for d in ("low", "high")]
    f = OUT / "est_tuned_A.json"
    res = json.loads(f.read_text()) if f.exists() else {"tuned": {}, "base": {}}
    for cfg in cfgs:
        c = P.configs()[cfg]
        key = f"{c['bw']}|{c['timing']}"
        if key in res["tuned"]:
            print(f"{key}: tuned already", flush=True)
            continue
        n_sc = int(p2cfg["bandwidths"][c["bw"]]["n_sc"])
        data = [(j, _meas("tuning", cfg, j)) for j in jobs]
        clock = time.perf_counter()
        # stage 1: v1 grid (frozen v1 estimator; its own front/back handling)
        memos = [{} for _ in data]
        scores = {}
        for combo in itertools.product(*P.GRID.values()):
            params = dict(zip(P.GRID, combo))
            errs = []
            for (j, d), memo in zip(data, memos):
                meas = {k: d["meas"][k] for k in KEYS}
                out = P.track(meas, {"oru": d["oru"]}, c["timing"], params, n_sc, df, memo)
                e = np.linalg.norm(out["xy_ekf"] - d["ue"][..., :2], axis=-1)
                errs.append(np.where(np.isfinite(e), e, 1e3).ravel())
            e = np.concatenate(errs)
            scores[combo] = (float(np.median(e)), float(np.percentile(e, 90)))
        best = min(scores, key=lambda k: (scores[k][0], scores[k][1], k))
        p0 = dict(zip(P.GRID, best))
        res["base"][key] = {"params": p0, "median_m": scores[best][0], "p90_m": scores[best][1], "grid_points": len(scores)}
        print(f"{key} v1: {res['base'][key]} ({(time.perf_counter() - clock) / 60:.1f} min)", flush=True)
        # stage 2: variant A grid with the panel-restricted estimator A
        grid = P.VGRID["A"]
        memos = [{} for _ in data]
        scores = {}
        for combo in itertools.product(*grid.values()):
            params = {**p0, **dict(zip(grid, combo))}
            errs = []
            for (j, d), memo in zip(data, memos):
                out = track_v2(d["meas"], d["oru"], c["timing"], params, n_sc, df, j, raw, p2cfg, "A", memo)
                e = np.linalg.norm(out["xy_ekf"] - d["ue"][..., :2], axis=-1)
                errs.append(np.where(np.isfinite(e), e, 1e3).ravel())
            e = np.concatenate(errs)
            scores[combo] = (float(np.median(e)), float(np.percentile(e, 90)))
        best = min(scores, key=lambda k: (scores[k][0], scores[k][1], k))
        res["tuned"][key] = {"params": {**p0, **dict(zip(grid, best))}, "median_m": scores[best][0], "p90_m": scores[best][1], "grid_points": len(scores)}
        print(f"{key} A: {res['tuned'][key]} ({(time.perf_counter() - clock) / 60:.1f} min)", flush=True)
        res |= {"grid_v1": P.GRID, "grid_A": grid, "jobs": [list(j) for j in jobs], "objective": "median EKF error, then p90, pooled over tuning runs",
                "measurements": "dominant path of the panel with the larger snapshot energy (results/TVT/T4/meas/tuning)"}
        OUT.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, indent=1) + "\n")


def run(set_name: str, cfg: str, scenario: str, mounts: list[str]) -> None:
    import p2_estimate as P
    from sim.scenes.config import load_yaml
    from sim.tvt.est_panels import track_v2
    from sim.tvt.panels import est_params, est_track_path
    from sim.tvt.seeds import check, load

    raw = load_yaml(ROOT / scenario)
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    c = P.configs()[cfg]
    params = est_params(c["bw"], c["timing"])
    n_sc = int(p2cfg["bandwidths"][c["bw"]]["n_sc"])
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    for s in check(load()[set_name]):
        for m in mounts:
            for d in ("low", "high"):
                job = (s, m, d)
                x = _meas(set_name, cfg, job)
                tr = track_v2(x["meas"], x["oru"], c["timing"], params, n_sc, df, job, raw, p2cfg, "A")
                f = est_track_path(set_name, cfg, job)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez(f, **tr)
                e = np.linalg.norm(tr["xy_ekf"] - x["ue"][..., :2], axis=-1)
                print(job, cfg, f"median {np.nanmedian(e[20:]):.3f} m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "run"])
    ap.add_argument("--cfgs", nargs="*", default=["bw400_tdoa_s1_p2_b"])
    ap.add_argument("--cfg", default="bw400_tdoa_s1_p2_b")
    ap.add_argument("--set", default="development")
    ap.add_argument("--scenario", default="configs/m2_scenario.yaml")
    ap.add_argument("--mounts", nargs="*", default=["lamppost", "facade"])
    a = ap.parse_args()
    if a.stage == "tune":
        tune(a.cfgs)
    else:
        run(a.set, a.cfg, a.scenario, a.mounts)


if __name__ == "__main__":
    main()
