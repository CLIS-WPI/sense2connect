"""Paper-2 camera-ready, phase 2 (pre-declared in results/P2/phase2_predeclared.md): the isotropic TVT tracker
on the paper-2 held-out seeds 3001-3010.

Code: this branch = tvt at 36223bc (last isotropic-array commit) plus this driver only; sim/tvt, the
measurement (scripts/tvt_t4_measure.measure) and the tracker run (scripts/tvt_t4_track.run) are unchanged.
DELIBERATE, LOGGED OVERRIDE of the TVT seed guard (sim/tvt/seeds.check refuses the consumed paper-2 held-out
seeds 3001-3010 always): the seeds are taken from configs/p2.yaml heldout2 instead of sim/tvt/seeds, and each
run appends a record to results/P2/phase2_override_log.jsonl (time, commit, seeds, stage, reason).
Stage "measure": paper-2 main configuration bw400_tdoa_s1_p2_b (same synthesis and random streams as
scripts/p2_estimate.measure_job); regression of the dominant-path measurement against the paper-2 held-out
measurements results/P2/est/heldout2/<cfg>/<job>.npz (must be exact, else abort).
Stage "track": visibility condition "none", sigma_map 0, frozen parameters configs/tvt_frozen/tracker.json
(tuned on seeds 101-105) and configs/tvt_frozen/visibility_calibration.json (loaded, unused for "none").
Outputs: results/TVT/T4/meas/heldout2/<cfg>/ and results/TVT/T4/track/heldout2/none_map0/ (results/TVT is
mounted on a separate directory for this run).
Run: python scripts/p2cr_tracker_heldout.py measure|track
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CFG = "bw400_tdoa_s1_p2_b"
SET = "heldout2"
REASON = ("paper-2 camera-ready phase 2, pre-declared in results/P2/phase2_predeclared.md: stronger baseline "
          "(isotropic TVT tracker, visibility none, frozen parameters) run once on the paper-2 held-out seeds")


def log_override(stage: str, seeds: list[int]) -> None:
    try:
        commit = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    rec = {"time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "commit": commit, "stage": stage,
           "seeds": seeds, "guard": "sim/tvt/seeds.check bypassed (seeds from configs/p2.yaml heldout2)", "reason": REASON}
    with open(ROOT / "results" / "P2" / "phase2_override_log.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("OVERRIDE", json.dumps(rec), flush=True)


def main() -> None:
    from sim.scenes.config import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "track"])
    a = ap.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    seeds = [int(s) for s in p2cfg["heldout2"]]
    assert seeds == list(range(3001, 3011)), seeds
    jobs = [(s, m, d) for s in seeds for m in ("lamppost", "facade") for d in ("low", "high")]
    log_override(a.stage, seeds)
    if a.stage == "measure":
        from tvt_t4_measure import OUT, measure

        ecfg = load_yaml(ROOT / "configs" / "tvt.yaml")["extract"]
        out = OUT / SET / CFG
        out.mkdir(parents=True, exist_ok=True)
        worst, clock = 0.0, time.perf_counter()
        for job in jobs:
            t0 = time.perf_counter()
            res = measure(job, raw, p2cfg, ecfg, CFG)
            ref = ROOT / "results" / "P2" / "est" / SET / CFG / f"{job[1]}_{job[2]}_{job[0]}.npz"
            with np.load(ref) as g:
                d = max(float(np.max(np.abs(g[k] - res["dom"][k]))) for k in ("tau_ns", "uy", "uz"))
            worst = max(worst, d)
            if d != 0.0:
                raise SystemExit(f"{job}: dominant-path measurement differs from {ref} (max |diff| {d})")
            np.savez(out / f"{job[1]}_{job[2]}_{job[0]}.npz", **{f"dom_{k}": v for k, v in res["dom"].items()},
                     **{f"comp_{k}": v for k, v in res["comp"].items()}, ue=res["ue"], oru=res["oru"], blocked=res["blocked"],
                     los_loss=res["los_loss"], delta_ns=res["truth"]["delta_ns"], b_clk_ns=res["truth"]["b_clk_ns"], period_ns=res["period_ns"])
            print(f"{job}: {time.perf_counter() - t0:.0f} s, paper-2 regression max |diff| {d:.2e}", flush=True)
        meta = {"set": SET, "cfg": CFG, "jobs": [list(j) for j in jobs], "extract": ecfg, "paper2_measurement_max_abs_diff": worst,
                "wall_s": time.perf_counter() - clock}
        (out / "meta_lamppost_facade.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps(meta), flush=True)
    else:
        from tvt_t4_track import run

        tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
        params = json.loads((ROOT / "configs" / "tvt_frozen" / "tracker.json").read_text())["params"]
        out = run(jobs, SET, "none", 0.0, params, raw, p2cfg, tcfg, cfg=CFG, calibration_path=ROOT / "configs" / "tvt_frozen" / "visibility_calibration.json")
        errs = np.concatenate([np.linalg.norm(r["xy"][20:] - r["ue"][20:, :, :2], axis=-1).ravel() for r in out["res"].values()])
        errs = errs[np.isfinite(errs)]
        print(f"held-out none map 0: median {np.median(errs):.3f} m, p90 {np.percentile(errs, 90):.3f} m, timing {out['timing']}", flush=True)


if __name__ == "__main__":
    main()
