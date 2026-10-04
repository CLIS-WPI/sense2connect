"""Wall time per stage of the v1 M3 pipeline on a few tuning jobs.

Run inside the container; sample GPU 1 from the host with
``nvidia-smi dmon -i 1 -s um -d 1`` while it runs. Writes
``results/M3/profile_before.json``. Totals for the full v1 run are
extrapolated from the per-call times and the v1 call counts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_m3_v1 as v1  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.tracks import load_detections, replay_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "M3" / "profile_before.json")
    args = parser.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    stages: dict[str, float] = {}

    t0 = time.perf_counter()
    jobs = v1._jobs(raw, cfg, [args.seed])
    stages["load_4_jobs_s"] = time.perf_counter() - t0

    job = jobs[0]
    directory = ROOT / "results" / "cache" / job["mount"] / job["density"] / f"seed_{job['seed']}"
    t0 = time.perf_counter()
    detections = load_detections(directory)
    det = cfg["sensing"]["budgets"][2]
    replay_map(
        detections,
        train=int(det["train"]),
        pfa=float(det["pfa"]),
        eps_m=float(det["eps_m"]),
        ghost_association_m=float(det["ghost_association_m"]),
        walls=[float(v) for v in raw["sensing_radar"]["wall_y_m"]],
        spec=raw["sensing_radar"],
        lanes=list(raw["lanes"]),
        sidewalks=list(raw["sidewalks"]),
        tuned=cfg["sensing"]["map_tracker"],
    )
    stages["track_replay_1_job_1_budget_s"] = time.perf_counter() - t0
    del detections

    t0 = time.perf_counter()
    v1._oracle_check(jobs)
    stages["oracle_check_4_jobs_s"] = time.perf_counter() - t0

    base = v1._base_params(cfg)
    params = {
        "none": base,
        "oracle": base,
        "beam": base,
        "a3": {**base, "offset_db": 1.0, "hysteresis_db": 1.0, "ttt_s": 0.08},
        "trend": {**base, "window_s": 0.2, "drop_db": 5.0, "horizon_s": 0.5},
        "xapp": {**base, "horizon_s": 1.0, "hold_s": 0.5},
    }
    per_call: dict[str, float] = {}
    for scheme, p in params.items():
        times = []
        for item in jobs:
            t0 = time.perf_counter()
            v1._run(item, scheme, p, cfg, 2 if scheme == "xapp" else None)
            times.append(time.perf_counter() - t0)
        per_call[scheme] = float(np.mean(times))
        print(f"{scheme}: {per_call[scheme]:.2f} s/job", flush=True)

    # v1 call counts: tuning 20 jobs, evaluation 40 jobs.
    n_tune, n_eval = 20, 40
    xgrid = 2 * len(cfg["xapp"]["horizon_s"]) * len(cfg["xapp"]["hold_s"])
    agrid = len(cfg["a3"]["offset_db"]) * len(cfg["a3"]["hysteresis_db"]) * len(cfg["a3"]["ttt_s"])
    tgrid = len(cfg["trend"]["window_s"]) * len(cfg["trend"]["drop_db"]) * len(cfg["trend"]["horizon_s"])
    extrapolated = {
        "load_all_jobs_s": stages["load_4_jobs_s"] / 4 * (n_tune + n_eval),
        "tune_s": n_tune * (per_call["none"] + xgrid * per_call["xapp"] + agrid * per_call["a3"] + tgrid * per_call["trend"]),
        "evaluate_s": n_eval * sum(per_call.values()),
        "delay_sweep_s": n_eval * len(cfg["e2"]["loop_delay_sweep_s"]) * per_call["xapp"],
        "ho_sweep_s": n_eval * len(cfg["e2"]["tau_ho_sweep_s"]) * per_call["xapp"],
    }
    extrapolated["total_s"] = sum(extrapolated.values())
    payload = {
        "seed": args.seed,
        "jobs": [f"{j['mount']}/{j['density']}" for j in jobs],
        "stages_measured_s": stages,
        "simulate_per_job_s": per_call,
        "v1_full_run_extrapolated_s": extrapolated,
        "device": "CPU only (v1 uses NumPy/Python; no torch in M3 v1)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
