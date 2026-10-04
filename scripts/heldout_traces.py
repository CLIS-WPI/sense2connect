"""Trace the held-out test seeds (configs/seeds_heldout.yaml; second external review, Block 3).

Runs the existing, unchanged trace and detection functions for every
held-out job (seed x mount x density) on GPU 1:
1. sensing channel cache + LoS events: scripts/run_m2_review.py ``_ensure``
   (RCSSolver + PathSolver via sim.sensing.trace.trace_job), 4 workers;
2. radar detections (1024 subcarriers, the CFAR grid of the M2 review):
   scripts/run_m2_review.py ``_detections`` -> detections_1024.json;
3. comm-only model-B traces: scripts/cache_comm.py ``_one`` -> comm_trace.json;
4. comm path geometry: scripts/cache_comm_geometry.py ``--seeds <held-out>``
   (run as a subprocess, unchanged).
Existing complete caches are skipped by those functions. The tuning and
development caches are never touched.

Run: ``python scripts/heldout_traces.py [--workers 4] [--steps 1 2 3 4]``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")


def _detect(job: tuple) -> str:
    import torch

    torch.set_num_threads(1)
    import run_m2_review as M
    from sim.scenes.config import load_yaml
    from sim.sensing.waveform import waveform_from_config

    mount, density, seed = job
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    M._detections(raw, mount, density, seed, waveform_from_config(raw), M._geometry(raw, mount), M.CFAR_GRID)
    return f"{mount}/{density}/{seed}"


def _comm(job: tuple) -> None:
    import cache_comm
    from sim.scenes.config import load_yaml

    mount, density, seed = job
    cache_comm._one((load_yaml(ROOT / "configs" / "m2_scenario.yaml"), mount, density, seed))


def main() -> None:
    import run_m2_review as M
    from seedsets import load_seeds

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="*", help="smoke test only: other seeds instead of the held-out set")
    ap.add_argument("--frames", type=int, default=0, help="smoke test only: 0 = full 60 s jobs")
    ap.add_argument("--mounts", nargs="*", default=list(MOUNTS))
    ap.add_argument("--densities", nargs="*", default=list(DENSITIES))
    args = ap.parse_args()
    held = args.seeds or [int(s) for s in load_seeds()["heldout"]]
    frames = None if args.frames <= 0 else int(args.frames)
    jobs = [(m, d, s) for s in held for m in args.mounts for d in args.densities]
    ctx = mp.get_context("spawn")
    clock = time.perf_counter()
    if 1 in args.steps:
        with ctx.Pool(args.workers) as pool:
            pool.map(M._ensure_worker, [(m, d, s, frames) for m, d, s in jobs], chunksize=1)
        print(f"step 1 sensing caches: {len(jobs)} jobs, {(time.perf_counter() - clock) / 60:.1f} min", flush=True)
    if 2 in args.steps:
        with ctx.Pool(args.workers) as pool:
            for done in pool.imap_unordered(_detect, jobs):
                print(f"detections {done}", flush=True)
        print(f"step 2 detections: {(time.perf_counter() - clock) / 60:.1f} min", flush=True)
    if 3 in args.steps:
        with ctx.Pool(args.workers) as pool:
            pool.map(_comm, jobs, chunksize=1)
        print(f"step 3 comm traces: {(time.perf_counter() - clock) / 60:.1f} min", flush=True)
    if 4 in args.steps:
        cmd = [sys.executable, str(ROOT / "scripts" / "cache_comm_geometry.py"), "--workers", str(args.workers), "--seeds", *map(str, held)]
        if frames:
            cmd += ["--frames", str(frames)]
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"step 4 comm geometry: {(time.perf_counter() - clock) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
