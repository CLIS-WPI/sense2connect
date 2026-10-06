"""TVT T6: ray traces of the second deployment (four-way intersection), scene design committed before
tracing (configs/tvt_intersection_design.md, commit bb1f60f).

Runs the UNCHANGED paper-1/2 trace functions with the intersection config (configs/tvt_intersection.yaml,
mount "corner", cache results/cache/corner/<density>/seed_<s>; scene configs/scenes/tvt_intersection,
registered in every worker by sim/tvt/scene_register.py):
1. sensing caches + LoS events  (run_m2_review._ensure; RCSSolver + PathSolver, deterministic)
2. radar detections             (run_m2_review._detections, the M2 CFAR grid, 1024 subcarriers)
3. comm path geometry           (cache_comm_geometry._one)
4. complex path coefficients    (p2_trace._one, results/P2/cache/corner/...)
Seeds: tuning 101-105 and development 1001-1010 (configs/seeds_tvt.yaml, guard applied); densities low
and high. comm_trace.json (paper-1 event statistics) is not produced. Existing complete caches are skipped.
Run: python scripts/tvt_t6_intersection_traces.py [--workers 4] [--steps 1 2 3 4] [--seeds ...]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CFG = ROOT / "configs" / "tvt_intersection.yaml"
MOUNT = "corner"


def _init() -> None:
    for p in (str(ROOT), str(ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from sim.tvt.scene_register import register

    register()


def _raw() -> dict:
    from sim.scenes.config import load_yaml

    return load_yaml(CFG)


def _ensure(job) -> str:
    import run_m2_review as M

    density, seed = job
    M._ensure(_raw(), MOUNT, density, seed, None)
    return f"sensing {density} {seed}"


def _detect(job) -> str:
    import torch

    torch.set_num_threads(1)
    import run_m2_review as M
    from sim.sensing.waveform import waveform_from_config

    density, seed = job
    raw = _raw()
    M._detections(raw, MOUNT, density, seed, waveform_from_config(raw), M._geometry(raw, MOUNT), M.CFAR_GRID)
    return f"detections {density} {seed}"


def _geometry(job) -> str:
    import cache_comm_geometry as G

    density, seed = job
    G._one((_raw(), MOUNT, density, seed, 0, False))
    return f"geometry {density} {seed}"


def _p2(job) -> str:
    import p2_trace as T

    density, seed = job
    T._one((_raw(), MOUNT, density, seed, 0, False))
    return f"p2 paths {density} {seed}"


def main() -> None:
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    s = load()
    seeds = check(a.seeds or (s["tuning"] + s["development"]))
    jobs = [(d, x) for x in seeds for d in ("low", "high")]
    ctx = mp.get_context("spawn")
    clock = time.perf_counter()
    for step, fn in ((1, _ensure), (2, _detect), (3, _geometry), (4, _p2)):
        if step not in a.steps:
            continue
        with ctx.Pool(a.workers, initializer=_init) as pool:
            for msg in pool.imap_unordered(fn, jobs):
                print(msg, flush=True)
        print(f"step {step}: {(time.perf_counter() - clock) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
