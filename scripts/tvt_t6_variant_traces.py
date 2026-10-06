"""TVT T6: ray traces of the sensitivity variants (UE speed, O-RU height) and of the second deployment.

Generalises scripts/tvt_t6_intersection_traces.py: --config selects the scenario config (alias mounts keep
the caches apart: configs/tvt_sweep_ueslow.yaml, tvt_sweep_uefast.yaml, tvt_sweep_h3.yaml, tvt_sweep_h10.yaml,
tvt_intersection.yaml) and --mounts the mount names. UE-speed variants do not move any sensing target, so
their radar steps (1, 2) are not needed (the radar detections of the paper-1 mount are reused).

Runs the UNCHANGED paper-1/2 trace functions with the intersection config (configs/tvt_intersection.yaml,
mount "corner", cache results/cache/corner/<density>/seed_<s>; scene configs/scenes/tvt_intersection,
registered in every worker by sim/tvt/scene_register.py):
1. sensing caches + LoS events  (run_m2_review._ensure; RCSSolver + PathSolver, deterministic)
2. radar detections             (run_m2_review._detections, the M2 CFAR grid, 1024 subcarriers)
3. comm path geometry           (cache_comm_geometry._one)
4. complex path coefficients    (p2_trace._one, results/P2/cache/corner/...)
Seeds: tuning 101-105 and development 1001-1010 (configs/seeds_tvt.yaml, guard applied); densities low
and high. comm_trace.json (paper-1 event statistics) is not produced. Existing complete caches are skipped.
Run: python scripts/tvt_t6_variant_traces.py --config configs/tvt_sweep_h3.yaml --mounts lamppost_h3 [--steps 1 2 3 4] [--sets development]
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
_OPTS = {"config": str(CFG)}


def _init(config: str | None = None) -> None:
    for p in (str(ROOT), str(ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from sim.tvt.scene_register import register

    register()
    if config:
        _OPTS["config"] = config


def _raw() -> dict:
    from sim.scenes.config import load_yaml

    return load_yaml(Path(_OPTS["config"]))


def _ensure(job) -> str:
    mount, job = job[0], job[1:]
    import run_m2_review as M

    density, seed = job
    M._ensure(_raw(), mount, density, seed, None)
    return f"sensing {density} {seed}"


def _detect(job) -> str:
    mount, job = job[0], job[1:]
    import torch

    torch.set_num_threads(1)
    import run_m2_review as M
    from sim.sensing.waveform import waveform_from_config

    density, seed = job
    raw = _raw()
    M._detections(raw, mount, density, seed, waveform_from_config(raw), M._geometry(raw, mount), M.CFAR_GRID)
    return f"detections {density} {seed}"


def _geometry(job) -> str:
    mount, job = job[0], job[1:]
    import cache_comm_geometry as G

    density, seed = job
    G._one((_raw(), mount, density, seed, 0, False))
    return f"geometry {density} {seed}"


def _p2(job) -> str:
    mount, job = job[0], job[1:]
    import p2_trace as T

    density, seed = job
    T._one((_raw(), mount, density, seed, 0, False))
    return f"p2 paths {density} {seed}"


def main() -> None:
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mounts", nargs="*", required=True)
    ap.add_argument("--sets", nargs="*", default=["development"])
    a = ap.parse_args()
    s = load()
    seeds = check(a.seeds or [x for name in a.sets for x in s[name]])
    jobs = [(m, d, x) for x in seeds for m in a.mounts for d in ("low", "high")]
    cfgp = str((ROOT / a.config).resolve())
    ctx = mp.get_context("spawn")
    clock = time.perf_counter()
    for step, fn in ((1, _ensure), (2, _detect), (3, _geometry), (4, _p2)):
        if step not in a.steps:
            continue
        with ctx.Pool(a.workers, initializer=_init, initargs=(cfgp,)) as pool:
            for msg in pool.imap_unordered(fn, jobs):
                print(msg, flush=True)
        print(f"step {step}: {(time.perf_counter() - clock) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
