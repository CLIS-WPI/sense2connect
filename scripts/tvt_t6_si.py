"""TVT T6: residual radar self-interference (full-duplex leakage after cancellation) - radar detections.

Model: the residual self-interference that survives analogue/digital cancellation and is not removed by
the (blind, slow-time mean) static-clutter removal is spread by phase noise over range-Doppler and acts
like additional white noise; it is parameterised by its interference-to-noise ratio INR [dB] and
enters as a raised effective noise figure NF + 10 log10(1 + 10^(INR/10)) of the radar receiver (the
communication link is unaffected). Detections are recomputed from the unchanged sensing channel caches
with the unchanged paper-1 detection function (scripts/run_m2_review._detections) and the paper-1 CFAR
setting of the trackers (configs/m3.yaml sensing budgets: train 4, pfa 1e-3, guard 2); the cache
directory lookup is redirected to results/TVT/T6/si/inr<k>/<mount>/<density>/seed_<s>, which holds
symbolic links to the read-only sensing caches, so results/cache is never written.
The sensing-trace caches carry whole-tree provenance (commit + hash of uncommitted changes), so this
script runs from a frozen clone of the repository (scripts/tvt_frozen.sh); --ensure first re-traces the
sensing caches of the requested seeds there with the unchanged paper-1 trace function (same scene,
radios, solver settings and seeds; deterministic). INR 0 reproduces the nominal detections and is the
regression against the paper-1 detection caches.
Run (frozen clone): python scripts/tvt_t6_si.py --ensure --inr 0 10 20 [--workers 4]
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "results" / "TVT" / "T6" / "si"


def si_dir(inr: float, mount: str, density: str, seed: int) -> Path:
    return OUT / f"inr{inr:g}" / mount / density / f"seed_{seed}"


def _link(inr, mount, density, seed) -> Path:
    src = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    dst = si_dir(inr, mount, density, seed)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.startswith("detections_"):
            continue
        t = dst / f.name
        if not t.exists():
            os.symlink(f, t)
    return dst


def _ensure(item) -> str:
    mount, density, seed = item
    import run_m2_review as M
    from sim.scenes.config import load_yaml

    M._ensure(load_yaml(ROOT / "configs" / "m2_scenario.yaml"), mount, density, seed, None)
    return f"sensing {mount} {density} {seed}"


def _one(item) -> str:
    inr, mount, density, seed = item
    import torch

    torch.set_num_threads(2)
    import run_m2_review as M
    from sim.scenes.config import load_yaml
    from sim.sensing.waveform import waveform_from_config

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    raw["sensing_radar"]["noise"]["noise_figure_db"] = float(raw["sensing_radar"]["noise"]["noise_figure_db"]) + 10 * math.log10(1 + 10 ** (inr / 10))
    d = _link(inr, mount, density, seed)
    M.cache_dir = lambda root, m, de, s: d  # redirect reads (links) and the detections write
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    budgets = cfg["sensing"]["budgets"]
    grid = [{"guard": int(raw["sensing_radar"]["cfar"]["guard"]), "train": int(b["train"]), "pfa": float(b["pfa"])} for b in budgets.values()]
    grid = [dict(t) for t in {tuple(sorted(g.items())) for g in grid}]
    M._detections(raw, mount, density, seed, waveform_from_config(raw), M._geometry(raw, mount), grid)
    return f"inr {inr} {mount} {density} {seed}"


def main() -> None:
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--inr", type=float, nargs="*", default=[10.0, 20.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sets", nargs="*", default=["development"])
    ap.add_argument("--ensure", action="store_true", help="re-trace missing sensing caches first (frozen clone)")
    a = ap.parse_args()
    s = load()
    seeds = check([x for name in a.sets for x in s[name]])
    if a.ensure:
        jobs = [(m, d, x) for x in seeds for m in ("lamppost", "facade") for d in ("low", "high")]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_ensure, jobs):
                print(msg, flush=True)
    items = [(inr, m, d, x) for inr in a.inr for x in seeds for m in ("lamppost", "facade") for d in ("low", "high")]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_one, items):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
