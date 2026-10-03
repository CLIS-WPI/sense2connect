"""Run independent sensing jobs on the one visible GPU.

Each job is one ``(seed, mount, density)``. ``--workers`` processes share
that GPU. ``--check`` repeats the jobs one at a time and requires the
cached paths to match byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sim.scenes.traffic import prepare_scenario
from sim.sensing.trace import trace_job
from sim.scenes.config import load_yaml

def _scenario(seed: int, mount: str, density: str, frames: int) -> dict:
    return prepare_scenario(
        load_yaml(ROOT / "configs" / "m2_scenario.yaml"),
        seed=seed,
        mount=mount,
        density=density,
        duration_s=0.1 * frames,
        dt_s=0.1,
    )


def _run(job: tuple[int, str, str, int, str]) -> str:
    seed, mount, density, frames, root = job
    trace_job(
        _scenario(seed, mount, density, frames),
        mount=mount,
        density=density,
        cache_root=Path(root),
        max_frames=frames,
        process=False,
    )
    return f"{mount}/{density}/seed_{seed}"


def _compare(parallel_root: Path, sequential_root: Path) -> dict[str, object]:
    """Sensing arrays must match exactly. Comm residuals are reported beside that."""
    sensing_mismatch: list[str] = []
    comm_abs = 0.0
    files = 0
    for path in sorted(sequential_root.rglob("*.npz")):
        files += 1
        other = parallel_root / path.relative_to(sequential_root)
        with np.load(path, allow_pickle=True) as left, np.load(other, allow_pickle=True) as right:
            for name in left.files:
                if name.endswith("object_names") or left[name].dtype.kind == "O":
                    continue
                a = left[name]
                b = right[name]
                if a.shape != b.shape:
                    sensing_mismatch.append(f"{path.name}:{name}:shape")
                    continue
                if a.dtype.kind in "bOSU":
                    if not np.array_equal(a, b) and not name.endswith("object_names"):
                        sensing_mismatch.append(f"{path.name}:{name}")
                    continue
                delta = float(np.max(np.abs(a - b))) if a.size else 0.0
                if name.startswith("comm_"):
                    comm_abs = max(comm_abs, delta)
                elif delta != 0.0:
                    sensing_mismatch.append(f"{path.name}:{name}:{delta}")
    return {
        "files": files,
        "sensing_bitwise": not sensing_mismatch,
        "sensing_mismatch": sensing_mismatch[:8],
        "comm_max_abs": comm_abs,
    }


def _digest(directory: Path) -> dict[str, str]:
    """Hash each stored array. Nameless-mesh labels are not part of the check."""
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        key = str(path.relative_to(directory))
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as data:
                blobs = [
                    name.encode() + data[name].tobytes()
                    for name in sorted(data.files)
                    if name not in {"rcs_object_names", "bg_object_names", "comm_object_names"}
                    and data[name].dtype.kind != "O"
                ]
            hashes[key] = hashlib.sha256(b"".join(blobs)).hexdigest()
        else:
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--mounts", nargs="+", default=["lamppost"])
    parser.add_argument("--densities", nargs="+", default=["low"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "results" / "cache" / "batch")
    args = parser.parse_args()
    jobs = [
        (seed, mount, density, args.frames)
        for seed in args.seeds
        for mount in args.mounts
        for density in args.densities
    ]
    parallel_root = args.cache_root / "parallel"
    if parallel_root.exists():
        shutil.rmtree(parallel_root)
    started = time.perf_counter()
    payloads = [(*job, str(parallel_root)) for job in jobs]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        list(pool.map(_run, payloads))
    parallel_s = time.perf_counter() - started
    print(f"parallel_s {parallel_s:.3f} workers {args.workers} jobs {len(jobs)}")
    if not args.check:
        return
    sequential_root = args.cache_root / "sequential"
    if sequential_root.exists():
        shutil.rmtree(sequential_root)
    started = time.perf_counter()
    for job in jobs:
        _run((*job, str(sequential_root)))
    sequential_s = time.perf_counter() - started
    report = _compare(parallel_root, sequential_root)
    print(f"sequential_s {sequential_s:.3f}")
    print(report)
    if report["sensing_mismatch"]:
        raise SystemExit("sensing paths are not bitwise identical")


if __name__ == "__main__":
    main()
