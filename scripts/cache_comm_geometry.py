"""Comm-only path geometry for the 10 ms model-B timeline (M3 rework).

``comm_trace.json`` keeps per-link sums and the sensing cache keeps comm
paths without vertices, so model B cannot be re-evaluated between
snapshots from them. This script re-solves the comm scene only
(``PathSolver``, deterministic, no sensing targets, no RCS) and stores, per
0.1 s snapshot and per (UE, O-RU, path): the segment points [m], the
unblocked path power (sum over the 8x8 array, unit transmit power), the
path class, and the load-stable ``path_key``. Sensing caches are not
touched. Each job also records how far the re-solved model-B powers are
from ``comm_trace.json`` at the snapshot times.

Run: ``python scripts/cache_comm_geometry.py [--workers 4] [--seeds ...]``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml  # noqa: E402

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")
KIND = "comm_geometry"
CLASS_CODE = {"los": 0, "ground": 1, "wall": 2, "double": 3}
MAX_POINTS = 4  # max_depth 2: tx, two bounces, rx


def sources() -> list[Path]:
    """Modules whose content defines the geometry cache."""
    return [
        ROOT / "scripts" / "cache_comm_geometry.py",
        ROOT / "sim" / "scenes" / "loop.py",
        ROOT / "sim" / "scenes" / "motion.py",
        ROOT / "sim" / "scenes" / "traffic.py",
        ROOT / "sim" / "scenes" / "config.py",
        ROOT / "sim" / "comm" / "blockage.py",
    ]


def job_dir(mount: str, density: str, seed: int) -> Path:
    return ROOT / "results" / "cache" / mount / density / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--frames", type=int, default=0, help="0 = full 60 s job")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    chosen = args.seeds or [int(s) for s in list(seeds["tuning"]) + list(seeds["evaluation"])]
    jobs = [(raw, m, d, int(s), args.frames, args.force) for s in chosen for m in MOUNTS for d in DENSITIES]
    clock = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, min(4, args.workers))) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    print(json.dumps({"jobs": len(rows), "wall_s": time.perf_counter() - clock}, indent=2), flush=True)
    worst = max((row.get("check", {}).get("max_rel_blocked", 0.0) for row in rows), default=0.0)
    print(f"max relative blocked-power difference vs comm_trace.json: {worst:.3e}", flush=True)


def _one(item: tuple) -> dict[str, Any]:
    raw, mount, density, seed, frames, force = item
    import torch

    torch.set_num_threads(4)
    from sim.scenes.loop import (
        _add_radios,
        _device_positions,
        _object_catalog,
        _path_power,
        _path_segments,
        _scene_by_name,
        _solver_kwargs,
        _target_catalog,
        blockage_for_snapshot,
        pack_paths,
    )
    from sim.scenes.motion import move_radios, states_at, track_identity
    from sim.scenes.traffic import prepare_scenario
    from sim.sensing.provenance import scoped_version
    from sionna.rt import PathSolver, load_scene

    directory = job_dir(mount, density, seed)
    out_npz = directory / "comm_geometry.npz"
    out_meta = directory / "comm_geometry_meta.json"
    if out_npz.exists() and out_meta.exists() and not force:
        print(f"skip {mount} {density} {seed}", flush=True)
        return json.loads(out_meta.read_text(encoding="utf-8"))
    clock = time.perf_counter()
    scenario = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    scenario["blockage"]["min_gap_s"] = 0.5
    n_snap = int(scenario["n_snapshots"]) if not frames else int(frames)
    comm = load_scene(_scene_by_name(str(scenario["scene"])))
    _add_radios(comm, scenario, include_targets=False)
    catalog = _object_catalog(comm)
    target_sizes = _target_catalog(scenario)
    oru_names = [oru["name"] for oru in scenario["orus"]]
    ue_names = [ue["name"] for ue in scenario["ues"]]
    radios = [comm.get(name) for name in oru_names + ue_names]
    solver = PathSolver(deterministic=True)
    wavelength_m = float(np.asarray(comm.wavelength.numpy()).reshape(-1)[0])
    n_ue, n_oru = len(ue_names), len(oru_names)

    per_snap: list[list[list[list[tuple]]]] = []
    ue_pos = np.zeros((n_snap, n_ue, 3))
    oru_pos = np.zeros((n_snap, n_oru, 3))
    blocked_check = np.full((n_snap, n_ue, n_oru), np.nan)
    unblocked_check = np.full((n_snap, n_ue, n_oru), np.nan)
    max_paths = 0
    for index in range(n_snap):
        t_s = index * float(scenario["dt_s"])
        states = states_at(scenario, t_s)
        move_radios(radios, states)
        packed = pack_paths(solver(comm, **_solver_kwargs(scenario, "comm")))
        tx = _device_positions(comm, oru_names)
        rx = _device_positions(comm, ue_names)
        for u, name in enumerate(ue_names):
            ue_pos[index, u] = rx[name]
        for c, name in enumerate(oru_names):
            oru_pos[index, c] = tx[name]
        links = []
        for u, ue in enumerate(ue_names):
            row = []
            for c, oru in enumerate(oru_names):
                paths = []
                for p in range(int(packed["a"].shape[-2])):
                    parsed = _path_segments(
                        packed["vertices"], packed["interactions"], packed["objects"], packed["valid"],
                        p, u, c, tx[oru], rx[ue], catalog,
                    )
                    if parsed is None:
                        continue
                    _pid, key, cls, segments = parsed
                    points = [segments[0][0]] + [seg[1] for seg in segments]
                    paths.append((points, _path_power(packed["a"], u, c, p), CLASS_CODE[cls], key))
                max_paths = max(max_paths, len(paths))
                row.append(paths)
            links.append(row)
        per_snap.append(links)
        _, _, metrics = blockage_for_snapshot(
            scenario, packed, tx, rx, {name: states[name] for name in target_sizes}, target_sizes,
            wavelength_m, catalog, store_segments=False,
            identities={spec["name"]: track_identity(scenario, spec, t_s) for spec in list(scenario["vehicles"]) + list(scenario["pedestrians"])},
        )
        for metric in metrics:
            u = ue_names.index(metric["ue"])
            c = oru_names.index(metric["oru"])
            blocked_check[index, u, c] = float(metric["blocked_power"])
            unblocked_check[index, u, c] = float(metric["unblocked_power"])

    n_p = max(1, max_paths)
    points = np.full((n_snap, n_ue, n_oru, n_p, MAX_POINTS, 3), np.nan)
    n_points = np.zeros((n_snap, n_ue, n_oru, n_p), dtype=np.int8)
    power = np.zeros((n_snap, n_ue, n_oru, n_p))
    cls = np.full((n_snap, n_ue, n_oru, n_p), -1, dtype=np.int8)
    keys = np.full((n_snap, n_ue, n_oru, n_p), "", dtype="<U96")
    for s, links in enumerate(per_snap):
        for u, row in enumerate(links):
            for c, paths in enumerate(row):
                for p, (pts, pw, code, key) in enumerate(paths):
                    points[s, u, c, p, : len(pts)] = np.asarray(pts)
                    n_points[s, u, c, p] = len(pts)
                    power[s, u, c, p] = pw
                    cls[s, u, c, p] = code
                    keys[s, u, c, p] = key

    check = _compare_trace(directory, blocked_check, unblocked_check, ue_names, oru_names)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        points_m=points,
        n_points=n_points,
        unblocked_power=power,
        path_class=cls,
        path_key=keys,
        ue_position_m=ue_pos,
        oru_position_m=oru_pos,
        blocked_power_snapshot=blocked_check,
    )
    meta = {
        "seed": seed,
        "mount": mount,
        "density": density,
        "dt_s": float(scenario["dt_s"]),
        "n_snapshots": n_snap,
        "ues": ue_names,
        "orus": oru_names,
        "wavelength_m": wavelength_m,
        "max_paths": n_p,
        "class_code": CLASS_CODE,
        "solver": _solver_kwargs(scenario, "comm"),
        "runtime_s": time.perf_counter() - clock,
        "check": check,
        "provenance": {"kind": KIND, **scoped_version(sources())},
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {mount} {density} {seed} {meta['runtime_s']:.0f} s check {check}", flush=True)
    return meta


def _compare_trace(directory: Path, blocked: np.ndarray, unblocked: np.ndarray, ues: list[str], orus: list[str]) -> dict[str, float]:
    """Relative difference of the re-solved powers to ``comm_trace.json`` at the snapshots."""
    path = directory / "comm_trace.json"
    if not path.exists():
        return {"max_rel_blocked": float("nan"), "max_rel_unblocked": float("nan"), "n": 0}
    trace = json.loads(path.read_text(encoding="utf-8"))
    rel_b, rel_u = [], []
    for row in trace["ue_trace"]:
        s = int(row["snapshot"])
        if s >= blocked.shape[0]:
            continue
        u, c = ues.index(row["ue"]), orus.index(row["oru"])
        for ref, got, out in ((row["blocked_power"], blocked[s, u, c], rel_b), (row["unblocked_power"], unblocked[s, u, c], rel_u)):
            ref = float(ref)
            out.append(abs(float(got) - ref) / max(abs(ref), 1e-30))
    return {
        "max_rel_blocked": float(max(rel_b)) if rel_b else float("nan"),
        "max_rel_unblocked": float(max(rel_u)) if rel_u else float("nan"),
        "n": len(rel_b),
    }


if __name__ == "__main__":
    main()
