"""Paper 2: complex path coefficients (and the finite-difference validation subset).

Re-solves the comm scene of one job exactly as scripts/cache_comm_geometry.py
(same scene, radios, PathSolver settings and seed, deterministic, no
sensing targets) and stores, per 0.1 s snapshot, UE, O-RU and path (in the
paper-1 path order, matched by path_key):
- a_center: complex path coefficient at the O-RU array centre, from the
  per-element coefficients a_m (synthetic 8x8 array) as
  mean_m a_m conj(s_m(u)), s_m(u) = exp(+j 2 pi/lambda u . r_m), u = unit
  direction from the O-RU along the first path segment; the residual
  max_m |a_m - a_center s_m| / |a_center| is recorded (steering check);
- tau_sionna [s]; the paper-1 cache's points/power are not rewritten.
Check: per-path power sum_m |a_m|^2 equals the paper-1 cached power and the
path keys match (cross-process tolerance 1.5e-5 relative).

With --fd (validation subset): four extra receivers per UE at x +/- 1 cm and
y +/- 1 cm; per persisting path (same key) the central differences of the
Sionna delay and of the AoA (from the raw, unrounded first vertex) are
stored for comparison with the analytic image-method derivatives.

Writes results/P2/cache/<mount>/<density>/seed_<s>/p2_paths.npz (+ meta with
stage-scoped provenance) or p2_fd.npz. Never writes into results/cache.
Run: python scripts/p2_trace.py [--workers 4] [--seeds ...] [--fd --frames N].
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")
KIND = "p2_paths"
FD_STEP_M = 0.01
OUT_ROOT = ROOT / "results" / "P2" / "cache"


def sources() -> list[Path]:
    return [ROOT / "scripts" / "p2_trace.py", ROOT / "sim" / "positioning" / "array.py", ROOT / "sim" / "scenes" / "loop.py",
            ROOT / "sim" / "scenes" / "motion.py", ROOT / "sim" / "scenes" / "traffic.py", ROOT / "sim" / "scenes" / "config.py"]


def _one(item: tuple) -> dict[str, Any]:
    raw, mount, density, seed, frames, fd = item
    import torch

    torch.set_num_threads(4)
    from sim.positioning.array import element_positions, steering
    from sim.scenes.loop import _add_radios, _device_positions, _object_catalog, _path_segments, _scene_by_name, _solver_kwargs, pack_paths
    from sim.scenes.motion import move_radios, states_at
    from sim.scenes.traffic import prepare_scenario
    from sim.sensing.provenance import scoped_version
    from sionna.rt import PathSolver, Receiver, load_scene

    out_dir = OUT_ROOT / mount / density / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "p2_fd" if fd else "p2_paths"
    if (out_dir / f"{name}.npz").exists() and (out_dir / f"{name}_meta.json").exists():
        return json.loads((out_dir / f"{name}_meta.json").read_text())
    clock = time.perf_counter()
    geo_dir = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    with np.load(geo_dir / "comm_geometry.npz") as g:
        keys_ref = g["path_key"]
        power_ref = g["unblocked_power"]
    scenario = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    n_snap = int(scenario["n_snapshots"]) if not frames else int(frames)
    comm = load_scene(_scene_by_name(str(scenario["scene"])))
    _add_radios(comm, scenario, include_targets=False)
    catalog = _object_catalog(comm)
    oru_names = [o["name"] for o in scenario["orus"]]
    ue_names = [u["name"] for u in scenario["ues"]]
    offsets = [np.array(v) for v in ((FD_STEP_M, 0, 0), (-FD_STEP_M, 0, 0), (0, FD_STEP_M, 0), (0, -FD_STEP_M, 0))] if fd else []
    extra = []
    if fd:
        st0 = states_at(scenario, 0.0)
        for ue in ue_names:
            for k, off in enumerate(offsets):
                extra.append(Receiver(f"{ue}_fd{k}", position=(st0[ue]["position_m"] + off).tolist(), velocity=(0.0, 0.0, 0.0)))
        comm.add(extra)
    radios = [comm.get(n) for n in oru_names + ue_names]
    rx_names = ue_names + [r.name for r in extra]
    solver = PathSolver(deterministic=True)
    wl = float(np.asarray(comm.wavelength.numpy()).reshape(-1)[0])
    r_el = element_positions(wl)
    n_u, n_c, n_p = len(ue_names), len(oru_names), keys_ref.shape[-1]
    a_center = np.zeros((n_snap, n_u, n_c, n_p), dtype=np.complex128)
    tau_s = np.full((n_snap, n_u, n_c, n_p), np.nan)
    resid = np.full((n_snap, n_u, n_c, n_p), np.nan)
    pw_err = np.zeros((n_snap, n_u, n_c, n_p))
    key_mismatch = 0
    fd_out = {"tau": np.full((n_snap, n_u, n_c, n_p, 4), np.nan), "u": np.full((n_snap, n_u, n_c, n_p, 4, 3), np.nan)}
    for s in range(n_snap):
        t_s = s * float(scenario["dt_s"])
        states = states_at(scenario, t_s)
        move_radios(radios, states)
        for r in extra:
            ue, k = r.name.rsplit("_fd", 1)
            r.position = (states[ue]["position_m"] + offsets[int(k)]).tolist()
        paths = solver(comm, **_solver_kwargs(scenario, "comm"))
        packed = pack_paths(paths)
        tx = _device_positions(comm, oru_names)
        rx = _device_positions(comm, rx_names)
        a = packed["a"]  # [rx, rx_ant, tx, tx_ant, path, time]
        for u, ue in enumerate(ue_names):
            for c, oru in enumerate(oru_names):
                slot = 0
                for p in range(int(a.shape[-2])):
                    parsed = _path_segments(packed["vertices"], packed["interactions"], packed["objects"], packed["valid"], p, u, c, tx[oru], rx[ue], catalog)
                    if parsed is None:
                        continue
                    _pid, key, _cls, segs = parsed
                    if slot >= n_p or key != keys_ref[s, u, c, slot]:
                        key_mismatch += 1
                        slot += 1
                        continue
                    am = a[u, 0, c, :, p, 0].astype(np.complex128)
                    first = segs[0][1] if len(segs) > 1 else rx[ue]
                    if len(segs) > 1:  # raw (unrounded) first vertex
                        depth0 = int(np.flatnonzero(packed["interactions"][:, u, c, p] != 0)[0])
                        first = packed["vertices"][depth0, u, c, p].astype(np.float64)
                    d = first - tx[oru]
                    uvec = d / np.linalg.norm(d)
                    sv = steering(uvec, r_el, wl)
                    ac = np.mean(am * np.conj(sv))
                    a_center[s, u, c, slot] = ac
                    resid[s, u, c, slot] = float(np.max(np.abs(am - ac * sv)) / max(abs(ac), 1e-300))
                    tau_s[s, u, c, slot] = float(packed["tau"][u, c, p])
                    pw = float(np.sum(np.abs(am) ** 2))
                    pw_err[s, u, c, slot] = abs(pw - power_ref[s, u, c, slot]) / max(power_ref[s, u, c, slot], 1e-300)
                    if fd:
                        for k in range(4):
                            ri = n_u + u * 4 + k
                            for q in range(int(a.shape[-2])):
                                pk = _path_segments(packed["vertices"], packed["interactions"], packed["objects"], packed["valid"], q, ri, c, tx[oru], rx[rx_names[ri]], catalog)
                                if pk is None or pk[1] != key:
                                    continue
                                if len(pk[3]) > 1:
                                    d0 = int(np.flatnonzero(packed["interactions"][:, ri, c, q] != 0)[0])
                                    f2 = packed["vertices"][d0, ri, c, q].astype(np.float64)
                                else:
                                    f2 = rx[rx_names[ri]]
                                dd = f2 - tx[oru]
                                fd_out["tau"][s, u, c, slot, k] = float(packed["tau"][ri, c, q])
                                fd_out["u"][s, u, c, slot, k] = dd / np.linalg.norm(dd)
                                break
                    slot += 1
    meta = {"seed": seed, "mount": mount, "density": density, "n_snapshots": n_snap, "wavelength_m": wl, "fd": bool(fd), "fd_step_m": FD_STEP_M,
            "key_mismatch": key_mismatch, "max_steering_residual": float(np.nanmax(resid)), "max_power_rel_err": float(np.max(pw_err)),
            "runtime_s": time.perf_counter() - clock, "provenance": {"kind": KIND, **scoped_version(sources())}}
    if fd:
        np.savez(out_dir / "p2_fd.npz", tau_nominal=tau_s, fd_tau=fd_out["tau"], fd_u=fd_out["u"], offsets=np.array(offsets))
    else:
        np.savez(out_dir / "p2_paths.npz", a_center=a_center, tau_sionna=tau_s, steering_residual=resid, power_rel_err=pw_err)
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{name} {mount} {density} {seed}: {meta['runtime_s']:.0f} s, key mismatch {key_mismatch}, steering residual {meta['max_steering_residual']:.2e}, "
          f"power err {meta['max_power_rel_err']:.2e}", flush=True)
    return meta


def main() -> None:
    from p2_seeds import load as load_seeds
    from sim.scenes.config import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--sets", nargs="*", default=["tuning", "evaluation"], help="seed sets from p2_seeds.load()")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--mounts", nargs="*", default=list(MOUNTS))
    ap.add_argument("--densities", nargs="*", default=list(DENSITIES))
    args = ap.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    seeds = load_seeds()
    chosen = args.seeds or [int(s) for name in args.sets for s in seeds[name]]
    jobs = [(raw, m, d, int(s), args.frames, args.fd) for s in chosen for m in args.mounts for d in args.densities]
    clock = time.perf_counter()
    with mp.get_context("spawn").Pool(max(1, min(4, args.workers))) as pool:
        metas = pool.map(_one, jobs, chunksize=1)
    print(json.dumps({"jobs": len(metas), "wall_s": time.perf_counter() - clock, "key_mismatch": sum(m["key_mismatch"] for m in metas),
                      "max_steering_residual": max(m["max_steering_residual"] for m in metas),
                      "max_power_rel_err": max(m["max_power_rel_err"] for m in metas)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
