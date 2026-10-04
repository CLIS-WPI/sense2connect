"""10 ms comm timeline with model B evaluated analytically at every step.

Path geometry and unblocked path powers come from the 0.1 s comm geometry
cache (``scripts/cache_comm_geometry.py``) and are held between snapshots;
the UE end of every path is moved to the exact UE position at each 10 ms
step. Blocker positions are the analytic poses at that time
(``sim.scenes.motion.states_at``). Model B is applied to every segment of
every path for every blocker, so blocked power is never interpolated.

The GPU path (``model_b_timeline``) uses ``sim.comm.blockage_torch`` in
float64; ``model_b_timeline_numpy`` is the reference built on
``sim.comm.blockage.path_blocker_loss``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.comm.blockage import path_blocker_loss, sum_blockage_db
from sim.scenes.motion import states_at, track_identity

CLASS_LOS = 0
LOSS_CAP_DB = 200.0  # stands in for an infinite (fully blocked) model-B loss


def comm_times(n_snapshots: int, dt_sense_s: float, dt_comm_s: float) -> np.ndarray:
    """Comm grid [s] from 0 to the last snapshot time."""
    n = int(round((n_snapshots - 1) * dt_sense_s / dt_comm_s)) + 1
    return np.arange(n, dtype=np.float64) * dt_comm_s


def load_geometry(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Geometry arrays and metadata of one job."""
    meta = json.loads((directory / "comm_geometry_meta.json").read_text(encoding="utf-8"))
    with np.load(directory / "comm_geometry.npz") as data:
        geom = {key: data[key] for key in data.files}
    return geom, meta


def actor_tracks(scenario: dict[str, Any], times_s: np.ndarray) -> dict[str, Any]:
    """Blocker poses [T, B, 3], sizes [B, 3], kinds, lap indices [T, B], UE positions [T, U, 3]."""
    blockers = list(scenario["vehicles"]) + list(scenario["pedestrians"])
    names = [spec["name"] for spec in blockers]
    ues = [ue["name"] for ue in scenario["ues"]]
    pos = np.zeros((len(times_s), len(names), 3))
    ue_pos = np.zeros((len(times_s), len(ues), 3))
    ue_vel = np.zeros((len(times_s), len(ues), 3))
    lap = np.zeros((len(times_s), len(names)), dtype=np.int32)
    for k, t_s in enumerate(times_s):
        states = states_at(scenario, float(t_s))
        for b, name in enumerate(names):
            pos[k, b] = states[name]["position_m"]
        for u, name in enumerate(ues):
            ue_pos[k, u] = states[name]["position_m"]
            ue_vel[k, u] = states[name]["velocity_mps"]
        for b, spec in enumerate(blockers):
            lap[k, b] = int(track_identity(scenario, spec, float(t_s)).rsplit("#", 1)[1])
    sizes = np.array([[float(s["length_m"]), float(s["width_m"]), float(s["height_m"])] for s in blockers])
    return {
        "blocker_position_m": pos,
        "blocker_size_m": sizes,
        "blocker_kind": [str(s["kind"]) for s in blockers],
        "blocker_name": names,
        "blocker_lap": lap,
        "ue_position_m": ue_pos,
        "ue_velocity_mps": ue_vel,
        "ue_names": ues,
    }


def held_segments(geom: dict[str, np.ndarray], times_s: np.ndarray, ue_pos: np.ndarray, dt_sense_s: float) -> dict[str, np.ndarray]:
    """Segments [T, U, C, P, S, 2, 3] of the held snapshot with the exact UE end point.

    Returns starts, ends, segment validity, path power and class per step.
    """
    n_snap = geom["points_m"].shape[0]
    snap = np.minimum(np.floor(times_s / dt_sense_s + 1e-9).astype(np.int64), n_snap - 1)
    points = geom["points_m"][snap].copy()  # [T, U, C, P, 4, 3]
    n_points = geom["n_points"][snap].astype(np.int64)  # [T, U, C, P]
    last = np.arange(points.shape[-2])[None, None, None, None, :] == (n_points[..., None] - 1)
    ue_end = np.broadcast_to(ue_pos[:, :, None, None, None, :], points.shape)
    points = np.where(last[..., None], ue_end, points)
    starts = points[..., :-1, :]
    ends = points[..., 1:, :]
    seg_valid = np.arange(points.shape[-2] - 1)[None, None, None, None, :] < (n_points[..., None] - 1)
    return {
        "snap": snap,
        "starts": np.nan_to_num(starts),
        "ends": np.nan_to_num(ends),
        "seg_valid": seg_valid,
        "power": geom["unblocked_power"][snap],
        "path_class": geom["path_class"][snap].astype(np.int64),
    }


def _reduce(loss_tucpb: Any, power: Any, cls: Any, xp: Any) -> dict[str, Any]:
    """Link quantities from per-(path, blocker) losses [dB] (``xp`` = numpy or torch)."""
    finite = xp.isfinite(loss_tucpb)
    capped = xp.where(finite, loss_tucpb, xp.full_like(loss_tucpb, math.inf))
    path_loss = capped.sum(-1)  # [T, U, C, P]
    gain = xp.where(xp.isfinite(path_loss), 10.0 ** (-xp.nan_to_num(path_loss, posinf=0.0) / 10.0), xp.zeros_like(path_loss))
    valid = cls >= 0
    p_un = xp.where(valid, power, xp.zeros_like(power))
    p_b = p_un * gain
    blocked = p_b.sum(-1)
    unblocked = p_un.sum(-1)
    los_mask = cls == CLASS_LOS
    los_power = xp.where(los_mask, p_un, xp.full_like(p_un, -1.0))
    if xp is np:
        los_idx = np.argmax(los_power, axis=-1)
        has_los = np.take_along_axis(los_power, los_idx[..., None], -1)[..., 0] >= 0.0
        los_loss = np.take_along_axis(path_loss, los_idx[..., None], -1)[..., 0]
        per_blocker = np.take_along_axis(capped, los_idx[..., None, None], -2)[..., 0, :]
        alt = np.where(valid & ~los_mask, p_b, -1.0).max(-1)
    else:
        los_idx = los_power.argmax(-1)
        has_los = los_power.gather(-1, los_idx[..., None])[..., 0] >= 0.0
        los_loss = path_loss.gather(-1, los_idx[..., None])[..., 0]
        idx = los_idx[..., None, None].expand(*los_idx.shape, 1, capped.shape[-1])
        per_blocker = capped.gather(-2, idx)[..., 0, :]
        alt = xp.where(valid & ~los_mask, p_b, xp.full_like(p_b, -1.0)).amax(-1)
    los_loss = xp.where(has_los, los_loss, xp.full_like(los_loss, math.inf))
    if xp is np:
        los_blocked = np.take_along_axis(p_b, los_idx[..., None], -1)[..., 0]
    else:
        los_blocked = p_b.gather(-1, los_idx[..., None])[..., 0]
    los_blocked = xp.where(has_los, los_blocked, xp.zeros_like(los_blocked))
    capped_pb = xp.where(xp.isfinite(per_blocker), per_blocker, xp.full_like(per_blocker, LOSS_CAP_DB))
    if xp is np:
        dominant = np.argmax(capped_pb, axis=-1)
        dom_loss = np.take_along_axis(capped_pb, dominant[..., None], -1)[..., 0]
    else:
        dominant = capped_pb.argmax(-1)
        dom_loss = capped_pb.gather(-1, dominant[..., None])[..., 0]
    dominant = xp.where(has_los & (dom_loss > 0.0), dominant, xp.full_like(dominant, -1))
    return {
        "blocked_power": blocked,
        "unblocked_power": unblocked,
        "los_loss_db": los_loss,
        "los_blocker": dominant,
        "best_alt_power": xp.where(alt >= 0.0, alt, xp.zeros_like(alt)),
        "los_blocked_power": los_blocked,
    }


def model_b_timeline(
    seg: dict[str, np.ndarray],
    blocker_pos: np.ndarray,
    blocker_size: np.ndarray,
    wavelength_m: float,
    *,
    device: str = "cuda",
    max_elements: int = 60_000_000,
) -> dict[str, np.ndarray]:
    """Batched GPU model B over (time × UE × cell × path × segment × blocker), chunked in time."""
    import torch

    from sim.comm.blockage_torch import screen_loss_db

    n_t = seg["starts"].shape[0]
    per_t = int(np.prod(seg["starts"].shape[1:-1])) * blocker_pos.shape[1]
    chunk = max(1, int(max_elements // max(per_t, 1)))
    out: dict[str, list[np.ndarray]] = {}
    size = torch.as_tensor(blocker_size, device=device, dtype=torch.float64)
    for lo in range(0, n_t, chunk):
        hi = min(n_t, lo + chunk)
        starts = torch.as_tensor(seg["starts"][lo:hi], device=device)[..., None, :]  # [t,U,C,P,S,1,3]
        ends = torch.as_tensor(seg["ends"][lo:hi], device=device)[..., None, :]
        centers = torch.as_tensor(blocker_pos[lo:hi], device=device)[:, None, None, None, None, :, :]
        loss = screen_loss_db(starts, ends, centers, size[:, 0], size[:, 1], size[:, 2], wavelength_m)
        valid = torch.as_tensor(seg["seg_valid"][lo:hi], device=device)[..., None]
        loss = torch.where(valid, loss, torch.zeros_like(loss))  # [t,U,C,P,S,B]
        inf_any = (~torch.isfinite(loss)).any(dim=-2)
        per_pb = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)).sum(dim=-2)
        per_pb = torch.where(inf_any, torch.full_like(per_pb, math.inf), per_pb)  # [t,U,C,P,B]
        power = torch.as_tensor(seg["power"][lo:hi], device=device)
        cls = torch.as_tensor(seg["path_class"][lo:hi], device=device)
        red = _reduce(per_pb, power, cls, torch)
        for key, value in red.items():
            out.setdefault(key, []).append(value.detach().cpu().numpy())
        del loss, per_pb, starts, ends, centers
    return {key: np.concatenate(parts, axis=0) for key, parts in out.items()}


def model_b_timeline_numpy(
    seg: dict[str, np.ndarray],
    blocker_pos: np.ndarray,
    blocker_size: np.ndarray,
    wavelength_m: float,
    steps: np.ndarray,
) -> dict[str, np.ndarray]:
    """NumPy reference on selected time steps (``path_blocker_loss`` per path and blocker)."""
    _, n_u, n_c, n_p, n_s, _ = seg["starts"].shape
    n_b = blocker_pos.shape[1]
    per_pb = np.zeros((len(steps), n_u, n_c, n_p, n_b))
    for i, k in enumerate(steps):
        for u in range(n_u):
            for c in range(n_c):
                for p in range(n_p):
                    segments = [
                        (seg["starts"][k, u, c, p, s], seg["ends"][k, u, c, p, s])
                        for s in range(n_s)
                        if seg["seg_valid"][k, u, c, p, s]
                    ]
                    for b in range(n_b):
                        if not segments:
                            continue
                        loss, _ = path_blocker_loss(segments, blocker_pos[k, b], *blocker_size[b], wavelength_m)
                        per_pb[i, u, c, p, b] = loss
    return _reduce(per_pb, seg["power"][steps], seg["path_class"][steps], np)


def path_loss_reference(per_blocker_db: list[float]) -> float:
    """Model-B sum over blockers, as in ``sim.comm.blockage.sum_blockage_db``."""
    return sum_blockage_db(per_blocker_db)
