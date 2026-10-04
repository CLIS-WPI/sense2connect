"""LoS blockage prediction from map-constrained tracks and model B."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.comm.blockage import path_blocker_loss, sum_blockage_db


def predict_positions(track: dict[str, Any], times_s: np.ndarray) -> np.ndarray:
    """Constant-velocity positions [N, 3]. Lane/sidewalk tracks stay on the line."""
    origin = np.array([track["x_m"], track["y_m"], track["z_m"]], dtype=np.float64)
    velocity = np.array([track["vx_mps"], track["vy_mps"], 0.0], dtype=np.float64)
    if track.get("mode") in ("lane", "sidewalk") and track.get("line_y_m") is not None:
        velocity[1] = 0.0
        origin[1] = float(track["line_y_m"])
    return origin[None, :] + times_s[:, None] * velocity[None, :]


def los_loss_db(
    oru_m: np.ndarray,
    ue_m: np.ndarray,
    blockers: list[tuple[np.ndarray, tuple[float, float, float]]],
    wavelength_m: float,
) -> float:
    """Model-B loss [dB] of the LoS segment through every predicted blocker."""
    losses = []
    start = np.asarray(oru_m, dtype=np.float64)
    end = np.asarray(ue_m, dtype=np.float64)
    span = end - start
    length = float(np.linalg.norm(span[:2]))
    for center, size in blockers:
        point = np.asarray(center, dtype=np.float64)
        if length > 1e-6:
            along = float(np.dot(point[:2] - start[:2], span[:2]) / (length * length))
            closest = start[:2] + max(0.0, min(1.0, along)) * span[:2]
            if float(np.linalg.norm(point[:2] - closest)) > 0.5 * size[0] + 8.0:
                continue
        loss, _ = path_blocker_loss([(start, end)], point, size[0], size[1], size[2], wavelength_m)
        losses.append(loss)
    return sum_blockage_db(losses) if losses else 0.0


def blockage_window(
    losses_db: np.ndarray,
    times_s: np.ndarray,
    block_db: float,
    clear_db: float,
) -> tuple[float | None, float | None]:
    """First interval where loss reaches ``block_db``, ending when it falls below ``clear_db``."""
    blocked = np.asarray(losses_db, dtype=np.float64) >= float(block_db)
    if not np.any(blocked):
        return None, None
    start = int(np.argmax(blocked))
    end = start
    while end + 1 < len(losses_db) and float(losses_db[end + 1]) >= float(clear_db):
        end += 1
    t0 = float(times_s[start])
    t1 = float(times_s[end])
    return t0, max(t1, t0)


def predict_link(
    oru_m: np.ndarray,
    ue_m: np.ndarray,
    tracks: list[dict[str, Any]],
    sizes: dict[str, tuple[float, float, float]],
    wavelength_m: float,
    horizon_s: float,
    dt_s: float,
    block_db: float,
    clear_db: float,
) -> dict[str, float | None]:
    """Predicted LoS blockage start/end [s] relative to now, within ``horizon_s``."""
    step = max(float(dt_s), 0.2)
    steps = int(round(horizon_s / step)) + 1
    times = np.arange(steps, dtype=np.float64) * step
    losses = np.zeros(steps, dtype=np.float64)
    for index, t_s in enumerate(times):
        blockers = []
        for track in tracks:
            if not track.get("confirmed"):
                continue
            pos = predict_positions(track, np.array([t_s]))[0]
            size = sizes.get("vehicle", (8.0, 2.5, 3.0))
            if track.get("mode") == "sidewalk":
                size = sizes.get("pedestrian", (0.5, 0.5, 1.75))
            elif track.get("mode") == "lane":
                size = sizes.get("bus", (12.0, 2.5, 3.2))
            blockers.append((pos, size))
        losses[index] = los_loss_db(oru_m, ue_m, blockers, wavelength_m)
    start, end = blockage_window(losses, times, block_db, clear_db)
    return {"start_s": start, "end_s": end, "clear": start is None}
