"""Batched LoS blockage prediction from confirmed map-tracker tracks.

For every E2 report r (0.1 s), UE, cell and look-ahead tau in [0, 3] s the
predicted LoS model-B loss [dB] is computed once on the GPU. Every
horizon H is then a cut of the same tensor. The predictor is the v1 rule
(``xapp.predict``): constant velocity, lane/sidewalk tracks pinned to the
line, class-agnostic sizes (lane -> bus, sidewalk -> pedestrian, free ->
bus), blockers more than 0.5 L + 8 m from the segment ignored. The UE is
extrapolated with its known velocity from a noisy position fix.
``predict_numpy`` is the reference on top of ``xapp.predict.los_loss_db``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MODE = {"free": 0, "lane": 1, "sidewalk": 2}


def pack_tracks(tracks_by_snap: list[list[dict[str, Any]]], sizes: dict[str, tuple[float, float, float]]) -> dict[str, np.ndarray]:
    """Confirmed tracks per report, padded to ``K``: state [R, K, 5], size [R, K, 3], valid [R, K]."""
    n_r = len(tracks_by_snap)
    confirmed = [[row for row in rows if row["confirmed"]] for rows in tracks_by_snap]
    k = max(1, max((len(rows) for rows in confirmed), default=1))
    state = np.zeros((n_r, k, 5))
    size = np.ones((n_r, k, 3))
    valid = np.zeros((n_r, k), dtype=bool)
    for r, rows in enumerate(confirmed):
        for i, row in enumerate(rows):
            x, y, z, vx, vy = row["x_m"], row["y_m"], row["z_m"], row["vx_mps"], row["vy_mps"]
            mode = row.get("mode", "free")
            if mode in ("lane", "sidewalk") and row.get("line_y_m") is not None:
                y, vy = float(row["line_y_m"]), 0.0
            state[r, i] = (x, y, z, vx, vy)
            if mode == "sidewalk":
                size[r, i] = sizes["pedestrian"]
            else:
                size[r, i] = sizes["bus"]
            valid[r, i] = True
    return {"state": state, "size": size, "valid": valid}


def ue_fixes(ue_pos: np.ndarray, ue_vel: np.ndarray, sigma_m: float, seed: int) -> np.ndarray:
    """Noisy horizontal UE fix per report [R, U, 3]; height exact. Deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_m, size=ue_pos.shape)
    noise[..., 2] = 0.0
    return ue_pos + noise


def predict_torch(
    tracks: dict[str, np.ndarray],
    ue_fix: np.ndarray,
    ue_vel: np.ndarray,
    oru_pos: np.ndarray,
    taus_s: np.ndarray,
    wavelength_m: float,
    *,
    device: str = "cuda",
    max_elements: int = 40_000_000,
) -> np.ndarray:
    """Predicted LoS loss [dB], shape [R, U, C, tau]; ``inf`` = fully blocked."""
    import torch

    from sim.comm.blockage_torch import screen_loss_db

    n_r, n_k = tracks["valid"].shape
    n_u, n_c, n_t = ue_fix.shape[1], oru_pos.shape[0], len(taus_s)
    per_r = n_u * n_c * n_t * n_k
    chunk = max(1, max_elements // max(per_r, 1))
    tau = torch.as_tensor(taus_s, device=device, dtype=torch.float64)
    oru = torch.as_tensor(oru_pos, device=device, dtype=torch.float64)
    out = []
    for lo in range(0, n_r, chunk):
        hi = min(n_r, lo + chunk)
        st = torch.as_tensor(tracks["state"][lo:hi], device=device)
        sz = torch.as_tensor(tracks["size"][lo:hi], device=device)
        valid = torch.as_tensor(tracks["valid"][lo:hi], device=device)
        fix = torch.as_tensor(ue_fix[lo:hi], device=device)
        vel = torch.as_tensor(ue_vel[lo:hi], device=device)
        # positions: blockers [r, tau, k, 3], UE [r, u, tau, 3]
        bpos = torch.stack(
            (
                st[:, None, :, 0] + tau[None, :, None] * st[:, None, :, 3],
                st[:, None, :, 1] + tau[None, :, None] * st[:, None, :, 4],
                st[:, None, :, 2].expand(-1, n_t, -1),
            ),
            dim=-1,
        )
        upos = fix[:, :, None, :] + tau[None, None, :, None] * vel[:, :, None, :]
        # segment ORU -> UE: [r, u, c, tau, 1, 3]; blocker [r, 1, 1, tau, k, 3]
        start = oru[None, None, :, None, None, :].expand(hi - lo, n_u, n_c, n_t, 1, 3)
        end = upos[:, :, None, :, None, :].expand(hi - lo, n_u, n_c, n_t, 1, 3)
        center = bpos[:, None, None, :, :, :]
        length = sz[:, None, None, None, :, 0]
        loss = screen_loss_db(start, end, center, length, sz[:, None, None, None, :, 1], sz[:, None, None, None, :, 2], wavelength_m)
        # prefilter of xapp.predict.los_loss_db: horizontal distance to the clamped segment
        span = (end - start)[..., :2]
        rel = (center - start)[..., :2]
        l2 = (span * span).sum(-1)
        along = torch.where(l2 > 1e-12, (rel * span).sum(-1) / l2.clamp_min(1e-12), torch.zeros_like(l2))
        closest = start[..., :2] + along.clamp(0.0, 1.0)[..., None] * span
        far = torch.linalg.norm(center[..., :2] - closest, dim=-1) > 0.5 * length + 8.0
        far = far & (l2.sqrt() > 1e-6)
        keep = valid[:, None, None, None, :] & ~far
        loss = torch.where(keep, loss, torch.zeros_like(loss))
        inf_any = (~torch.isfinite(loss)).any(-1)
        total = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)).sum(-1)
        total = torch.where(inf_any, torch.full_like(total, math.inf), total)
        out.append(total.cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_numpy(
    tracks: dict[str, np.ndarray],
    ue_fix: np.ndarray,
    ue_vel: np.ndarray,
    oru_pos: np.ndarray,
    taus_s: np.ndarray,
    wavelength_m: float,
    reports: list[int],
) -> np.ndarray:
    """Reference for selected reports via ``xapp.predict.los_loss_db``."""
    from xapp.predict import los_loss_db

    n_u, n_c = ue_fix.shape[1], oru_pos.shape[0]
    out = np.zeros((len(reports), n_u, n_c, len(taus_s)))
    for i, r in enumerate(reports):
        for t_i, tau in enumerate(taus_s):
            blockers = []
            for k in np.flatnonzero(tracks["valid"][r]):
                x, y, z, vx, vy = tracks["state"][r, k]
                blockers.append((np.array([x + tau * vx, y + tau * vy, z]), tuple(tracks["size"][r, k])))
            for u in range(n_u):
                ue = ue_fix[r, u] + tau * ue_vel[r, u]
                for c in range(n_c):
                    out[i, u, c, t_i] = los_loss_db(oru_pos[c], ue, blockers, wavelength_m)
    return out


def trigger_table(loss: np.ndarray, taus_s: np.ndarray, horizon_s: float, block_db: float, clear_db: float) -> dict[str, np.ndarray]:
    """Per report, UE and serving cell c: is a blockage of c predicted within H while the other cell stays clear?

    ``loss`` is [R, U, C=2, tau]. The window starts at the first tau <= H with
    loss >= ``block_db`` and runs while loss >= ``clear_db`` (up to H). The
    other cell must stay below ``clear_db`` over that window. Returns
    ``trigger`` [R, U, C], the predicted start and end [s] relative to the
    report (``nan`` if none).
    """
    inside = taus_s <= horizon_s + 1e-9
    series = loss[..., inside]
    n_t = series.shape[-1]
    blocked = series >= block_db
    has = blocked.any(-1)
    start = np.where(has, blocked.argmax(-1), n_t)
    idx = np.arange(n_t)
    # end: last index of the run of loss >= clear_db that starts at ``start``
    above = series >= clear_db
    after = idx[None, None, None, :] >= start[..., None]
    breaks = after & ~above
    first_break = np.where(breaks.any(-1), breaks.argmax(-1), n_t)
    in_window = after & (idx[None, None, None, :] < first_break[..., None])
    other = series[:, :, ::-1, :]
    other_max = np.where(in_window, other, -np.inf).max(-1)
    trigger = has & (other_max < clear_db)
    taus_in = taus_s[inside]
    start_s = np.where(has, taus_in[np.minimum(start, n_t - 1)], np.nan)
    end_s = np.where(has, taus_in[np.clip(first_break - 1, 0, n_t - 1)], np.nan)
    return {"trigger": trigger, "start_s": start_s, "end_s": end_s}
