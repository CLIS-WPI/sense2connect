"""Model-B loss [dB] of every path at the 0.1 s epochs (as paper 1, per path instead of per link).

Uses the paper-1 segment construction (xapp.timeline.held_segments: the cached
snapshot geometry with the exact UE end point) and the paper-1 GPU model B
(sim.comm.blockage_torch.screen_loss_db), both imported read-only; the loss
of a path is the sum over blockers of the per-blocker loss summed over the
path's segments, infinite if any blocker fully blocks it (as in
xapp.timeline._reduce). Blocker poses from xapp.timeline.actor_tracks.
"""

from __future__ import annotations

import math

import numpy as np


def path_losses_db(geom: dict[str, np.ndarray], scenario: dict, wavelength_m: float, *, device: str = "cuda", chunk: int = 50) -> dict[str, np.ndarray]:
    """loss_db [T, U, C, P] (inf = fully blocked; 0 for empty path slots) at t = k * 0.1 s, k < n_snapshots."""
    import torch

    from sim.comm.blockage_torch import screen_loss_db
    from xapp.timeline import actor_tracks, held_segments

    n = geom["points_m"].shape[0]
    times = np.arange(n) * 0.1
    act = actor_tracks(scenario, times)
    seg = held_segments(geom, times, act["ue_position_m"], 0.1)
    size = torch.as_tensor(act["blocker_size_m"], device=device, dtype=torch.float64)
    out, dom = [], []
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        starts = torch.as_tensor(seg["starts"][lo:hi], device=device)[..., None, :]
        ends = torch.as_tensor(seg["ends"][lo:hi], device=device)[..., None, :]
        centers = torch.as_tensor(act["blocker_position_m"][lo:hi], device=device)[:, None, None, None, None, :, :]
        loss = screen_loss_db(starts, ends, centers, size[:, 0], size[:, 1], size[:, 2], wavelength_m)
        valid = torch.as_tensor(seg["seg_valid"][lo:hi], device=device)[..., None]
        loss = torch.where(valid, loss, torch.zeros_like(loss))  # [t, U, C, P, S, B]
        inf_any = (~torch.isfinite(loss)).any(dim=-2)
        per_pb = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)).sum(dim=-2)
        per_pb = torch.where(inf_any, torch.full_like(per_pb, math.inf), per_pb)  # [t, U, C, P, B]
        out.append(per_pb.sum(-1).cpu().numpy())
        capped = torch.where(torch.isfinite(per_pb), per_pb, torch.full_like(per_pb, 1e9))
        dom.append(capped.argmax(-1).cpu().numpy())
    return {"loss_db": np.concatenate(out, 0), "dominant_blocker": np.concatenate(dom, 0), "ue_position_m": act["ue_position_m"],
            "ue_velocity_mps": act["ue_velocity_mps"], "blocker_position_m": act["blocker_position_m"], "blocker_size_m": act["blocker_size_m"]}


def diffracted_los(oru: np.ndarray, ue: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Shortest detour point q of a LoS O-RU -> UE around a box blocker (variant b, "biased", estimator only).

    The box (length along x, width along y, height; centre at half height) blocks
    the segment. Candidates: over the top edge (the point of the segment's vertical
    plane nearest the blocker centre, lifted to the box height) and around either
    side (the same point moved horizontally, perpendicular to the segment, just
    outside the box footprint). Returns q [..., 3] minimising |oru - q| + |q - ue|
    (the knife-edge picture behind model B; a modelling choice for the estimator's
    biased-LoS variant, documented in results/P2/report.md).
    """
    d = ue - oru
    dh = d[..., :2]
    L2 = np.maximum((dh * dh).sum(-1), 1e-12)
    s = np.clip(((center[..., :2] - oru[..., :2]) * dh).sum(-1) / L2, 0.0, 1.0)
    base = oru + s[..., None] * d  # point on the segment nearest the blocker (horizontally)
    top = base.copy()
    top[..., 2] = np.maximum(base[..., 2], 2.0 * center[..., 2] + 0.01)  # box top = 2 x centre height
    perp = np.stack([-dh[..., 1], dh[..., 0]], -1) / np.sqrt(L2)[..., None]
    half = 0.5 * (size[..., 0] * np.abs(perp[..., 0]) + size[..., 1] * np.abs(perp[..., 1])) + 0.01
    off = ((center[..., :2] - base[..., :2]) * perp).sum(-1)
    cands = [top]
    for sign in (1.0, -1.0):
        q = base.copy()
        q[..., :2] = base[..., :2] + (off + sign * half)[..., None] * perp
        cands.append(q)
    lengths = np.stack([np.linalg.norm(q - oru, axis=-1) + np.linalg.norm(ue - q, axis=-1) for q in cands], -1)
    k = np.argmin(lengths, -1)
    return np.take_along_axis(np.stack(cands, -2), k[..., None, None].repeat(3, -1), -2)[..., 0, :]
