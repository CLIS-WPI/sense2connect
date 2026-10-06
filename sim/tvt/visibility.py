"""Path visibility prediction from blocker-track posteriors (TVT T3), torch on the GPU.

A source s of O-RU c is the LoS or a virtual-anchor chain of up to two specular bounces on
known planes (normals/anchors from sim.positioning.geometry.bounce_planes). For a UE position
p its polyline G_{c,s}(p) is rebuilt with the image method (bounce points on the planes).
Visibility q_{c,s} = P{ model-B loss of G_{c,s}(p) from all blockers < threshold } (10 dB, the
paper-1/2 event level), evaluated with K joint Monte Carlo samples of the predicted UE
position and of the blocker positions N(mean, cov) extrapolated with constant velocity over
the look-ahead. Model B as in the frozen code (sim.comm.blockage_torch.screen_loss_db; loss of
a path = sum over blockers of the per-blocker loss summed over its segments, inf if any
blocker fully blocks it). A prior blockage probability for regions the radar does not cover
multiplies q: q <- q (1 - p_prior), p_prior per (O-RU, LoS/NLoS) from the tuning seeds.
"""

from __future__ import annotations

import math

import numpy as np


def polyline(oru: np.ndarray, ue: np.ndarray, normals: np.ndarray, anchors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Points [..., 4, 3] (O-RU, bounce 1, bounce 2, UE; unused bounce slots repeat the UE) of the specular path to ``ue``."""
    from sim.positioning.geometry import mirror

    v0, v1 = valid[..., 0], valid[..., 1]
    n0, n1 = normals[..., 0, :], normals[..., 1, :]
    q0, q1 = anchors[..., 0, :], anchors[..., 1, :]
    p1 = np.where(v1[..., None], mirror(ue, n1, q1), ue)       # UE mirrored across the last plane
    p01 = np.where(v0[..., None], mirror(p1, n0, q0), p1)      # ... and across the first
    # bounce 1: on plane 0 along O -> p01
    den0 = ((p01 - oru) * n0).sum(-1)
    t0 = np.where(np.abs(den0) > 1e-12, ((q0 - oru) * n0).sum(-1) / np.where(np.abs(den0) > 1e-12, den0, 1.0), 0.0)
    b1 = oru + t0[..., None] * (p01 - oru)
    # bounce 2: on plane 1 along b1 -> p1
    den1 = ((p1 - b1) * n1).sum(-1)
    t1 = np.where(np.abs(den1) > 1e-12, ((q1 - b1) * n1).sum(-1) / np.where(np.abs(den1) > 1e-12, den1, 1.0), 0.0)
    b2 = b1 + t1[..., None] * (p1 - b1)
    b1 = np.where(v0[..., None], b1, ue)
    b2 = np.where(v1[..., None], b2, ue)
    shape = np.broadcast_shapes(oru.shape, ue.shape, b1.shape, b2.shape)
    return np.stack([np.broadcast_to(x, shape) for x in (oru, b1, b2, ue)], axis=-2)


def path_loss_samples(points, blocker_pos, blocker_size, blocker_valid, wavelength_m: float, *, device: str = "cuda"):
    """Model-B loss [dB] [..., P] for polylines points [..., P, 4, 3] and blockers [..., B, 3] ("..." without the path axis).

    Segment k from points[k] to points[k+1]; zero-length segments contribute 0.
    """
    import torch

    from sim.comm.blockage_torch import screen_loss_db

    pts = torch.as_tensor(points, device=device, dtype=torch.float64)
    starts = pts[..., :-1, :][..., None, :]  # [..., P, S, 1, 3]
    ends = pts[..., 1:, :][..., None, :]
    c = torch.as_tensor(blocker_pos, device=device, dtype=torch.float64)[..., None, None, :, :]  # [..., 1, 1, B, 3]
    sz = torch.as_tensor(blocker_size, device=device, dtype=torch.float64)[..., None, None, :, :]
    loss = screen_loss_db(starts, ends, c, sz[..., 0], sz[..., 1], sz[..., 2], wavelength_m)  # [..., P, S, B]
    seglen = torch.linalg.norm(ends - starts, dim=-1)  # [..., P, S, 1]
    ok = (seglen > 1e-6) & torch.as_tensor(np.array(blocker_valid), device=device)[..., None, None, :]
    loss = torch.where(ok, loss, torch.zeros_like(loss))
    inf_any = (~torch.isfinite(loss)).any(dim=-2)  # [..., P, B]
    per_pb = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)).sum(dim=-2)
    per_pb = torch.where(inf_any, torch.full_like(per_pb, math.inf), per_pb)
    inf_p = (~torch.isfinite(per_pb)).any(-1)
    tot = torch.where(torch.isfinite(per_pb), per_pb, torch.zeros_like(per_pb)).sum(-1)
    return torch.where(inf_p, torch.full_like(tot, math.inf), tot), per_pb


def sample_blockers(mean: np.ndarray, cov: np.ndarray, horizon_s: float, K: int, rng: np.random.Generator) -> np.ndarray:
    """K joint samples of blocker positions [K, ..., B, 3] after ``horizon_s`` (constant velocity; z kept)."""
    sh = mean.shape[:-1]
    z = rng.standard_normal((K,) + sh + (5,))
    # Cholesky with a small jitter; zero covariance -> exact mean
    c = cov + 1e-12 * np.eye(5)
    L = np.linalg.cholesky(c)
    s = mean[None] + np.einsum("...ij,k...j->k...i", L, z)
    pos = s[..., :3].copy()
    pos[..., 0] += s[..., 3] * horizon_s
    pos[..., 1] += s[..., 4] * horizon_s
    return pos


def visibility(oru, ue_samples, normals, anchors, valid_planes, path_valid, blk_samples, blk_size, blk_valid, wavelength_m: float, thr_db: float = 10.0,
               *, device: str = "cuda"):
    """q [..., U, C, P] = fraction of the K joint samples in which the path is not blocked (loss < thr_db).

    oru [..., C, 3]; ue_samples [K, ..., U, 3]; planes [..., U, C, P, 2, (3)]; blk_samples [K, ..., B, 3]; blk_size [..., B, 3]; blk_valid [..., B].
    Returns q (numpy) and the per-sample loss tensor is not kept.
    """
    import torch

    K = ue_samples.shape[0]
    pts = polyline(oru[None, ..., None, :, None, :], ue_samples[..., :, None, None, :], normals[None], anchors[None], valid_planes[None])  # [K, ..., U, C, P, 4, 3]
    # path_loss_samples broadcasts blockers over the polyline dims except the path axis: [K, ..., U, C, B, 3]
    bp = blk_samples[..., None, None, :, :]
    bs = blk_size[None, ..., None, None, :, :]
    bv = np.broadcast_to(blk_valid[None, ..., None, None, :], bp.shape[:-1])
    loss, _ = path_loss_samples(pts, bp, bs, bv, wavelength_m, device=device)  # [K, ..., U, C, P]
    vis = (loss < thr_db).to(torch.float64).mean(0)
    pv = torch.as_tensor(path_valid, device=device)
    return torch.where(pv, vis, torch.zeros_like(vis)).cpu().numpy()
