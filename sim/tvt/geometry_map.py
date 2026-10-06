"""Uncertain map (TVT T1): derivatives of path delay and angles w.r.t. surface offsets.

Each known reflecting surface s (configs/p2.yaml known_planes: axis-aligned planes) may be
displaced along its normal by an unknown offset d_s [m]. Moving the plane of bounce i by d
along its unit normal n_i moves its anchor q_i -> q_i + d n_i, so every mirror
M_i(x) = x - 2 ((x - q_i) . n_i) n_i shifts by dM_i/dd = 2 n_i (independent of x) and has
the linear part R_i = I - 2 n_i n_i^T. With the image method of sim/positioning/geometry.py
(frozen; same conventions):
- O-RU image in path order  O' = M_1(M_0(O)):  dO'/dd_0 = R_1 2 n_0,  dO'/dd_1 = 2 n_1;
- UE image in reverse order P' = M_0(M_1(p)):  dP'/dd_1 = R_0 2 n_1,  dP'/dd_0 = 2 n_0;
- tau = |p - O'| / c:  dtau/dd_i = -(p - O')^T dO'/dd_i / (|p - O'| c);
- u = (P' - O) / |P' - O|:  du/dd_i = (I - u u^T) dP'/dd_i / |P' - O|, az/el by the chain rule.
A single bounce uses slot 0 only. ``surface_ids`` maps every bounce to a global surface index
(facades, i.e. the vertical x- and y-planes by default; -1 = not an uncertain surface).
"""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0


def surfaces(known: dict[str, list[float]], axes: str = "xy") -> list[tuple[int, float]]:
    """Global list of uncertain surfaces (axis index, offset [m]) in config order; default: facades (x, y planes)."""
    out = []
    for k, name in enumerate("xyz"):
        if name in axes:
            out += [(k, float(v)) for v in known.get(name, [])]
    return out


def surface_ids(normals: np.ndarray, anchors: np.ndarray, valid: np.ndarray, surf: list[tuple[int, float]], tol_m: float = 1e-6) -> np.ndarray:
    """Surface index per bounce slot [..., 2] (-1 if absent or not an uncertain surface).

    ``normals``/``anchors`` from sim.positioning.geometry.bounce_planes with the known planes (snapped exactly).
    """
    ids = np.full(valid.shape, -1, dtype=np.int64)
    ax = np.argmax(np.abs(normals), axis=-1)
    aligned = np.abs(normals).max(-1) > 1.0 - 1e-9
    for j, (k, off) in enumerate(surf):
        hit = valid & aligned & (ax == k) & (np.abs(anchors[..., k] - off) <= tol_m)
        ids = np.where(hit, j, ids)
    return ids


def _images(oru: np.ndarray, ue: np.ndarray, normals: np.ndarray, anchors: np.ndarray, valid: np.ndarray):
    """O-RU image O' (path order), UE image P' (reverse order) and the reflections R_0, R_1 (identity if absent)."""
    from sim.positioning.geometry import mirror, reflect_matrix

    v0, v1 = valid[..., 0], valid[..., 1]
    n0, n1 = normals[..., 0, :], normals[..., 1, :]
    q0, q1 = anchors[..., 0, :], anchors[..., 1, :]
    R0, R1 = reflect_matrix(n0), reflect_matrix(n1)
    eye = np.broadcast_to(np.eye(3), R0.shape)
    R0 = np.where(v0[..., None, None], R0, eye)
    R1 = np.where(v1[..., None, None], R1, eye)
    o = np.where(v0[..., None], mirror(oru, n0, q0), oru)
    o = np.where(v1[..., None], mirror(o, n1, q1), o)
    p_ = np.where(v1[..., None], mirror(ue, n1, q1), ue)
    p_ = np.where(v0[..., None], mirror(p_, n0, q0), p_)
    return o, p_, R0, R1


def _image_jacobian(oru: np.ndarray, ue: np.ndarray, o: np.ndarray, p_: np.ndarray, dO: np.ndarray, dP: np.ndarray) -> dict[str, np.ndarray]:
    """d(tau, az, el) for image displacements dO, dP [..., K, 3] (K parameters): dtau [..., K] (s/m), daz, del [..., K] (rad/m)."""
    eye = np.broadcast_to(np.eye(3), o.shape[:-1] + (3, 3))
    dv = ue - o
    dist = np.maximum(np.linalg.norm(dv, axis=-1), 1e-12)
    dtau = -(dv[..., None, :] * dO).sum(-1) / dist[..., None] / C0
    w = p_ - oru
    rw = np.maximum(np.linalg.norm(w, axis=-1), 1e-12)
    u = w / rw[..., None]
    proj = eye - u[..., :, None] * u[..., None, :]
    du = (proj[..., None, :, :] @ dP[..., :, :, None])[..., 0] / rw[..., None, None]  # [..., K, 3]
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    rho2 = np.maximum(ux * ux + uy * uy, 1e-18)
    g_az = np.stack([-uy / rho2, ux / rho2, np.zeros_like(ux)], axis=-1)
    g_el = np.stack([np.zeros_like(ux), np.zeros_like(ux), 1.0 / np.sqrt(np.maximum(1.0 - uz * uz, 1e-18))], axis=-1)
    daz = (du * g_az[..., None, :]).sum(-1)
    del_ = (du * g_el[..., None, :]).sum(-1)
    return {"dtau": dtau, "daz": daz, "del": del_}


def offset_derivatives(oru: np.ndarray, ue: np.ndarray, normals: np.ndarray, anchors: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """d(tau, az, el)/d d_i for the two bounce slots: dtau [..., 2] (s/m), daz [..., 2], del [..., 2] (rad/m); 0 for absent slots."""
    v0, v1 = valid[..., 0], valid[..., 1]
    n0, n1 = normals[..., 0, :], normals[..., 1, :]
    o, p_, R0, R1 = _images(oru, ue, normals, anchors, valid)
    dO = np.stack([(R1 @ (2.0 * n0)[..., None])[..., 0], 2.0 * n1], axis=-2)  # [..., 2, 3]
    dP = np.stack([2.0 * n0, (R0 @ (2.0 * n1)[..., None])[..., 0]], axis=-2)
    slot = np.stack([v0, v1], axis=-1)
    dO = np.where(slot[..., None], dO, 0.0)
    dP = np.where(slot[..., None], dP, 0.0)
    return _image_jacobian(oru, ue, o, p_, dO, dP)


def va_derivatives(oru: np.ndarray, ue: np.ndarray, normals: np.ndarray, anchors: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """d(tau, az, el)/d delta for a free 3-D displacement delta [m] of the virtual anchor (O-RU image) of every NLoS path.

    The path is the isometry F = M_last o ... o M_first (linear part R = R_1 R_0, translation t):
    O' = F(O) = R O + t and P' = F^-1(p) = R^T (p - t). The VA displacement shifts t by delta:
    dO'/d delta = I, dP'/d delta = -R^T. The surface offset d_i of offset_derivatives is the special
    case delta = 2 d n_i along the plane normal (same images, same chain rule). Returns dtau [..., 3]
    (s/m), daz, del [..., 3] (rad/m); 0 for LoS paths (no bounce).
    """
    o, p_, R0, R1 = _images(oru, ue, normals, anchors, valid)
    R = R1 @ R0
    nlos = valid.any(-1)
    eye = np.broadcast_to(np.eye(3), R.shape)
    dO = np.where(nlos[..., None, None], eye, 0.0)  # rows: parameter k, columns: xyz
    dP = np.where(nlos[..., None, None], -R, 0.0)  # (-R^T)^T = -R: row k = d P' / d delta_k
    return _image_jacobian(oru, ue, o, p_, dO, dP)
