"""Path geometry and analytic position derivatives (image method), NumPy.

Uplink: the UE at p transmits; O-RU c receives. A cached path is the
polyline O-RU -> bounce points -> UE (paper-1 comm geometry). Specular
bounces on planar surfaces: the plane of bounce i has the normal along
(d_out - d_in) (unit directions of the incoming and outgoing segments
in the O-RU -> UE order) and contains the bounce point. Then
- delay   tau(p) = |p - O'| / c,  O' = O mirrored across the planes in
  path order (first bounce first);
- AoA at the O-RU: unit vector u(p) = (P'(p) - O) / |P'(p) - O|,
  P'(p) = UE mirrored across the planes in reverse order.
Both are exact for specular paths while the bounce points stay on the
planes. The UE height is known; derivatives are w.r.t. (x, y) [m].
Angles: az = atan2(u_y, u_x), el = asin(u_z) [rad].
"""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0


def reflect_matrix(n: np.ndarray) -> np.ndarray:
    """Linear part of the mirror across a plane with unit normal n: I - 2 n n^T ([..., 3, 3])."""
    eye = np.broadcast_to(np.eye(3), n.shape[:-1] + (3, 3))
    return eye - 2.0 * n[..., :, None] * n[..., None, :]


def mirror(x: np.ndarray, n: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mirror point(s) x across the plane through q with unit normal n."""
    return x - 2.0 * ((x - q) * n).sum(-1, keepdims=True) * n


def bounce_planes(points: np.ndarray, n_points: np.ndarray, known: dict[str, list[float]] | None = None,
                  snap_tol_m: float = 0.005) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planes (normal [..., 2, 3], point [..., 2, 3], valid [..., 2]) of up to two bounces.

    ``points`` [..., 4, 3] is O-RU, bounce 1, bounce 2, UE (NaN padded),
    ``n_points`` [...] the number of valid points (2 = LoS). With ``known``
    (axis -> plane offsets [m], configs/p2.yaml known_planes) a normal within
    1 deg of a coordinate axis is snapped to that axis and the anchor
    coordinate to the nearest known offset within ``snap_tol_m``.
    """
    shp = points.shape[:-2]
    normals = np.zeros(shp + (2, 3))
    anchors = np.zeros(shp + (2, 3))
    valid = np.zeros(shp + (2,), dtype=bool)
    pts = np.nan_to_num(points)
    for i in range(2):
        has = n_points >= i + 3
        prev, cur, nxt = pts[..., i, :], pts[..., i + 1, :], pts[..., i + 2, :]
        d_in = cur - prev
        d_out = nxt - cur
        d_in = d_in / np.maximum(np.linalg.norm(d_in, axis=-1, keepdims=True), 1e-12)
        d_out = d_out / np.maximum(np.linalg.norm(d_out, axis=-1, keepdims=True), 1e-12)
        nv = d_out - d_in
        nn = np.linalg.norm(nv, axis=-1, keepdims=True)
        ok = has & (nn[..., 0] > 1e-9)
        nrm = np.where(ok[..., None], nv / np.maximum(nn, 1e-12), 0.0)
        anc = cur.copy()
        if known is not None:
            ax = np.argmax(np.abs(nrm), axis=-1)
            aligned = np.abs(nrm).max(-1) > np.cos(np.radians(1.0))
            for k, name in enumerate("xyz"):
                offs = np.asarray(known.get(name, []), dtype=np.float64)
                sel = ok & aligned & (ax == k)
                if not sel.any():
                    continue
                unit = np.zeros(3)
                unit[k] = 1.0
                nrm = np.where(sel[..., None], unit * np.sign(nrm[..., k:k + 1]), nrm)
                if offs.size:
                    dist = np.abs(anc[..., k][..., None] - offs)
                    j = np.argmin(dist, axis=-1)
                    near = sel & (np.take_along_axis(dist, j[..., None], -1)[..., 0] <= snap_tol_m)
                    anc[..., k] = np.where(near, offs[j], anc[..., k])
        normals[..., i, :] = nrm
        anchors[..., i, :] = anc
        valid[..., i] = ok
    return normals, anchors, valid


def path_geometry(oru: np.ndarray, ue: np.ndarray, normals: np.ndarray, anchors: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Delay [s], AoA (az, el) [rad] and their (x, y) Jacobians for paths with up to two bounces.

    Shapes: oru, ue [..., 3]; normals, anchors [..., 2, 3]; valid [..., 2].
    Returns tau [...], az [...], el [...], u [..., 3], dtau [..., 2], daz [..., 2], del_ [..., 2].
    """
    # image of the O-RU in path order
    img_o = oru.copy()
    for i in range(2):
        m = valid[..., i][..., None]
        img_o = np.where(m, mirror(img_o, normals[..., i, :], anchors[..., i, :]), img_o)
    # image of the UE in reverse order, and the chain of linear maps
    img_p = ue.copy()
    lin = np.broadcast_to(np.eye(3), ue.shape[:-1] + (3, 3)).copy()
    for i in (1, 0):
        m = valid[..., i]
        img_p = np.where(m[..., None], mirror(img_p, normals[..., i, :], anchors[..., i, :]), img_p)
        r = reflect_matrix(normals[..., i, :])
        lin = np.where(m[..., None, None], r @ lin, lin)
    dv = ue - img_o
    dist = np.linalg.norm(dv, axis=-1)
    tau = dist / C0
    dtau = (dv / np.maximum(dist, 1e-12)[..., None])[..., :2] / C0
    w = img_p - oru
    rw = np.linalg.norm(w, axis=-1)
    u = w / np.maximum(rw, 1e-12)[..., None]
    # du/dp = (I - u u^T) / |w| * lin  (columns x, y)
    proj = np.broadcast_to(np.eye(3), u.shape[:-1] + (3, 3)) - u[..., :, None] * u[..., None, :]
    du = (proj @ lin)[..., :, :2] / np.maximum(rw, 1e-12)[..., None, None]  # [..., 3, 2]
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    rho2 = np.maximum(ux * ux + uy * uy, 1e-18)
    az = np.arctan2(uy, ux)
    el = np.arcsin(np.clip(uz, -1.0, 1.0))
    g_az = np.stack([-uy / rho2, ux / rho2, np.zeros_like(ux)], axis=-1)  # d az / d u
    g_el = np.stack([np.zeros_like(ux), np.zeros_like(ux), 1.0 / np.sqrt(np.maximum(1.0 - uz * uz, 1e-18))], axis=-1)
    daz = (g_az[..., None, :] @ du)[..., 0, :]
    del_ = (g_el[..., None, :] @ du)[..., 0, :]
    return {"tau": tau, "az": az, "el": el, "u": u, "dtau": dtau, "daz": daz, "del": del_}


def polyline_length(points: np.ndarray, n_points: np.ndarray) -> np.ndarray:
    """Length [m] of the cached path polylines (for the image-method consistency check)."""
    pts = np.nan_to_num(points)
    seg = np.linalg.norm(pts[..., 1:, :] - pts[..., :-1, :], axis=-1)  # [..., 3]
    k = np.arange(seg.shape[-1])
    return np.where(k < (n_points[..., None] - 1), seg, 0.0).sum(-1)


def first_segment_direction(points: np.ndarray) -> np.ndarray:
    """Unit direction from the O-RU to the first point after it (cached AoA direction)."""
    pts = np.nan_to_num(points)
    d = pts[..., 1, :] - pts[..., 0, :]
    return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
