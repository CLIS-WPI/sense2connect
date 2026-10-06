"""Map sources for the multipath tracker (TVT T4): LoS and virtual-anchor chains on the digital twin.

Faces: the street-facing vertical faces of the building bounding boxes of the scene (from
paper2/tables/geometry.json, written by scripts/p2_geometry.py from the scene meshes) and the
ground. A vertical face is kept if the point 1 m outside its centre is not inside any building
(it faces open space); coplanar faces with overlapping extents are merged into one (one reflector). Sources of one O-RU: LoS, every single bounce, every ordered double
bounce on two different faces (paper-1 traces use max_depth 2). A source is VALID for a UE
position if its specular points lie on the faces (within their extents, 5 cm margin) and the
image-method geometry is consistent (O-RU and UE on the reflecting side of every face).
Each face may be displaced along its normal by an unknown offset d_f (tracker state); the
predicted measurement of a source and its Jacobian w.r.t. (x, y) and the offsets of its faces
use the frozen image-method code (sim/positioning/geometry.py) and sim/tvt/geometry_map.py.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

C0 = 299_792_458.0
ROOT = Path(__file__).resolve().parents[2]


def faces_from_geometry(path: Path | None = None) -> list[dict]:
    """Faces: normal n (unit, pointing into open space), anchor q, extent box (min, max) over the 3 axes."""
    g = json.loads((path or ROOT / "paper2" / "tables" / "geometry.json").read_text())
    boxes = {k: v for k, v in g["street"]["mesh_bboxes"].items() if k != "floor"}
    faces = []

    def inside_any(p):
        return any(all(b["min"][i] - 1e-6 < p[i] < b["max"][i] + 1e-6 for i in range(3)) for b in boxes.values())

    for name, b in boxes.items():
        mn, mx = np.array(b["min"]), np.array(b["max"])
        for ax in (0, 1):
            for sign, coord in ((-1.0, mn[ax]), (1.0, mx[ax])):
                n = np.zeros(3)
                n[ax] = sign
                lo, hi = mn.copy(), mx.copy()
                lo[ax] = hi[ax] = coord
                centre = 0.5 * (lo + hi)
                centre[2] = 1.5
                if inside_any(centre + 1.0 * n):
                    continue
                faces.append({"name": f"{name}:{'xyz'[ax]}{'+' if sign > 0 else '-'}", "n": n, "q": centre.copy(), "lo": lo, "hi": hi, "axis": ax})
    # merge coplanar faces with the same normal whose extents overlap (one reflector; e.g. the south facade is
    # split into two boxes of different heights): one face with the union extent, so that no two sources
    # predict the same path (a duplicate would be counted twice by the sequential update)
    merged = True
    while merged:
        merged = False
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                f, h = faces[i], faces[j]
                if not np.allclose(f["n"], h["n"]) or abs((f["q"] - h["q"]) @ f["n"]) > 1e-6:
                    continue
                ax = [k for k in range(3) if abs(f["n"][k]) < 0.5]
                if all(f["lo"][k] <= h["hi"][k] + 1e-6 and h["lo"][k] <= f["hi"][k] + 1e-6 for k in ax):
                    lo, hi = np.minimum(f["lo"], h["lo"]), np.maximum(f["hi"], h["hi"])
                    q = 0.5 * (lo + hi)
                    q[2] = 1.5
                    q[f["axis"]] = f["q"][f["axis"]]
                    faces[i] = {"name": f["name"] + "+" + h["name"], "n": f["n"], "q": q, "lo": lo, "hi": hi, "axis": f["axis"]}
                    del faces[j]
                    merged = True
                    break
            if merged:
                break
    fl = g["street"]["mesh_bboxes"]["floor"]
    faces.append({"name": "ground", "n": np.array([0.0, 0.0, 1.0]), "q": np.array([0.0, 0.0, fl["max"][2]]), "lo": np.array(fl["min"]),
                  "hi": np.array(fl["max"]), "axis": 2})
    return faces


def street_faces(faces: list[dict], y_band: tuple[float, float] = (-12.0, 12.0)) -> list[dict]:
    """Faces that can be seen from the street (a part of the face lies within |y| <= 12 m, or the ground)."""
    out = []
    for f in faces:
        if f["axis"] == 2:
            out.append(f)
        elif f["axis"] == 1 and y_band[0] <= f["q"][1] <= y_band[1]:
            out.append(f)
        elif f["axis"] == 0 and f["lo"][1] <= y_band[1] and f["hi"][1] >= y_band[0]:
            out.append(f)
    return out


def source_list(n_faces: int) -> list[tuple[int, ...]]:
    """() = LoS, (i,) single bounce, (i, j) double bounce (i != j), path order O-RU -> UE."""
    return [()] + [(i,) for i in range(n_faces)] + [(i, j) for i, j in itertools.permutations(range(n_faces), 2)]


def source_planes(faces: list[dict], sources: list[tuple[int, ...]], offsets: np.ndarray | None = None):
    """normals [S, 2, 3], anchors [S, 2, 3], valid [S, 2], face ids [S, 2] (-1 unused); ``offsets`` [F] shift faces along n."""
    S = len(sources)
    N = np.zeros((S, 2, 3))
    A = np.zeros((S, 2, 3))
    V = np.zeros((S, 2), dtype=bool)
    fid = np.full((S, 2), -1, dtype=np.int64)
    for s, src in enumerate(sources):
        for k, f in enumerate(src):
            N[s, k] = faces[f]["n"]
            A[s, k] = faces[f]["q"] + (0.0 if offsets is None else offsets[f]) * faces[f]["n"]
            V[s, k] = True
            fid[s, k] = f
    return N, A, V, fid


def validity(oru: np.ndarray, ue: np.ndarray, faces: list[dict], sources, N, A, V, fid, margin_m: float = 0.05) -> np.ndarray:
    """Valid [..., S] for O-RU oru [..., 3] and UE ue [..., 3] (broadcast against the source axis)."""
    from sim.tvt.visibility import polyline

    pts = polyline(oru[..., None, :], ue[..., None, :], N, A, V)  # [..., S, 4, 3]
    ok = np.ones(pts.shape[:-2], dtype=bool)
    for k in range(2):
        # slot 0: O-RU -> b1 -> (b2, or the UE: polyline puts the UE in an unused slot); slot 1: b1 -> b2 -> UE
        b = pts[..., k + 1, :]
        prev = pts[..., k, :]
        nxt = pts[..., k + 2, :]
        used = V[..., k]
        n = N[..., k, :]
        q = A[..., k, :]
        sp = ((prev - q) * n).sum(-1)
        sn = ((nxt - q) * n).sum(-1)
        side_ok = (sp > 0.05) & (sn > 0.05)
        f = fid[..., k]
        lo = np.stack([faces[i]["lo"] if i >= 0 else np.full(3, -np.inf) for i in f.reshape(-1)]).reshape(f.shape + (3,))
        hi = np.stack([faces[i]["hi"] if i >= 0 else np.full(3, np.inf) for i in f.reshape(-1)]).reshape(f.shape + (3,))
        ext_ok = np.ones_like(side_ok)
        for ax in range(3):
            along = np.abs(n[..., ax]) < 0.5
            ext_ok &= ~along | ((b[..., ax] >= lo[..., ax] - margin_m) & (b[..., ax] <= hi[..., ax] + margin_m))
        ok &= ~used | (side_ok & ext_ok)
    return ok


def predict(oru: np.ndarray, ue: np.ndarray, N, A, V) -> dict[str, np.ndarray]:
    """tau [ns], u [3], Jacobians d(tau, u_y, u_z)/d(x, y) [..., 3, 2] and d/d(face offsets of the 2 slots) [..., 3, 2]."""
    from sim.positioning.geometry import mirror, reflect_matrix
    from sim.tvt.geometry_map import offset_derivatives

    img_o = oru.copy() if oru.ndim == ue.ndim else oru
    img_o = np.broadcast_to(img_o, np.broadcast_shapes(oru.shape, ue.shape, N[..., 0, :].shape)).copy()
    for i in range(2):
        m = V[..., i][..., None]
        img_o = np.where(m, mirror(img_o, N[..., i, :], A[..., i, :]), img_o)
    img_p = np.broadcast_to(ue, img_o.shape).copy()
    lin = np.broadcast_to(np.eye(3), img_o.shape[:-1] + (3, 3)).copy()
    for i in (1, 0):
        m = V[..., i]
        img_p = np.where(m[..., None], mirror(img_p, N[..., i, :], A[..., i, :]), img_p)
        lin = np.where(m[..., None, None], reflect_matrix(N[..., i, :]) @ lin, lin)
    dv = np.broadcast_to(ue, img_o.shape) - img_o
    dist = np.linalg.norm(dv, axis=-1)
    tau = dist / C0 * 1e9
    dtau = (dv / np.maximum(dist, 1e-12)[..., None])[..., :2] / C0 * 1e9  # ns/m
    w = img_p - oru
    rw = np.linalg.norm(w, axis=-1)
    u = w / np.maximum(rw, 1e-12)[..., None]
    proj = np.eye(3) - u[..., :, None] * u[..., None, :]
    du = (proj @ lin)[..., :, :2] / np.maximum(rw, 1e-12)[..., None, None]  # [..., 3, 2]
    Jp = np.stack([dtau, du[..., 1, :], du[..., 2, :]], axis=-2)  # [..., 3 (tau, uy, uz), 2 (x, y)]
    od = offset_derivatives(np.broadcast_to(oru, img_o.shape), np.broadcast_to(ue, img_o.shape), N, A, V)
    # offsets: tau [ns/m]; u_y, u_z from the az/el derivatives: du = (d u / d az) daz + (d u / d el) del
    az = np.arctan2(u[..., 1], u[..., 0])
    el = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    duy = np.cos(el)[..., None] * np.cos(az)[..., None] * od["daz"] - np.sin(el)[..., None] * np.sin(az)[..., None] * od["del"]
    duz = np.cos(el)[..., None] * od["del"]
    Jd = np.stack([od["dtau"] * 1e9, duy, duz], axis=-2)  # [..., 3, 2 slots]
    return {"tau": tau, "u": u, "Jp": Jp, "Jd": Jd}
