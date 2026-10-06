"""Estimator variants A and A+B (after the paper-2 review; estimator-only changes).

Both reuse the per-O-RU measurements of the frozen estimator (sim/positioning/
estimator.py, measure_torch) and its noise model; only fusion and tracking change.

A  mirror disambiguation. The 8 x 8 array lies in the y-z plane, so each O-RU's
   angle has a front (u_x > 0) and a back (u_x < 0) hypothesis. Per epoch every
   joint hypothesis (front/back per gated O-RU) is solved by weighted least
   squares (Gauss-Newton from each single-O-RU ray fix, the lower cost kept) and kept only if
   - CONSISTENT: the solution lies on the hypothesised side of each O-RU's array
     plane (sign(x - x_O-RU) = hypothesis);
   - WALKABLE in the digital-twin map: inside a sidewalk band (between the facade
     and the curb) or a pedestrian crossing corridor, within the street extent;
   - and, once the track exists, inside the EKF selection gate (Mahalanobis^2 to
     the prediction <= sel_gate).
   The track starts from a two-O-RU (joint) fix with cost <= res_gate; if none has
   appeared 1 s after the start, from the best admissible joint fix, else from the
   best admissible single-O-RU fix. Selection: the joint
   candidate with the lowest cost if its cost <= res_gate (ties within 1 -> nearest
   to the prediction); otherwise the single-O-RU candidate nearest to the prediction.
B  map-aided reflections: every O-RU's dominant path (in particular one that fails
   the LoS gate, which is then not dropped) is also tested as a single specular
   reflection on a known facade or the ground (virtual anchor = O-RU mirrored across the surface; equivalently the
   UE mirrored across it, image method): predicted delay |M(p) - o| / c and angle
   of M(p) - o, with M the mirror; the reflection hypotheses enter the same
   enumeration (with front/back), each with an additive cost penalty refl_penalty.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from sim.positioning.estimator import C0, predict_meas

LOS = ("los", None, None)


def walkable(xy: np.ndarray, wmap: dict) -> np.ndarray:
    """Digital-twin walkable area: sidewalk bands or crossing corridors, within the street extent."""
    x, y = xy[..., 0], xy[..., 1]
    ok_x = (x >= wmap["x_min"]) & (x <= wmap["x_max"])
    side = np.zeros_like(x, dtype=bool)
    for lo, hi in wmap["sidewalks_y"]:
        side |= (y >= lo) & (y <= hi)
    cross = np.zeros_like(x, dtype=bool)
    for xc in wmap["crossings_x"]:
        cross |= (np.abs(x - xc) <= wmap["crossing_half_width"]) & (y >= wmap["street_y"][0]) & (y <= wmap["street_y"][1])
    return ok_x & (side | cross)


def walk_map(raw: dict, scenario: dict, p2cfg: dict) -> dict:
    """Walkable map from the digital twin (scene planes, lanes/sidewalks of the config, crossing stations of the scenario)."""
    planes = p2cfg["known_planes"]
    y_s, y_n = min(planes["y"]), max(p for p in planes["y"] if p < 10.0)
    lanes_y = [float(l["origin_m"][1]) for l in raw["lanes"]]
    curb = max(abs(v) for v in lanes_y) + 0.5 * float(p2cfg.get("lane_width_m", 3.5))
    x0 = float(raw["sidewalks"][0]["origin_m"][0])
    length = float(raw["sidewalks"][0]["length_m"])
    crossings = [x0 + float(a["s_cross_m"]) for a in scenario["pedestrians"] if "s_cross_m" in a]
    return {"x_min": x0, "x_max": x0 + length, "sidewalks_y": [(y_s, -curb), (curb, y_n)], "street_y": (y_s, y_n),
            "crossings_x": crossings, "crossing_half_width": 1.5}


def _mirror_xy(p: np.ndarray, model: tuple) -> tuple[np.ndarray, np.ndarray]:
    """UE image M(p) [E, 3] for a reflection model and d M(p)_xyz / d(x, y) [3, 2]."""
    kind, axis, off = model
    J = np.zeros((3, 2))
    J[0, 0] = J[1, 1] = 1.0
    if kind == "los":
        return p, J
    q = p.copy()
    q[:, axis] = 2.0 * off - p[:, axis]
    if axis < 2:
        J[axis, axis] = -1.0
    return q, J


def solve(meas: dict, oru: np.ndarray, models: list, timing: str, sig_tau: np.ndarray, sig_u: np.ndarray, z_ue: float,
          init: np.ndarray, n_iter: int = 8) -> dict:
    """WLS over epochs for one joint hypothesis: models[c] = None (O-RU unused) or (kind, axis, offset). init [E, 2]."""
    E, C = meas["uy"].shape
    n_par = 3 if timing == "tdoa" else 2
    th = np.zeros((E, n_par))
    th[:, :2] = np.nan_to_num(init)
    used = [c for c in range(C) if models[c] is not None]
    if timing == "tdoa":
        p0 = np.concatenate([th[:, :2], np.full((E, 1), z_ue)], -1)
        res = []
        for c in used:
            q, _ = _mirror_xy(p0, models[c])
            res.append(meas["tau_ns"][:, c] - predict_meas(q, oru[:, c])[0] / C0 * 1e9)
        th[:, 2] = np.mean(res, 0) if res else 0.0
    for it in range(n_iter + 1):
        p = np.concatenate([th[:, :2], np.full((E, 1), z_ue)], -1)
        rows_r, rows_J = [], []
        for c in used:
            q, Jm = _mirror_xy(p, models[c])
            rng, py, pz = predict_meas(q, oru[:, c])
            d = q - oru[:, c]
            du = np.zeros((E, 2, 3))  # d(u_y, u_z)/d q
            du[:, 0] = -d[:, 1:2] * d / rng[:, None] ** 3
            du[:, 0, 1] += 1.0 / rng
            du[:, 1] = -d[:, 2:3] * d / rng[:, None] ** 3
            du[:, 1, 2] += 1.0 / rng
            dxy = du @ Jm  # [E, 2, 2]
            for k, res in enumerate((meas["uy"][:, c] - py, meas["uz"][:, c] - pz)):
                Jr = np.zeros((E, n_par))
                Jr[:, :2] = dxy[:, k]
                rows_r.append(res / sig_u[:, c])
                rows_J.append(Jr / sig_u[:, c][:, None])
            if timing != "aoa":
                Jr = np.zeros((E, n_par))
                Jr[:, :2] = ((d / rng[:, None]) @ Jm) / C0 * 1e9
                if timing == "tdoa":
                    Jr[:, 2] = 1.0
                pred = rng / C0 * 1e9 + (th[:, 2] if timing == "tdoa" else 0.0)
                rows_r.append((meas["tau_ns"][:, c] - pred) / sig_tau[:, c])
                rows_J.append(Jr / sig_tau[:, c][:, None])
        r = np.stack(rows_r, -1)
        J = np.stack(rows_J, -2)
        JtJ = J.transpose(0, 2, 1) @ J + 1e-9 * np.eye(n_par)
        if it == n_iter:
            break
        step = np.linalg.solve(JtJ, (J.transpose(0, 2, 1) @ r[..., None]))[..., 0]
        step[:, :2] = np.clip(step[:, :2], -20.0, 20.0)
        th = th + step
    cost = (r ** 2).sum(-1)
    return {"xy": th[:, :2], "cov": np.linalg.inv(JtJ)[:, :2, :2], "cost": cost}


def ray_fix_model(o: np.ndarray, uy, uz, front: float, z_ue: float, model: tuple) -> np.ndarray:
    """UE position from one O-RU's angle under a LoS or reflection model (image method)."""
    kind, axis, off = model
    ux = front * np.sqrt(np.clip(1.0 - uy ** 2 - uz ** 2, 0.0, 1.0))
    z_img = (2.0 * off - z_ue) if (kind != "los" and axis == 2) else z_ue
    t = (z_img - o[..., 2]) / np.where(np.abs(uz) > 1e-6, uz, np.nan)
    t = np.where(t > 0, t, np.nan)
    q = np.stack([o[..., 0] + t * ux, o[..., 1] + t * uy], -1)
    if kind != "los" and axis == 1:
        q[..., 1] = 2.0 * off - q[..., 1]
    return q


def candidates(meas: dict, oru: np.ndarray, use: np.ndarray, timing: str, sig_tau, sig_u, z_ue: float, wmap: dict,
               reflections: list | None) -> dict:
    """All hypotheses for one UE over the epochs: xy [E, H, 2], cov [E, H, 2, 2], cost [E, H] (inf = invalid), joint [H]."""
    E, C = use.shape
    per = []
    for c in range(C):
        opts = [None]
        for f in (1.0, -1.0):
            opts.append(("los", f, LOS))
            if reflections:
                for pl in reflections:
                    opts.append(("refl", f, pl))
        per.append(opts)
    xs, covs, costs, joints = [], [], [], []
    for combo in itertools.product(*per):
        if all(o is None for o in combo):
            continue
        models = [None if o is None else o[2] for o in combo]
        # applicability per epoch: the LoS model only for O-RUs that pass the LoS gate; reflection models (B) for every O-RU
        ok = np.ones(E, dtype=bool)
        for c, o in enumerate(combo):
            if o is not None and o[0] == "los":
                ok &= use[:, c]
        if not ok.any():
            continue
        fixes = [ray_fix_model(oru[:, c], meas["uy"][:, c], meas["uz"][:, c], o[1], z_ue, o[2]) for c, o in enumerate(combo) if o is not None]
        sol = None
        for init in fixes:  # Gauss-Newton from each single-O-RU ray fix; keep the lower cost per epoch
            s_ = solve(meas, oru, models, timing, sig_tau, sig_u, z_ue, init)
            s_["cost"] = np.where(np.isfinite(init).all(-1), s_["cost"], np.inf)
            if sol is None:
                sol = s_
            else:
                b = s_["cost"] < sol["cost"]
                sol = {"xy": np.where(b[:, None], s_["xy"], sol["xy"]), "cov": np.where(b[:, None, None], s_["cov"], sol["cov"]),
                       "cost": np.where(b, s_["cost"], sol["cost"])}
        valid = ok & np.isfinite(sol["cost"])
        p = np.concatenate([sol["xy"], np.full((E, 1), z_ue)], -1)
        for c, o in enumerate(combo):  # consistency: the solution lies on the hypothesised side of each array plane
            if o is None:
                continue
            q, _ = _mirror_xy(p, o[2])
            valid &= np.sign(q[:, 0] - oru[:, c, 0]) == o[1]
        valid &= walkable(sol["xy"], wmap)
        n_refl = sum(1 for o in combo if o is not None and o[0] == "refl")
        xs.append(sol["xy"])
        covs.append(sol["cov"])
        costs.append(np.where(valid, sol["cost"], np.inf))
        joints.append((sum(o is not None for o in combo) == C, n_refl))
    return {"xy": np.stack(xs, 1), "cov": np.stack(covs, 1), "cost": np.stack(costs, 1), "joint": np.array([j for j, _ in joints]),
            "n_refl": np.array([n for _, n in joints])}


def run_tracker_v2(cand: dict, params: dict, dt: float = 0.1) -> dict:
    """EKF with hypothesis selection (variant A / A+B); params: res_gate, q, chi2, sel_gate, refl_penalty."""
    from sim.positioning.estimator import REINIT_AFTER

    E, H = cand["cost"].shape
    q = float(params["q"])
    chi2 = float(params["chi2"])
    res_gate = float(params["res_gate"])
    sel_gate = float(params["sel_gate"])
    pen = float(params.get("refl_penalty", 0.0))
    cost_all = cand["cost"] + pen * cand["n_refl"][None, :]
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Qm = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2], [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
    xy_ekf = np.full((E, 2), np.nan)
    xy_fix = np.full((E, 2), np.nan)
    x = Pm = None
    rejected = 0
    waited = 0
    joint = cand["joint"]
    for e in range(E):
        if x is not None:
            x = F @ x
            Pm = F @ Pm @ F.T + Qm
        c = cost_all[e]
        fin = np.isfinite(c)
        if x is not None and fin.any():
            d = cand["xy"][e] - x[:2]
            S = Pm[:2, :2][None] + cand["cov"][e]
            m2 = np.einsum("hi,hij,hj->h", d, np.linalg.inv(S + 1e-9 * np.eye(2)), d)
            fin_g = fin & (m2 <= sel_gate)
        else:
            m2 = None
            fin_g = fin
        sel = None
        if x is None:
            waited += 1
        jj = np.flatnonzero(fin_g & joint & (c <= res_gate))
        if x is None and not jj.size and waited >= REINIT_AFTER:
            # no joint fix below res_gate for 1 s since the start: start from the best joint fix, else from the best single fix
            jj = np.flatnonzero(fin & joint)
            if not jj.size:
                jj = np.flatnonzero(fin)
        if jj.size:
            cmin = c[jj].min()
            ties = jj[c[jj] <= cmin + 1.0]
            k = ties[np.argmin(m2[ties])] if (m2 is not None and ties.size > 1) else ties[np.argmin(c[ties])]
            sel = k
        elif x is not None:
            ss = np.flatnonzero(fin_g & ~joint)
            if ss.size:
                sel = ss[np.argmin(m2[ss])]
        if sel is not None:
            z, R = cand["xy"][e, sel], cand["cov"][e, sel] + 1e-6 * np.eye(2)
            xy_fix[e] = z
            if x is None:
                x = np.array([z[0], z[1], 0.0, 0.0])
                Pm = np.diag([R[0, 0], R[1, 1], 4.0, 4.0])
            else:
                S = Pm[:2, :2] + R
                v = z - x[:2]
                if float(v @ np.linalg.solve(S, v)) <= chi2:
                    K = Pm[:, :2] @ np.linalg.inv(S)
                    x = x + K @ v
                    Pm = Pm - K @ Pm[:2, :]
                    rejected = 0
                else:
                    rejected += 1
                    if rejected >= REINIT_AFTER:
                        x = np.array([z[0], z[1], 0.0, 0.0])
                        Pm = np.diag([R[0, 0], R[1, 1], 4.0, 4.0])
                        rejected = 0
        elif x is not None:
            # no admissible candidate inside the selection gate for 1 s: allow re-acquisition from a joint fix
            rejected += 1
            if rejected >= REINIT_AFTER:
                jj = np.flatnonzero(fin & joint & (c <= res_gate))
                if jj.size:
                    k = jj[np.argmin(c[jj])]
                    z = cand["xy"][e, k]
                    x = np.array([z[0], z[1], 0.0, 0.0])
                    Pm = np.diag([cand["cov"][e, k][0, 0] + 1e-6, cand["cov"][e, k][1, 1] + 1e-6, 4.0, 4.0])
                    rejected = 0
        if x is not None:
            xy_ekf[e] = x[:2]
    return {"xy_ekf": xy_ekf, "xy_fix": xy_fix}


def reflection_planes(p2cfg: dict) -> list:
    """Known facades (y planes of the street canyon) and the ground as (kind, axis, offset)."""
    planes = p2cfg["known_planes"]
    ys = [y for y in planes["y"]]
    return [("refl", 1, float(y)) for y in (min(ys), max(y for y in ys if y < 10.0))] + [("refl", 2, float(planes["z"][0]))]
