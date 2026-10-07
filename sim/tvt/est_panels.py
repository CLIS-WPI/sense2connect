"""Paper-2 estimator A with back-to-back panels (configs/tvt.yaml panels.positioning).

The frozen estimator A (sim/positioning/estimator_v2.py) enumerates a front (u_x > 0) and a back (u_x < 0)
hypothesis per O-RU, because its single UPA cannot tell them apart. With two panels the measuring panel tells
the half-space. ``candidates`` below is the frozen ``estimator_v2.candidates`` with ONE added line: a hypothesis
of O-RU c is kept at epoch e only if its side equals the sign of the panel that measured the dominant path,
meas["sx"][e, c]. Everything else (models, Gauss-Newton fixes, consistency, walkability, costs, the tracker
run_tracker_v2, the noise model) is the frozen code. ``track_v2`` is scripts/p2_estimate.track_v2 with this
``candidates``. The same information (panel sign per component) is given to the TVT tracker.
"""

from __future__ import annotations

import itertools

import numpy as np


def candidates(meas: dict, oru: np.ndarray, use: np.ndarray, timing: str, sig_tau, sig_u, z_ue: float, wmap: dict,
               reflections: list | None, sx: np.ndarray) -> dict:
    """estimator_v2.candidates restricted to the measuring panel's half-space; sx [E, C] in {+1, -1}."""
    from sim.positioning.estimator_v2 import LOS, _mirror_xy, ray_fix_model, solve, walkable

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
        ok = np.ones(E, dtype=bool)
        for c, o in enumerate(combo):
            if o is not None and o[0] == "los":
                ok &= use[:, c]
        if not ok.any():
            continue
        fixes = [ray_fix_model(oru[:, c], meas["uy"][:, c], meas["uz"][:, c], o[1], z_ue, o[2]) for c, o in enumerate(combo) if o is not None]
        sol = None
        for init in fixes:
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
        for c, o in enumerate(combo):
            if o is None:
                continue
            q, _ = _mirror_xy(p, o[2])
            valid &= np.sign(q[:, 0] - oru[:, c, 0]) == o[1]
            valid &= sx[:, c] == o[1]  # TVT panels: the measuring panel tells the half-space (the only change)
        valid &= walkable(sol["xy"], wmap)
        n_refl = sum(1 for o in combo if o is not None and o[0] == "refl")
        xs.append(sol["xy"])
        covs.append(sol["cov"])
        costs.append(np.where(valid, sol["cost"], np.inf))
        joints.append((sum(o is not None for o in combo) == C, n_refl))
    return {"xy": np.stack(xs, 1), "cov": np.stack(covs, 1), "cost": np.stack(costs, 1), "joint": np.array([j for j, _ in joints]),
            "n_refl": np.array([n for _, n in joints])}


def track_v2(meas: dict, oru: np.ndarray, timing: str, params: dict, n_sc: int, df: float, job: tuple, raw: dict, p2cfg: dict, variant: str,
             memo: dict | None = None) -> dict:
    """scripts/p2_estimate.track_v2 with the panel-restricted candidates; meas["sx"] [T, U, C]."""
    import p2_estimate as P
    from sim.positioning.estimator import noise_model
    from sim.positioning.estimator_v2 import reflection_planes, run_tracker_v2, walk_map
    from sim.scenes.traffic import prepare_scenario

    T, U, C = meas["uy"].shape
    out = {"xy_ekf": np.zeros((T, U, 2)), "xy_fix": np.zeros((T, U, 2))}
    for uu in range(U):
        key = ("cand", uu)
        if memo is not None and key in memo:
            cand = memo[key]
        else:
            sc = prepare_scenario(raw, seed=job[0], mount=job[1], density=job[2], duration_s=60.0, dt_s=0.1)
            wmap = walk_map(raw, sc, p2cfg)
            m = {k: v[:, uu, :] for k, v in meas.items() if k != "sx"}
            use, st, su = noise_model(m, params, df, n_sc)
            cand = candidates(m, oru, use, timing, st, su, P.Z_UE, wmap, reflection_planes(p2cfg) if variant == "AB" else None,
                              np.asarray(meas["sx"][:, uu, :]))
            if memo is not None:
                memo[key] = cand
        tr = run_tracker_v2(cand, params)
        out["xy_ekf"][:, uu] = tr["xy_ekf"]
        out["xy_fix"][:, uu] = tr["xy_fix"]
    return out
