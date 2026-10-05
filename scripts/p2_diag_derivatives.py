"""Paper 2 DIAGNOSTIC (review 4): validation of the analytic image-method derivatives.

(a) Independent double-precision check (no Sionna). Random UE positions (x ~ U(-50, 50) m,
    y ~ U(-8.4, 9.4) m, z = 1.5 m) and both O-RUs of both mounts (paper-1 geometry cache);
    specular paths on the planar surfaces of the digital twin (configs/p2.yaml known_planes:
    facades y = -8.613 m and y = 9.572 m, ground z = -0.031 m), built with the image method:
    every single bounce and every ordered double bounce on two different planes whose bounce
    points lie between their neighbours (geometrically valid specular path). Analytic
    derivatives of delay, azimuth and elevation w.r.t. UE x and y (sim/positioning/geometry.py,
    frozen) vs central finite differences of the same float64 image-method geometry with the
    planes held fixed, steps h in {1e-4, 1e-3, 1e-2, 1e-1} m. Per quantity, step and bounce
    order: max / median absolute and relative errors (relative = |fd - an| / |an|; also
    normalised by the gradient norm |grad| of that quantity, which stays defined when one
    component is near zero) and the observed order of the median error between steps.
(b) Sionna re-trace check (float32). The validation subset of scripts/p2_trace.py --fd
    (2 seeds x 2 mounts, low density, 20 snapshots) re-traced with steps
    {0.5, 1, 2, 5} cm by calling the frozen p2_trace._one with FD_STEP_M and OUT_ROOT replaced
    (outputs in results/P2/diag_deriv/h_<cm>cm; results/cache and results/P2/cache untouched).
    Per path (all six derivatives: {delay, az, el} x {x, y}) the paper's criterion: agreement to
    1e-3 relative. Every DISAGREEING path is assigned one cause, in this priority:
    1. path changes with the step: the path (same path_key) is missing at one of the four
       offsets (appears / vanishes), or the Sionna delay does not follow the fixed-plane
       image-method model across the step: the second difference tau(+h) - 2 tau(0) + tau(-h)
       of Sionna deviates from that of the float64 image-method delay on the snapped planes by
       more than 1e-3 h |dtau/dx_k| + 6 sqrt(2) eps32 tau (kink: an interaction point leaves its
       face / not the same specular path on both sides; first-order plane errors cancel in the
       second difference; tested on the delay, the one quantity whose nominal Sionna value is
       stored);
    2. near-zero derivative: a failing value with |an| < 1e-2 |grad| and |fd - an| <= 1e-3 |grad|
       (agrees relative to the gradient norm; relative error ill-defined);
    3. float32 rounding: every failing value satisfies |fd - an| <= 1e-3 |an| + 3 floor, with
       the float32 floor of scripts/test_p2.py (delay sqrt(2) eps32 tau / (2h); angles
       sqrt(2) v / (2 h r), v = 2e-5 m float32 interaction-point precision, r the (horizontal)
       range to the first point);
    4. unexplained.
    Non-exclusive counts per criterion are reported too, with the error scaling across steps
    (float32 noise ~ 1/h, truncation ~ h^2).
Writes results/P2/diag_derivatives.json.
Run: python scripts/p2_diag_derivatives.py [--retrace]  (--retrace needs the GPU; a few s per job).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STEPS_A = (1e-4, 1e-3, 1e-2, 1e-1)
STEPS_B = (0.005, 0.01, 0.02, 0.05)
N_UE = 1000
F32_EPS = float(np.finfo(np.float32).eps)
VPREC = 2e-5
JOBS = [("lamppost", 1001), ("lamppost", 1002), ("facade", 1001), ("facade", 1002)]
OUT_B = ROOT / "results" / "P2" / "diag_deriv"
QUANT = ("tau", "az", "el")


def wrap(a: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * a))


# ---------------------------------------------------------------- (a) float64 check
def specular_paths(oru: np.ndarray, ue: np.ndarray, planes: list[tuple[np.ndarray, np.ndarray]]):
    """Valid specular image-method paths O-RU -> bounces -> UE: list of (order, normals [2,3], anchors [2,3], valid [2])."""
    from sim.positioning.geometry import mirror

    out = []
    for order in (1, 2):
        for combo in itertools.permutations(range(len(planes)), order):
            nrm = np.zeros((2, 3))
            anc = np.zeros((2, 3))
            val = np.zeros(2, dtype=bool)
            for i, k in enumerate(combo):
                nrm[i], anc[i] = planes[k]
                val[i] = True
            # bounce points: images of the UE in reverse order; B_i on the line O_img_i -> next point
            imgs = [ue]
            for i in reversed(range(order)):
                imgs.insert(0, mirror(imgs[0], nrm[i], anc[i]))
            # imgs[i] = UE mirrored across planes i..order-1; walk from the O-RU forwards
            pts = [oru]
            ok = True
            for i in range(order):
                a, b = pts[-1], imgs[i]
                da, db = np.dot(a - anc[i], nrm[i]), np.dot(b - anc[i], nrm[i])
                if da * db >= 0 or abs(da) < 0.05:  # same side (no crossing) or start on the plane
                    ok = False
                    break
                t = da / (da - db)
                pts.append(a + t * (b - a))
            if not ok:
                continue
            pts.append(ue)
            # each bounce point lies strictly between its neighbours along the reflected chain (reflect, not pass through)
            for i in range(1, order + 1):
                n_ = nrm[i - 1]
                s0, s1 = np.dot(pts[i - 1] - anc[i - 1], n_), np.dot(pts[i + 1] - anc[i - 1], n_)
                if s0 * s1 <= 0 or abs(s1) < 0.05:
                    ok = False
            if ok:
                out.append((order, nrm, anc, val))
    return out


def part_a(kp: dict) -> dict:
    from sim.positioning.geometry import path_geometry

    rng = np.random.default_rng(20261005)
    planes = []
    for ax, name in ((1, "y"), (2, "z")):
        for off in kp[name]:
            if name == "y" and off > 10.0:  # set-back facade at 10.34 m is behind the 9.57 m face from the street
                continue
            n = np.zeros(3)
            n[ax] = 1.0
            q = np.zeros(3)
            q[ax] = off
            planes.append((n, q))
    orus = []
    for mount in ("lamppost", "facade"):
        g = np.load(ROOT / "results" / "cache" / mount / "low" / "seed_1001" / "comm_geometry.npz")
        orus += [g["oru_position_m"][0, c] for c in range(g["oru_position_m"].shape[1])]
    orus = np.unique(np.round(np.array(orus), 6), axis=0)
    rec = {(o, h, q, k): [] for o in (0, 1, 2) for h in STEPS_A for q in QUANT for k in ("abs", "rel", "rel_grad")}
    n_paths = {0: 0, 1: 0, 2: 0}
    for _ in range(N_UE):
        ue = np.array([rng.uniform(-50, 50), rng.uniform(-8.4, 9.4), 1.5])
        for oru in orus:
            paths = [(0, np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2, dtype=bool))] + specular_paths(oru, ue, planes)
            for order, N, A, V in paths:
                n_paths[order] += 1
                g = path_geometry(oru, ue, N, A, V)
                an = {"tau": g["dtau"], "az": g["daz"], "el": g["del"]}
                for h in STEPS_A:
                    for k in range(2):
                        e = np.zeros(3)
                        e[k] = h
                        gp, gm = path_geometry(oru, ue + e, N, A, V), path_geometry(oru, ue - e, N, A, V)
                        fd = {"tau": (gp["tau"] - gm["tau"]) / (2 * h), "az": wrap(gp["az"] - gm["az"]) / (2 * h), "el": (gp["el"] - gm["el"]) / (2 * h)}
                        for q in QUANT:
                            err = abs(float(fd[q]) - float(an[q][k]))
                            rec[(order, h, q, "abs")].append(err)
                            rec[(order, h, q, "rel")].append(err / max(abs(float(an[q][k])), 1e-300))
                            rec[(order, h, q, "rel_grad")].append(err / max(float(np.linalg.norm(an[q])), 1e-300))
    res = {"n_ue": N_UE, "orus": orus.tolist(), "planes": [[n.tolist(), q.tolist()] for n, q in planes], "n_paths": {str(k): v for k, v in n_paths.items()},
           "units": {"tau": "s/m", "az": "rad/m", "el": "rad/m"}, "per_order": {}, "all_nlos": {}}
    for label, orders in (("los", (0,)), ("single", (1,)), ("double", (2,)), ("nlos", (1, 2))):
        tab = {}
        for h in STEPS_A:
            for q in QUANT:
                cell = {}
                for k in ("abs", "rel", "rel_grad"):
                    v = np.concatenate([np.asarray(rec[(o, h, q, k)]) for o in orders])
                    cell[f"max_{k}"] = float(v.max())
                    cell[f"median_{k}"] = float(np.median(v))
                tab[f"{h:g}|{q}"] = cell
        # observed order of the median absolute error between consecutive steps (log-log slope)
        slopes = {}
        for q in QUANT:
            med = [tab[f"{h:g}|{q}"]["median_abs"] for h in STEPS_A]
            slopes[q] = [float(math.log10(med[i + 1] / med[i]) / math.log10(STEPS_A[i + 1] / STEPS_A[i])) if med[i] > 0 and med[i + 1] > 0 else None
                         for i in range(len(STEPS_A) - 1)]
        res["per_order"][label] = {"table": tab, "loglog_slope_median_abs": slopes}
    # best step: the step with the smallest max relative error over single + double bounce paths and all quantities
    nl = res["per_order"]["nlos"]["table"]
    worst = {h: max(nl[f"{h:g}|{q}"]["max_rel"] for q in QUANT) for h in STEPS_A}
    hb = min(worst, key=worst.get)
    res["best_step_m"] = hb
    res["max_rel_at_best_step"] = worst[hb]
    res["max_rel_grad_at_best_step"] = max(nl[f"{hb:g}|{q}"]["max_rel_grad"] for q in QUANT)
    res["max_rel_per_step"] = {f"{h:g}": w for h, w in worst.items()}
    return res


# ---------------------------------------------------------------- (b) Sionna re-trace
def retrace() -> None:
    import p2_trace as T
    from sim.scenes.config import load_yaml

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    for h in STEPS_B:
        T.FD_STEP_M = h
        T.OUT_ROOT = OUT_B / f"h_{100 * h:g}cm"
        for mount, seed in JOBS:
            T._one((raw, mount, "low", seed, 20, True))


def part_b(kp: dict) -> dict:
    from sim.positioning.geometry import bounce_planes, path_geometry

    res = {"jobs": [f"{m}_low_{s}" for m, s in JOBS], "steps": {}}
    for h in STEPS_B:
        cnt = {"paths_nominal": 0, "paths_all_offsets": 0, "agree": 0, "disagree": 0,
               "cause_exclusive": {"path_changes_vanish": 0, "path_changes_kink": 0, "near_zero_derivative": 0, "float32_rounding": 0, "unexplained": 0},
               "criterion_nonexclusive": {"kink": 0, "near_zero": 0, "float32": 0}}
        err_abs = {q: [] for q in QUANT}
        err_rel = {q: [] for q in QUANT}
        well = {q: [] for q in QUANT}
        same_as_p2_trace = None
        for mount, seed in JOBS:
            fp = OUT_B / f"h_{100 * h:g}cm" / mount / "low" / f"seed_{seed}" / "p2_fd.npz"
            d = np.load(ROOT / "results" / "cache" / mount / "low" / f"seed_{seed}" / "comm_geometry.npz")
            f = np.load(fp)
            if abs(h - 0.01) < 1e-12:
                ref = np.load(ROOT / "results" / "P2" / "cache" / mount / "low" / f"seed_{seed}" / "p2_fd.npz")
                a, b = ref["fd_tau"], f["fd_tau"]
                m = np.isfinite(a) & np.isfinite(b)
                dev = float(np.max(np.abs(a[m] - b[m]) / np.abs(a[m]))) if m.any() else 0.0
                same = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
                same_as_p2_trace = {"max_rel_dev": max(dev, (same_as_p2_trace or {}).get("max_rel_dev", 0.0)),
                                    "same_finite_mask": same and (same_as_p2_trace or {}).get("same_finite_mask", True)}
            ns = f["fd_tau"].shape[0]
            P, n, cls = d["points_m"][:ns], d["n_points"][:ns].astype(int), d["path_class"][:ns]
            ue = np.broadcast_to(d["ue_position_m"][:ns, :, None, None, :], P.shape[:-2] + (3,))
            oru = np.broadcast_to(d["oru_position_m"][:ns, None, :, None, :], P.shape[:-2] + (3,))
            N, A, V = bounce_planes(P, n, kp)
            g = path_geometry(oru, ue, N, A, V)
            first = np.nan_to_num(P[..., 1, :])
            rng_first = np.linalg.norm(first - oru, axis=-1)
            rng_h = np.linalg.norm((first - oru)[..., :2], axis=-1)
            uu = f["fd_u"]
            az = np.arctan2(uu[..., 1], uu[..., 0])
            el = np.arcsin(np.clip(uu[..., 2], -1, 1))
            tau0 = f["tau_nominal"]
            nom = {"tau": tau0}
            vals = {"tau": f["fd_tau"], "az": az, "el": el}
            floor = {"tau": math.sqrt(2) * F32_EPS * g["tau"] / (2 * h), "az": math.sqrt(2) * VPREC / (2 * h * np.maximum(rng_h, 1e-9)),
                     "el": math.sqrt(2) * VPREC / (2 * h * np.maximum(rng_first, 1e-9))}
            an_all = {"tau": g["dtau"], "az": g["daz"], "el": g["del"]}
            have = cls >= 0
            full = have & np.all(np.isfinite(f["fd_tau"]), -1)
            cnt["paths_nominal"] += int(have.sum())
            cnt["paths_all_offsets"] += int(full.sum())
            vanish = have & ~full
            fail_any = np.zeros(have.shape, bool)
            kink_any = np.zeros(have.shape, bool)
            fail_not_nz = np.zeros(have.shape, bool)  # failing values not explained as near-zero
            fail_not_f32 = np.zeros(have.shape, bool)  # failing values not within the float32 floor
            nz_any = np.zeros(have.shape, bool)
            for k, (ip, im) in enumerate(((0, 1), (2, 3))):
                for q in QUANT:
                    v = vals[q]
                    dif = wrap if q == "az" else (lambda x: x)
                    fd = dif(v[..., ip] - v[..., im]) / (2 * h)
                    an = an_all[q][..., k]
                    gn = np.linalg.norm(an_all[q], axis=-1)
                    err = np.abs(fd - an)
                    fail = full & ~(err <= 1e-3 * np.abs(an))
                    tol1 = 1e-3 * np.abs(an) + 3 * floor[q]
                    if q == "tau":
                        e = np.zeros(3)
                        e[k] = h
                        tp, tm = path_geometry(oru, ue + e, N, A, V)["tau"], path_geometry(oru, ue - e, N, A, V)["tau"]
                        d2s = v[..., ip] - 2 * nom[q] + v[..., im]
                        d2a = tp - 2 * g["tau"] + tm
                        kink = full & (np.abs(d2s - d2a) > 1e-3 * h * np.abs(an) + 6 * math.sqrt(2) * F32_EPS * g["tau"])
                    else:
                        kink = np.zeros_like(full)
                    nz = fail & (np.abs(an) < 1e-2 * gn) & (err <= 1e-3 * gn)
                    f32 = fail & (err <= tol1)
                    fail_any |= fail
                    kink_any |= kink
                    nz_any |= nz
                    fail_not_nz |= fail & ~nz
                    fail_not_f32 |= fail & ~f32
                    err_abs[q].append(err[full])
                    err_rel[q].append((err / np.maximum(np.abs(an), 1e-300))[full])
                    well[q].append((np.abs(an) >= 1e-2 * gn)[full])
            agree = full & ~fail_any
            dis = full & fail_any
            cnt["agree"] += int(agree.sum())
            cnt["disagree"] += int(dis.sum())
            c_kink = dis & kink_any  # a kink in the delay explains any failing value of that path
            c_nz = dis & ~c_kink & ~fail_not_nz
            c_f32 = dis & ~c_kink & ~c_nz & ~fail_not_f32
            c_unx = dis & ~c_kink & ~c_nz & ~c_f32
            ce = cnt["cause_exclusive"]
            ce["path_changes_vanish"] += int(vanish.sum())
            ce["path_changes_kink"] += int(c_kink.sum())
            ce["near_zero_derivative"] += int(c_nz.sum())
            ce["float32_rounding"] += int(c_f32.sum())
            ce["unexplained"] += int(c_unx.sum())
            cn = cnt["criterion_nonexclusive"]
            cn["kink"] += int(c_kink.sum())
            cn["near_zero"] += int((dis & nz_any).sum())
            cn["float32"] += int((dis & ~fail_not_f32).sum())
        stats = {}
        for q in QUANT:
            ea, er, wd = np.concatenate(err_abs[q]), np.concatenate(err_rel[q]), np.concatenate(well[q])
            stats[q] = {"max_abs": float(ea.max()), "median_abs": float(np.median(ea)), "max_rel": float(er.max()), "median_rel": float(np.median(er)),
                        "n_values": int(ea.size), "n_near_zero_an": int((~wd).sum()),
                        "max_rel_well_defined": float(er[wd].max()), "median_rel_well_defined": float(np.median(er[wd]))}
        n_dis_all = cnt["disagree"] + cnt["cause_exclusive"]["path_changes_vanish"]
        cnt["share_paths_agree_of_persisting"] = cnt["agree"] / max(cnt["paths_all_offsets"], 1)
        cnt["share_of_disagreements_exclusive"] = {k: v / max(n_dis_all, 1) for k, v in cnt["cause_exclusive"].items()}
        cnt["share_of_persisting_disagreements_exclusive"] = {k: v / max(cnt["disagree"], 1) for k, v in cnt["cause_exclusive"].items() if k != "path_changes_vanish"}
        res["steps"][f"{100 * h:g}cm"] = {"counts": cnt, "errors": stats, "check_vs_p2_trace_1cm": same_as_p2_trace}
    # scaling of the median absolute error with h (float32 noise ~ h^-1, truncation ~ h^2)
    res["loglog_slope_median_abs"] = {q: [float(math.log10(res["steps"][f"{100 * STEPS_B[i + 1]:g}cm"]["errors"][q]["median_abs"]
                                                          / res["steps"][f"{100 * STEPS_B[i]:g}cm"]["errors"][q]["median_abs"]) / math.log10(STEPS_B[i + 1] / STEPS_B[i]))
                                          for i in range(len(STEPS_B) - 1)] for q in QUANT}
    return res


def main() -> None:
    from sim.scenes.config import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrace", action="store_true")
    args = ap.parse_args()
    kp = load_yaml(ROOT / "configs" / "p2.yaml")["known_planes"]
    if args.retrace:
        retrace()
    out = {"definition": __doc__, "a_float64": part_a(kp), "b_sionna_float32": part_b(kp)}
    (ROOT / "results" / "P2" / "diag_derivatives.json").write_text(json.dumps(out, indent=1) + "\n")
    a = out["a_float64"]
    print(f"(a) paths {a['n_paths']}, best step {a['best_step_m']:g} m, max rel {a['max_rel_at_best_step']:.2e}, per step {a['max_rel_per_step']}")
    for lab in ("single", "double"):
        t = a["per_order"][lab]["table"]
        for h in STEPS_A:
            print(f"  {lab:6s} h={h:g}: " + "  ".join(f"{q} abs max {t[f'{h:g}|{q}']['max_abs']:.1e} med {t[f'{h:g}|{q}']['median_abs']:.1e} rel max {t[f'{h:g}|{q}']['max_rel']:.1e} "
                                                     f"med {t[f'{h:g}|{q}']['median_rel']:.1e}" for q in QUANT))
        print(f"  slopes {a['per_order'][lab]['loglog_slope_median_abs']}")
    b = out["b_sionna_float32"]
    for s, r in b["steps"].items():
        c = r["counts"]
        print(f"(b) {s}: nominal {c['paths_nominal']}, persisting {c['paths_all_offsets']}, agree {c['agree']} ({100 * c['share_paths_agree_of_persisting']:.0f} %), "
              f"causes {c['cause_exclusive']}, nonexclusive {c['criterion_nonexclusive']}, check {r['check_vs_p2_trace_1cm']}")
        print("     " + "  ".join(f"{q}: abs max {e['max_abs']:.1e} med {e['median_abs']:.1e} rel max {e['max_rel_well_defined']:.1e} med {e['median_rel_well_defined']:.1e} (near-zero {e['n_near_zero_an']})" for q, e in r["errors"].items()))
    print(f"(b) slopes {b['loglog_slope_median_abs']}")


if __name__ == "__main__":
    main()
