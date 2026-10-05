"""Paper 2 diagnostic (after the review; development seeds; no change to the frozen pipeline): sources of the estimator's error tail.

For the main configuration (bw400_tdoa_s1_p2_b; EKF output of results/P2/est/dev),
(--variant v1 | A | AB, --set dev | ...) every epoch with error > 0.5 m (the "tail") is attributed, in this order, to
(a) a FRONT/BACK MIRROR solution: the estimate is closer to the mirror image of
    the true UE across the array plane (x = x_O-RU, y-z plane) of either O-RU than
    to the true UE, the mirror image being > 1 m from the truth (the planar array
    cannot tell the two sides apart); strict variant: also within 1 m of the mirror;
(b) a REFLECTED PATH dominating at an O-RU: the measured spatial frequency
    (u_y, u_z) of at least one O-RU is > 0.02 from its true LoS direction and closer
    to a reflected path's direction than to the LoS (paper-1 geometry cache);
(c) other (EKF lag after an outlier, multipath bias of the LoS estimate, ...).
Reported per mount and blocked/unblocked (blocked = at least one O-RU with LoS
model-B loss >= 10 dB): share of tail epochs, share of the total squared error
(RMSE^2), RMSE and p90 with the class's tail epochs excluded. Also on the raw
WLS fixes. Writes results/P2/diag_tail_dev.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = "bw400_tdoa_s1_p2_b"
TAIL_M = 0.5


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1", choices=["v1", "A", "AB"])
    ap.add_argument("--set", default="dev")
    args = ap.parse_args()
    base = ROOT / "results" / "P2" / ("est" if args.variant == "v1" else f"est_{args.variant}") / args.set / CFG
    mbase = ROOT / "results" / "P2" / "est" / args.set / CFG
    recs = []
    for f in sorted(base.glob("*_track.npz")):
        mount, density, seed = f.stem.replace("_track", "").split("_")
        tr = np.load(f)
        ms = np.load(mbase / f.name.replace("_track", ""))
        g = np.load(ROOT / "results" / "cache" / mount / density / f"seed_{seed}" / "comm_geometry.npz")
        ue = ms["ue"][..., :2]  # [T, U, 2]
        oru = ms["oru"]  # [T, C, 3]
        T, U = ue.shape[:2]
        pts = np.nan_to_num(g["points_m"])
        first = pts[..., 1, :] - pts[..., 0, :]
        u_path = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)  # [T, U, C, P, 3]
        cls = g["path_class"]
        u_los = np.where((cls == 0)[..., None], u_path, np.nan)
        u_los = np.nanmean(u_los, axis=-2)  # [T, U, C, 3]
        meas = np.stack([ms["uy"], ms["uz"]], -1)  # [T, U, C, 2]
        d_los = np.linalg.norm(meas - u_los[..., 1:], axis=-1)
        d_nlos = np.linalg.norm(meas[..., None, :] - u_path[..., 1:], axis=-1)
        d_nlos = np.where(cls > 0, d_nlos, np.inf).min(-1)
        reflected = ((d_los > 0.02) & (d_nlos < d_los)).any(-1)  # [T, U]
        for kind, est in (("ekf", tr["xy_ekf"]), ("fix", tr["xy_fix"])):
            err = np.linalg.norm(est - ue, axis=-1)
            err = np.where(np.isfinite(err), err, np.nan)
            mirror = np.zeros((T, U), dtype=bool)
            mirror_strict = np.zeros((T, U), dtype=bool)
            for c in range(oru.shape[1]):
                mir = ue.copy()
                mir[..., 0] = 2 * oru[:, None, c, 0] - ue[..., 0]
                dm = np.linalg.norm(est - mir, axis=-1)
                sep = np.linalg.norm(mir - ue, axis=-1)
                m = (dm < err) & (sep > 1.0)
                mirror |= m
                mirror_strict |= m & (dm < 1.0)
            blocked = ms["blocked"].any(-1)
            recs.append({"kind": kind, "mount": mount, "seed": int(seed), "err": err[20:], "mirror": mirror[20:], "mirror_strict": mirror_strict[20:],
                         "reflected": reflected[20:], "blocked": blocked[20:]})
    out = {"definition": __doc__, "config": CFG, "tail_m": TAIL_M, "results": {}}
    lines = []
    for kind in ("ekf", "fix"):
        for mount in ("all", "lamppost", "facade"):
            for state in ("all", "unblocked", "blocked"):
                rs = [r for r in recs if r["kind"] == kind and (mount == "all" or r["mount"] == mount)]
                cat = lambda k: np.concatenate([r[k].ravel() for r in rs])  # noqa: E731
                e, mi, ms_, rf, bl = cat("err"), cat("mirror"), cat("mirror_strict"), cat("reflected"), cat("blocked")
                sel = np.isfinite(e) & (np.ones_like(bl) if state == "all" else (bl if state == "blocked" else ~bl))
                e, mi, ms_, rf = e[sel], mi[sel], ms_[sel], rf[sel]
                tail = e > TAIL_M
                cls_a = tail & mi
                cls_b = tail & ~mi & rf
                cls_c = tail & ~mi & ~rf
                se = np.sum(e ** 2)
                res = {"n_epochs": int(e.size), "tail_share_of_epochs": float(tail.mean()), "rmse_m": float(np.sqrt(np.mean(e ** 2))), "p90_m": float(np.percentile(e, 90)),
                       "mirror_strict_share_of_tail": float((tail & ms_).sum() / max(tail.sum(), 1))}
                for name, m in (("a_mirror", cls_a), ("b_reflected", cls_b), ("c_other", cls_c)):
                    keep = ~m
                    res[name] = {"share_of_tail": float(m.sum() / max(tail.sum(), 1)), "share_of_sq_error": float(np.sum(e[m] ** 2) / se),
                                 "rmse_without_m": float(np.sqrt(np.mean(e[keep] ** 2))), "p90_without_m": float(np.percentile(e[keep], 90))}
                out["results"][f"{kind}|{mount}|{state}"] = res
                lines.append(f"| {kind} | {mount} | {state} | {res['tail_share_of_epochs']:.3f} | {res['rmse_m']:.2f} | {res['p90_m']:.2f} | "
                             + " | ".join(f"{res[k]['share_of_tail']:.2f} / {res[k]['share_of_sq_error']:.2f} / {res[k]['rmse_without_m']:.2f} / {res[k]['p90_without_m']:.2f}"
                                          for k in ("a_mirror", "b_reflected", "c_other")) + f" | {res['mirror_strict_share_of_tail']:.2f} |")
    out["variant"], out["set"] = args.variant, args.set
    suffix = "" if args.variant == "v1" else f"_{args.variant}"
    (ROOT / "results" / "P2" / f"diag_tail_{args.set}{suffix}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("| est | mount | state | tail share | RMSE | p90 | (a) mirror: tail share / sq-err share / RMSE w/o / p90 w/o | (b) reflected | (c) other | (a) strict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
