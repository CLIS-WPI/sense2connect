"""Paper 2: stored verification statistics behind the paper's method claims (no change to frozen code or tests).

1. Analytic vs re-traced derivatives (the +/-1 cm finite-difference subset written by
   scripts/p2_trace.py --fd; same comparison as scripts/test_p2.py
   GeometryTest.test_nlos_derivative_vs_retrace): share of DERIVATIVE values (path x {delay,
   azimuth, elevation} x {x, y}) and share of PATHS (all six derivatives of a path in an epoch)
   whose analytic image-method derivative agrees with the re-trace to 1e-3 relative.
2. GPU Fisher information vs references: the achieved deviations of the unit tests
   (scripts/test_p2.py FisherTest: GPU Gram vs explicit NumPy synthesis, relative to the largest
   entry; full GPU pipeline PEB vs a brute-force numerical FIM with sync and calibration priors) and
   of the job-level GPU vs NumPy reference check of scripts/p2_peb.py (results/P2/peb/<set>/meta.json).
Writes results/P2/verification_stats.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def deriv_agreement() -> dict:
    from sim.positioning.geometry import bounce_planes, path_geometry
    from sim.scenes.config import load_yaml

    kp = load_yaml(ROOT / "configs" / "p2.yaml")["known_planes"]
    files = sorted((ROOT / "results" / "P2" / "cache").glob("*/*/seed_*/p2_fd.npz"))
    n_val = n_val_ok = n_path = n_path_ok = 0
    for fp in files:
        mount, density, sd = fp.parts[-4], fp.parts[-3], fp.parts[-2]
        d = np.load(ROOT / "results" / "cache" / mount / density / sd / "comm_geometry.npz")
        f = np.load(fp)
        h = float(np.abs(f["offsets"]).max())
        ns = f["fd_tau"].shape[0]
        P, n, cls = d["points_m"][:ns], d["n_points"][:ns].astype(int), d["path_class"][:ns]
        ue = np.broadcast_to(d["ue_position_m"][:ns, :, None, None, :], P.shape[:-2] + (3,))
        oru = np.broadcast_to(d["oru_position_m"][:ns, None, :, None, :], P.shape[:-2] + (3,))
        N, A, V = bounce_planes(P, n, kp)
        g = path_geometry(oru, ue, N, A, V)
        uu = f["fd_u"]
        az = np.arctan2(uu[..., 1], uu[..., 0])
        el = np.arcsin(np.clip(uu[..., 2], -1, 1))
        ok_all = np.ones(cls.shape, dtype=bool)
        have = (cls >= 0)
        for k, (ip, im) in enumerate(((0, 1), (2, 3))):
            pairs = (((f["fd_tau"][..., ip] - f["fd_tau"][..., im]) / (2 * h), g["dtau"][..., k]),
                     (np.angle(np.exp(1j * (az[..., ip] - az[..., im]))) / (2 * h), g["daz"][..., k]),
                     ((el[..., ip] - el[..., im]) / (2 * h), g["del"][..., k]))
            for fd, an in pairs:
                valid = have & np.isfinite(fd)
                agree = np.abs(fd - an) <= 1e-3 * np.abs(an)
                n_val += int(valid.sum())
                n_val_ok += int((valid & agree).sum())
                have &= np.isfinite(fd)
                ok_all &= agree | ~np.isfinite(fd)
        n_path += int(have.sum())
        n_path_ok += int((have & ok_all).sum())
    return {"files": [str(p.relative_to(ROOT)) for p in files], "derivative_values": n_val, "values_within_1e-3": n_val_ok,
            "share_values": n_val_ok / max(n_val, 1), "paths": n_path, "paths_all_within_1e-3": n_path_ok, "share_paths": n_path_ok / max(n_path, 1)}


def fim_deviations() -> dict:
    import test_p2 as T

    rec = []

    class Rec(T.FisherTest):
        def assertLess(self, a, b, msg=None):  # record the achieved value instead of asserting
            rec.append((self._testMethodName, float(a), float(b)))

    out = {}
    for name, tol in (("test_gram_torch_equals_numpy", 1e-9), ("test_efim_vs_bruteforce_numeric_fim", 1e-3)):
        rec.clear()
        getattr(Rec(name), name)()
        # gram test asserts |Jn - Jt|max < 1e-9 * scale -> relative deviation = a / (b / tol); PEB test asserts rel < tol
        rel = [a / (b / tol) if name == "test_gram_torch_equals_numpy" else a for _, a, b in rec]
        out[name] = {"max_relative_deviation": max(rel), "test_tolerance": tol}
    for tag in ("heldout2", "heldout", "dev"):
        mf = ROOT / "results" / "P2" / "peb" / tag / "meta.json"
        if mf.exists():
            out[f"job_gpu_vs_numpy_{tag}"] = json.loads(mf.read_text())["gpu_vs_numpy_max_rel"]
    out["largest_test_deviation"] = max(out[k]["max_relative_deviation"] for k in ("test_gram_torch_equals_numpy", "test_efim_vs_bruteforce_numeric_fim"))
    return out


def main() -> None:
    res = {"definition": __doc__, "derivatives": deriv_agreement(), "fim": fim_deviations()}
    (ROOT / "results" / "P2" / "verification_stats.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, indent=1)[:2500])


if __name__ == "__main__":
    main()
