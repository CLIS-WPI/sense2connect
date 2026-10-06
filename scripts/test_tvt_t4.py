"""TVT T4 tests: tracker parts (BP association, map sources). Run: python scripts/test_tvt_t4.py"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_tvt import WL, _cuda  # noqa: E402


class TrackerPartsTest(unittest.TestCase):
    def test_bp_close_to_exact_marginals(self):
        from sim.tvt.bp import bp_marginals, exact_marginals

        rng = np.random.default_rng(8)
        worst = 0.0
        for _ in range(30):
            I, J = rng.integers(1, 5), rng.integers(1, 5)
            w0 = rng.uniform(0.05, 1.0, I)
            w = rng.exponential(1.0, (I, J)) * (rng.random((I, J)) < 0.7)
            b, _ = bp_marginals(w0, w)
            e = exact_marginals(w0, w)
            self.assertTrue(np.allclose(b.sum(1), 1.0))
            worst = max(worst, float(np.abs(b - e).max()))
        self.assertLess(worst, 0.1)  # loopy BP is approximate; typical errors are a few percent

    def test_bp_unique_assignment_is_exact(self):
        from sim.tvt.bp import bp_marginals, exact_marginals

        w0 = np.array([0.1, 0.2])
        w = np.array([[5.0, 0.0], [0.0, 3.0]])
        b, _ = bp_marginals(w0, w)
        self.assertTrue(np.allclose(b, exact_marginals(w0, w), atol=1e-9))

    def test_map_sources_reproduce_traced_delays(self):
        job = ROOT / "results" / "cache" / "lamppost" / "low" / "seed_1001"
        if not (job / "comm_geometry.npz").exists():
            self.skipTest("cache absent")
        from sim.tvt.sources import faces_from_geometry, predict, source_list, source_planes, street_faces, validity

        F = street_faces(faces_from_geometry())
        src = source_list(len(F))
        N, A, V, fid = source_planes(F, src)
        g = np.load(job / "comm_geometry.npz")
        miss = tot = 0
        for t in range(0, 600, 37):
            for u in range(2):
                for c in range(2):
                    ok = validity(g["oru_position_m"][t, c], g["ue_position_m"][t, u], F, src, N, A, V, fid)
                    pr = predict(g["oru_position_m"][t, c], g["ue_position_m"][t, u], N[ok], A[ok], V[ok])
                    pts, n = g["points_m"][t, u, c], g["n_points"][t, u, c]
                    for p in range(8):
                        if g["path_class"][t, u, c, p] < 0:
                            continue
                        L = np.sum(np.linalg.norm(np.diff(np.nan_to_num(pts[p][: n[p]]), axis=0), axis=1)) / 0.299792458
                        tot += 1
                        miss += int(np.min(np.abs(pr["tau"] - L)) > 0.02)
        self.assertLess(miss / tot, 0.05, f"{miss}/{tot} traced paths without a map source")

    def test_predict_jacobian_vs_finite_differences(self):
        from sim.tvt.sources import faces_from_geometry, predict, source_list, source_planes, street_faces, validity

        F = street_faces(faces_from_geometry())
        src = source_list(len(F))
        N, A, V, fid = source_planes(F, src)
        oru = np.array([12.0, 6.5, 5.0])
        ue = np.array([-3.0, -7.2, 1.5])
        ok = validity(oru, ue, F, src, N, A, V, fid)
        pr = predict(oru, ue, N[ok], A[ok], V[ok])
        h = 1e-5
        for k in range(2):
            e = np.zeros(3)
            e[k] = h
            pp, pm = predict(oru, ue + e, N[ok], A[ok], V[ok]), predict(oru, ue - e, N[ok], A[ok], V[ok])
            fd = np.stack([(pp["tau"] - pm["tau"]) / (2 * h), (pp["u"][:, 1] - pm["u"][:, 1]) / (2 * h), (pp["u"][:, 2] - pm["u"][:, 2]) / (2 * h)], -1)
            self.assertLess(np.max(np.abs(fd - pr["Jp"][:, :, k])), 1e-6)
        # face-offset Jacobian
        for slot in range(2):
            Ap, Am = A[ok].copy(), A[ok].copy()
            Ap[:, slot] += h * N[ok][:, slot]
            Am[:, slot] -= h * N[ok][:, slot]
            pp, pm = predict(oru, ue, N[ok], Ap, V[ok]), predict(oru, ue, N[ok], Am, V[ok])
            fd = np.stack([(pp["tau"] - pm["tau"]) / (2 * h), (pp["u"][:, 1] - pm["u"][:, 1]) / (2 * h), (pp["u"][:, 2] - pm["u"][:, 2]) / (2 * h)], -1)
            self.assertLess(np.max(np.abs(fd - pr["Jd"][:, :, slot])), 1e-6)

if __name__ == "__main__":
    unittest.main(verbosity=2)
