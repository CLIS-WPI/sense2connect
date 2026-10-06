"""TVT T3 tests: visibility prediction. Run: python scripts/test_tvt_t3.py"""

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


class VisibilityTest(unittest.TestCase):
    JOB = ROOT / "results" / "cache" / "lamppost" / "low" / "seed_1001"

    def test_polyline_reproduces_cached_points(self):
        if not (self.JOB / "comm_geometry.npz").exists():
            self.skipTest("cache absent")
        from sim.positioning.geometry import bounce_planes
        from sim.scenes.config import load_yaml
        from sim.tvt.visibility import polyline

        kp = load_yaml(ROOT / "configs" / "p2.yaml")["known_planes"]
        g = np.load(self.JOB / "comm_geometry.npz")
        P, n, cls = g["points_m"][:100], g["n_points"][:100].astype(int), g["path_class"][:100]
        N, A, V = bounce_planes(P, n, kp)
        oru = np.broadcast_to(g["oru_position_m"][:100, None, :, None, :], P.shape[:-2] + (3,))
        ue = np.broadcast_to(g["ue_position_m"][:100, :, None, None, :], P.shape[:-2] + (3,))
        pts = polyline(oru, ue, N, A, V)
        for k in (1, 2):
            m = (cls >= 0) & (n >= k + 2)
            d = np.linalg.norm(pts[..., k, :] - np.nan_to_num(P[..., k, :]), axis=-1)[m]
            self.assertLess(np.percentile(d, 99), 0.01, f"bounce {k}")  # cached points are rounded to 1 mm; snapped planes

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_perfect_tracks_true_ue_reproduce_labels(self):
        if not (self.JOB / "comm_geometry.npz").exists():
            self.skipTest("cache absent")
        import tvt_t3_visibility as V3
        from sim.scenes.config import load_yaml

        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        p2 = load_yaml(ROOT / "configs" / "p2.yaml")
        tc = load_yaml(ROOT / "configs" / "tvt.yaml")
        jd = V3.job_data((1001, "lamppost", "low"), raw, p2, tc, real=False)
        lab, ok, _ = V3.labels(jd, 0.0)
        q = V3.predict_job(jd, "perfect", "true", 0.0, 1, tc, None, np.random.default_rng(0))
        agree = ((q < 0.5) == lab)[ok].mean()
        self.assertGreater(agree, 0.995, agree)

if __name__ == "__main__":
    unittest.main(verbosity=2)
