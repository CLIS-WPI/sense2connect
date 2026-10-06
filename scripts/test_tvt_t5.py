"""TVT T5 tests: risk-aware planner core and learned predictor. Run: python scripts/test_tvt_t5.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_tvt import _cuda  # noqa: E402


class RiskPlannerTest(unittest.TestCase):
    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_viterbi_costs_match_frozen_planner_decision(self):
        from run_m5_planner import plan_first_switch
        from tvt_t5_handover import viterbi_costs

        rng = np.random.default_rng(3)
        for tau in (0, 2):
            for epoch in (1, 3):
                bad = rng.random((300, 12, 2)) < 0.4
                cur = rng.integers(0, 2, 300)
                st, sw = viterbi_costs(bad, cur, tau, epoch)
                dec = plan_first_switch(bad, cur, tau, epoch)
                self.assertTrue(np.array_equal(sw < st, dec))

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_viterbi_costs_vs_exhaustive(self):
        from tvt_t5_handover import viterbi_costs

        rng = np.random.default_rng(4)
        for _ in range(100):
            hs, tau, epoch = int(rng.integers(2, 9)), int(rng.integers(0, 3)), int(rng.choice([1, 2]))
            bad = rng.random((1, hs, 2)) < 0.5
            c0 = int(rng.integers(0, 2))

            def brute(first_switch):
                best = None
                c = 1 - c0 if first_switch else c0
                p = tau if (first_switch and tau > 0) else 0
                acc = 1 if (first_switch and tau > 0) else int(bad[0, 0, c])
                stack = [(1, c, p, acc)]
                while stack:
                    k, cc, pp, aa = stack.pop()
                    if k == hs:
                        best = aa if best is None else min(best, aa)
                        continue
                    q = max(pp - 1, 0)
                    stack.append((k + 1, cc, q, aa + (1 if q > 0 else int(bad[0, k, cc]))))
                    if k % epoch == 0:
                        o = 1 - cc
                        stack.append((k + 1, o, tau, aa + (1 if tau > 0 else int(bad[0, k, o]))))
                return best

            st, sw = viterbi_costs(bad, np.array([c0]), tau, epoch)
            self.assertEqual(st[0], brute(False))
            self.assertEqual(sw[0], brute(True))

    def test_risk_rule(self):
        from tvt_t5_handover import risk_table

        K, R = 10, 5
        c = np.zeros((K, R, 2, 2))
        c[..., 0] = 5.0  # stay
        c[..., 1] = 4.0  # switch
        costs = {((1, "a", "b"), 0): c}
        idx = [(0, (1, "a", "b"), 0)]
        self.assertTrue(risk_table(costs, idx, 0.0, 0.0).all())
        self.assertFalse(risk_table(costs, idx, 0.0, 2.0).any())  # theta larger than the gain
        # a heavy tail of the switch plan: mean better, CVaR worse -> lambda flips the decision
        c2 = c.copy()
        c2[:, :, :, 1] = 3.0
        c2[0, :, :, 1] = 30.0  # one bad sample
        costs = {((1, "a", "b"), 0): c2}
        self.assertFalse(risk_table(costs, idx, 0.0, 0.0).all())  # mean 5.7 > 5
        c3 = c.copy()
        c3[:, :, :, 1] = 4.0
        c3[0, :, :, 1] = 9.0  # mean 4.5 < 5, CVaR_0.9 = 9 > 5
        costs = {((1, "a", "b"), 0): c3}
        self.assertTrue(risk_table(costs, idx, 0.0, 0.0).all())
        self.assertFalse(risk_table(costs, idx, 1.0, 0.0).any())


class LearnedTest(unittest.TestCase):
    def test_features_and_loss_conversion(self):
        from sim.tvt.learned import N_FEAT, TAUS, features, labels, predict_loss

        oru = np.array([[12.0, 6.5, 5.0], [-28.0, -7.0, 5.0]])
        ue = np.array([[0.0, -7.0, 1.5], [10.0, -7.0, 1.5]])
        vel = np.array([[1.2, 0, 0], [1.0, 0, 0]])
        tracks = {"valid": np.array([[True, False]]), "mean": np.array([[[5.0, -2.5, 1.6, 10.0, 0.0], [0, 0, 0, 0, 0]]]),
                  "cov": np.tile(np.eye(5) * 0.1, (1, 2, 1, 1)), "size": np.array([[[12.0, 2.5, 3.2], [1, 1, 1]]]), "kind": np.array([[0, 0]])}
        f = features(oru, ue, vel, tracks, 0)
        self.assertEqual(f.shape, (2, 2, N_FEAT))
        self.assertEqual(f[0, 0, 3 + 8], 1.0)  # first track slot present
        self.assertEqual(f[0, 0, 3 + 9 + 8], 0.0)  # second slot empty
        los = np.zeros((600, 2, 2))
        los[100:150, 0, 1] = 20.0
        y = labels(los, 9, 2, 2)
        self.assertEqual(y.shape, (2, 2, len(TAUS)))
        self.assertEqual(y[0, 1, list(TAUS).index(0.25)], 1.0)
        prob = np.zeros((3, 2, 2, len(TAUS)))
        prob[1, 0, 1, :] = 0.8
        loss = predict_loss(prob, np.arange(0, 3.01, 0.5), 0.5)
        self.assertTrue(np.all(loss[1, 0, 1] == 200.0))
        self.assertTrue(np.all(loss[0] == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
