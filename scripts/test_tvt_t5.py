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


class J3ClassifierTest(unittest.TestCase):
    def test_error_types(self):
        """Synthetic lane: every outage step gets the error type of the definition in scripts/tvt_j3_diagnosis.py."""
        from sim.comm.phy import MAX_NR_SE
        from tvt_j3_diagnosis import CATS, classify_lane

        n_t, tau, ovh = 1000, 2, 0.1
        meta = {"bandwidth_hz": 400e6, "rate_req_bps": 400e6, "dt_s": 0.01, "tau_ho_steps": tau}
        snr = np.full((n_t, 2), 20.0)
        snr[100:200, 0] = -10.0  # cell 0 bad while serving, no switch -> missed
        snr[300:400, 0] = -10.0  # cell 0 bad while serving, switch at 350 -> late (mistimed) + necessary interruption
        snr[640:700, 0] = -10.0  # switch 1 -> 0 at 650 into the already bad cell 0 while cell 1 is usable -> unnecessary
        snr[800:820, 1] = 0.3  # usable without, not with the overhead -> overhead
        snr[900:910, :] = -10.0  # both bad -> unavoidable
        hos = {"step": np.array([350, 650]), "from": np.array([0, 1]), "to": np.array([1, 0])}
        serv = np.zeros(n_t, np.int64)
        serv[350:650] = 1
        serv[650:] = 0
        serv[700:] = 1  # (synthetic: serving trace only needs to be consistent with the steps classified below)
        intr = np.zeros(n_t, bool)
        for st in hos["step"]:
            intr[st:st + tau] = True
        se = np.minimum(np.log2(1.0 + 10.0 ** (snr / 10.0)), MAX_NR_SE)
        rate = np.where(intr, 0.0, 400e6 * (1 - ovh) * se[np.arange(n_t), serv])
        outr = intr | (rate < 400e6)
        cat, cls, sw = classify_lane(serv, outr, intr, snr, ovh, hos, [], meta)
        name = lambda k: CATS[cat[k]] if cat[k] >= 0 else None  # noqa: E731
        self.assertEqual(name(150), "missed_switch")
        self.assertEqual(name(320), "mistimed_switch")
        self.assertEqual(name(350), "switch_necessary")
        self.assertEqual(name(650), "unnecessary_switch")
        self.assertEqual(name(680), "unnecessary_switch")
        self.assertEqual(name(810), "overhead")
        self.assertEqual(name(905), "unavoidable")
        self.assertIsNone(name(500))
        self.assertEqual([x["necessary"] for x in sw], [True, False])
        self.assertEqual(int((cat >= 0).sum()), int(outr.sum()))


class SignalingTest(unittest.TestCase):
    @staticmethod
    def _lanes(snr, scheme, **kw):
        from xapp.schemes import Lanes

        n = snr.shape[0]
        f = lambda v: np.full(n, v)  # noqa: E731
        base = dict(snr_db=snr, scheme=np.asarray(scheme), offset_db=f(1.0), hysteresis_db=f(1.0), ttt_steps=f(4), filter_a=f(0.5), window_steps=f(20),
                    drop_db=f(10.0), trend_horizon_s=f(0.5), hold_steps=f(50), tau_ho_steps=f(2), e2_delay_steps=f(2), overhead=f(0.0),
                    initial_cell=np.zeros(n, np.int64), a5_thr1=f(5.0), a5_thr2=f(8.0))
        base.update(kw)
        return Lanes(**base)

    @staticmethod
    def _sig():
        from sim.scenes.config import load_yaml

        return load_yaml(ROOT / "configs" / "tvt.yaml")["signaling"]

    def test_ideal_equals_paper1_simulator(self):
        from sim.tvt.signaling import simulate_fa
        from xapp.schemes import REASON, simulate

        rng = np.random.default_rng(1)
        n, t = 8, 1500
        snr = np.cumsum(rng.normal(0, 1.5, (n, t, 2)), axis=1) * 0.3 + 15.0
        snr[:, 300:420, 0] -= 30.0
        snr[:, 900:980, 1] -= 30.0
        scheme = np.array([REASON["a3"], REASON["a5"], REASON["xapp"], REASON["trend"]] * 2)
        trig = rng.random((n, t // 10, 2)) < 0.05
        lanes = self._lanes(snr, scheme, trigger=trig, trigger_end_steps=np.full((n, t // 10, 2), 30))
        kw = dict(bandwidth_hz=400e6, rate_req_bps=400e6, max_se=7.4)
        a = simulate(lanes, **kw)
        b = simulate_fa(lanes, **kw, sig=None, cho=np.array([0, 1, 0, 0, 0, 1, 0, 0], bool))
        for key in ("serving", "outage_req", "outage0", "interrupted", "rate"):
            np.testing.assert_array_equal(a[key], b[key], key)
        for key in a["handovers"]:
            np.testing.assert_array_equal(a["handovers"][key], b["handovers"][key], key)

    def test_rlf_and_reestablishment(self):
        from sim.tvt.signaling import simulate_fa
        from xapp.schemes import REASON

        sig = self._sig()
        t = 1200
        snr = np.full((2, t, 2), 20.0)
        snr[0, 100:, 0] = -15.0  # lane 0: serving cell 0 lost for good, no handover (offset huge) -> RLF
        snr[1, 100:130, 0] = -15.0  # lane 1: short dip, recovers before T310 expires
        lanes = self._lanes(snr, np.array([REASON["a3"]] * 2), offset_db=np.full(2, 1e9))
        out = simulate_fa(lanes, bandwidth_hz=400e6, rate_req_bps=400e6, max_se=7.4, sig=sig)
        f = out["failures"]
        self.assertEqual(f["rlf_lane"].tolist(), [0])
        # Qout window 20 steps: the mean of the dB samples falls below -8 dB with m = 17 low samples
        # (400 - 35 m < -160 -> m > 16), first indication at the next multiple of 2 steps, T310 = 100 steps
        t_oos = 100 + 16
        t_oos += (-t_oos) % 2
        self.assertEqual(int(f["rlf_step"][0]), t_oos + 100)
        tau = int(round(sig["reestablishment"]["tau_re_s"] / 0.01))
        self.assertEqual(int(f["re_end"][0]), int(f["rlf_step"][0]) + 1 + tau)
        self.assertTrue(out["outage_req"][0, int(f["rlf_step"][0]) + 1: int(f["re_end"][0])].all())
        self.assertEqual(int(out["serving"][0, -1]), 1)  # re-established on the best cell
        self.assertFalse(out["outage_req"][1, 200:].any())

    def test_command_failure_and_cho(self):
        from sim.tvt.signaling import simulate_fa
        from xapp.schemes import REASON

        sig = self._sig()
        t = 600
        snr = np.full((2, t, 2), 20.0)
        snr[:, 100:, 0] = -20.0  # serving collapses abruptly; other cell good
        lanes = self._lanes(snr, np.array([REASON["a5"], REASON["a5"]]), filter_a=np.full(2, 0.02))
        out = simulate_fa(lanes, bandwidth_hz=400e6, rate_req_bps=400e6, max_se=7.4, sig=sig, cho=np.array([False, True]))
        f = out["failures"]
        ho = out["handovers"]
        # the slowly filtered A5 condition fires only after the Qout quality is below Qout -> the command fails (lane 0)
        self.assertGreater(int((f["hof_lane"] == 0).sum()), 0)
        self.assertEqual(int((f["hof_lane"] == 1).sum()), 0)  # CHO: no command on the critical path
        self.assertIn(1, ho["lane"].tolist())
        self.assertEqual(int(out["serving"][1, 300]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
