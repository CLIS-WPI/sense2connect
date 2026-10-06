"""TVT T2 tests: multipath extraction. Run: python scripts/test_tvt_t2.py"""

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


class ExtractTest(unittest.TestCase):
    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_recovers_three_paths(self):
        import torch

        from sim.positioning.array import element_positions
        from sim.positioning.estimator import synth_torch
        from sim.tvt.extract import extract

        r = element_positions(WL)
        f = (np.arange(792) - 395.5) * 120e3
        B = 20
        tau = np.array([40.0, 57.3, 95.1])
        u = np.array([[0.6, 0.79, 0.1], [0.3, -0.9, -0.3], [-0.5, 0.5, -0.708]])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        beta = np.array([3.0, 1.2 * np.exp(1j), 0.8 * np.exp(-2j)])
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1)
        Y = synth_torch(torch.as_tensor(np.tile(beta, (B, 1)), device="cuda"), torch.as_tensor(np.tile(tau, (B, 1)), device="cuda"),
                        torch.as_tensor(np.tile(u, (B, 1, 1)), device="cuda"), f, r, WL, gen=gen)
        ex = extract(Y, f, WL, dyn_range_db=40.0)
        for b in range(B):
            v = ex["valid"][b]
            self.assertEqual(int(v.sum()), 3)
            for p in range(3):
                d = np.abs(ex["tau_ns"][b][v] - tau[p]) + 50 * np.abs(ex["uy"][b][v] - u[p, 1]) + 50 * np.abs(ex["uz"][b][v] - u[p, 2])
                j = int(np.argmin(d))
                self.assertLess(abs(ex["tau_ns"][b][v][j] - tau[p]), 0.3)
                self.assertLess(abs(ex["uy"][b][v][j] - u[p, 1]), 0.01)
                self.assertLess(abs(abs(ex["beta"][b][v][j]) - abs(beta[p])) / abs(beta[p]), 0.05)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_noise_only_false_alarm_rate(self):
        import torch

        from sim.tvt.extract import extract

        f = (np.arange(792) - 395.5) * 120e3
        B = 400
        gen = torch.Generator(device="cuda")
        gen.manual_seed(2)
        Y = ((torch.randn((B, 792, 64), generator=gen, device="cuda") + 1j * torch.randn((B, 792, 64), generator=gen, device="cuda")) / math.sqrt(2)).to(torch.complex64)
        ex = extract(Y, f, WL, pfa=0.05)
        rate = ex["valid"][:, 0].mean()
        self.assertLess(rate, 0.08)  # per-snapshot false alarm <= pfa (threshold offset calibrated on noise)

if __name__ == "__main__":
    unittest.main(verbosity=2)
