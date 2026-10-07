"""Unit tests of the TVT extension (ROADMAP_TVT.md). Plain unittest; GPU tests skip without CUDA.

Run: python scripts/test_tvt.py [TestCase.test_name]
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WL = 299_792_458.0 / 28e9


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class SeedGuardTest(unittest.TestCase):
    def test_sets_disjoint_and_heldout_not_loaded(self):
        from sim.tvt.seeds import load

        s = load()
        self.assertEqual(s["development"], list(range(1001, 1011)))
        self.assertEqual(len(s["training"]), 40)
        self.assertNotIn("heldout", s)
        allv = [x for v in s.values() for x in v]
        self.assertEqual(len(allv), len(set(allv)))

    def test_heldout_refused_without_tag(self):
        from sim.tvt.seeds import HeldOutRefused, check, heldout

        with tempfile.TemporaryDirectory() as g:
            (Path(g) / "refs" / "tags").mkdir(parents=True)
            with self.assertRaises(HeldOutRefused):
                heldout(Path(g))
            with self.assertRaises(HeldOutRefused):
                check([1001, 4003], Path(g))
            self.assertEqual(check([1001, 101], Path(g)), [1001, 101])

    def test_heldout_opened_with_tag_consumed_never(self):
        from sim.tvt.seeds import HeldOutRefused, check, heldout

        with tempfile.TemporaryDirectory() as g:
            (Path(g) / "refs" / "tags").mkdir(parents=True)
            (Path(g) / "packed-refs").write_text("abc123 refs/tags/tvt-freeze\n")
            self.assertEqual(heldout(Path(g))[0], 4001)
            self.assertEqual(check([4001], Path(g)), [4001])
            with self.assertRaises(HeldOutRefused):
                check([3001], Path(g))
            with self.assertRaises(HeldOutRefused):
                check([2005], Path(g))

    def test_real_repo_has_no_freeze_tag(self):
        from sim.tvt.seeds import freeze_tag_exists

        self.assertFalse(freeze_tag_exists(), "tvt-freeze exists: held-out evaluation is allowed only in T7")


class ServiceModelTest(unittest.TestCase):
    def test_band_kernel_matches_explicit_mean(self):
        from sim.tvt.service import band_kernel

        f = (np.arange(1024) - 511.5) * 120e3
        tau = np.array([10e-9, 13.7e-9, 55e-9, 10e-9])
        k = band_kernel(tau, f)
        ref = np.exp(-2j * np.pi * f[None, None, :] * (tau[:, None, None] - tau[None, :, None])).mean(-1)
        self.assertLess(np.max(np.abs(k - ref)), 1e-10)

    def test_codebook_unit_norm_and_beam_centre_gain(self):
        from sim.positioning.array import element_positions, steering
        from sim.tvt.service import codebook

        W = codebook(WL, 2)
        self.assertEqual(W.shape, (256, 64))
        self.assertTrue(np.allclose(np.linalg.norm(W, axis=1), 1.0))
        # a plane wave from a beam centre direction gets the full array gain 64
        uy, uz = -1 + 3 / 16, -1 + 21 / 16
        u = np.array([math.sqrt(1 - uy * uy - uz * uz), uy, uz])
        s = steering(u, element_positions(WL), WL)
        self.assertAlmostEqual(float(np.max(np.abs(W.conj() @ s) ** 2)), 64.0, places=9)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_single_path_models(self):
        from sim.tvt.service import gains

        rng = np.random.default_rng(0)
        f = (np.arange(1024) - 511.5) * 120e3
        T, P = 50, 4
        a = np.zeros((T, 1, 1, P), complex)
        a[..., 0] = rng.standard_normal((T, 1, 1)) + 1j * rng.standard_normal((T, 1, 1))
        u = rng.standard_normal((T, 1, 1, P, 3))
        u[..., 0] = np.abs(u[..., 0])
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        tau = rng.uniform(10e-9, 100e-9, (T, 1, 1, P))
        g = np.ones((T, 1, 1, P))
        ps = gains("power_sum", a, u, tau, g, WL, f)["total"]
        mrt = gains("mrt", a, u, tau, g, WL, f)["total"]
        bb = gains("best_beam", a, u, tau, g, WL, f)["total"]
        self.assertTrue(np.allclose(mrt, ps, rtol=1e-9))
        self.assertTrue(np.all(bb <= mrt * (1 + 1e-9)))
        self.assertTrue(np.all(bb >= 0.3 * mrt))  # worst-case 2x-oversampled DFT scalloping in 2-D is about -4 dB

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_multipath_best_beam_below_mrt(self):
        from sim.tvt.service import gains

        rng = np.random.default_rng(1)
        f = (np.arange(1024) - 511.5) * 120e3
        T, P = 40, 6
        a = rng.standard_normal((T, 1, 2, P)) + 1j * rng.standard_normal((T, 1, 2, P))
        u = rng.standard_normal((T, 1, 2, P, 3))
        u[..., 0] = np.abs(u[..., 0])
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        tau = rng.uniform(10e-9, 100e-9, (T, 1, 2, P))
        g = rng.uniform(0, 1, (T, 1, 2, P))
        mrt = gains("mrt", a, u, tau, g, WL, f)["total"]
        bb = gains("best_beam", a, u, tau, g, WL, f)["total"]
        self.assertTrue(np.all(bb <= mrt * (1 + 1e-9)))

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_power_sum_reproduces_frozen_timeline(self):
        job = ROOT / "results" / "cache" / "lamppost" / "low" / "seed_1001"
        if not (job / "m3_timeline.npz").exists():
            self.skipTest("paper-1 timeline cache absent")
        import tvt_service as S
        from sim.scenes.config import load_yaml
        from sim.tvt.service import per_path_loss_timeline, timeline_fields
        from xapp.timeline import held_segments, load_geometry

        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        _, act = S._actors((raw, (1001, "lamppost", "low")))
        geom, meta = load_geometry(job)
        n = 400
        times = np.arange(n) * 0.01
        seg = held_segments(geom, times, act["ue_position_m"][:n], 0.1)
        loss = per_path_loss_timeline(seg, act["blocker_position_m"][:n], act["blocker_size_m"], float(meta["wavelength_m"]))
        a = np.load(ROOT / "results" / "P2" / "cache" / "lamppost" / "low" / "seed_1001" / "p2_paths.npz")["a_center"]
        out = timeline_fields("power_sum", seg, loss, a, float(meta["wavelength_m"]), (np.arange(1024) - 511.5) * 120e3, 10)
        fr = np.load(job / "m3_timeline.npz")
        for k, v in out.items():
            ref = fr[k][:n]
            m = ref > 0
            self.assertTrue(np.array_equal(v > 0, m), k)
            self.assertLess(np.max(np.abs(v[m] - ref[m]) / ref[m]), 1e-6, k)  # |a|^2 64 vs cached power (3e-7)

class FrozenParamsTest(unittest.TestCase):
    def test_frozen_configs_match_sources(self):
        """configs/tvt_frozen: every recorded source still has the recorded sha256 (if present) and the learned weights match."""
        import hashlib
        import json

        d = ROOT / "configs" / "tvt_frozen"
        man = json.loads((d / "manifest.json").read_text())
        for name, info in man["files"].items():
            self.assertEqual(hashlib.sha256((d / name).read_bytes()).hexdigest(), info["sha256"], name)
            for src, h in info["sources"].items():
                f = ROOT / src
                if f.exists():
                    self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), h, f"{name}: source {src} changed after the freeze")
        lj = json.loads((d / "learned.json").read_text())
        w = ROOT / lj["path"]
        if w.exists():
            self.assertEqual(hashlib.sha256(w.read_bytes()).hexdigest(), lj["sha256"])
        for m in ("ideal", "failure_aware"):
            ho = json.loads((d / f"handover_{m}.json").read_text())
            self.assertEqual(ho["signaling"], m)
            for lab, cell in ho["schemes"].items():
                for sch in ("A5", "CHO", "planner_tvt", "risk_tvt", "planner_tvt_perfect", "trigger_learned"):
                    self.assertIn("params", cell[sch], f"{m} {lab} {sch}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
