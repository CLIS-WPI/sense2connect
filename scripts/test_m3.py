"""Unit tests for the M3 PHY, predictor, and A3 trigger."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.phy import apply_sensing_overhead, shannon_bps, snr_linear, steering_upa
from xapp.baselines import a3_trigger, l3_filter, trend_trigger
from xapp.policy import should_handover, should_return
from xapp.predict import blockage_window, predict_positions
from xapp.simulate import interpolate_series, lerp_loss_db


class PhyTest(unittest.TestCase):
    def test_steering_is_unit_norm(self) -> None:
        weights = steering_upa(8, 8, 0.5 * math.pi, 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(weights)), 1.0, places=9)

    def test_shannon_caps_at_nr_mcs27(self) -> None:
        self.assertAlmostEqual(shannon_bps(1e6, 100e6) / 100e6, 7.4063, places=4)

    def test_sensing_overhead_is_one_fourteenth(self) -> None:
        self.assertAlmostEqual(apply_sensing_overhead(14.0, 1.0 / 14.0), 13.0, places=9)

    def test_snr_uses_tx_and_noise_figure(self) -> None:
        low = snr_linear(1e-8, 20.0, 122.88e6, 7.0, 290.0)
        high = snr_linear(1e-8, 30.0, 122.88e6, 7.0, 290.0)
        self.assertGreater(high, low)


class PredictTest(unittest.TestCase):
    def test_lane_track_stays_on_the_line(self) -> None:
        track = {"x_m": 0.0, "y_m": 0.2, "z_m": 1.0, "vx_mps": 10.0, "vy_mps": 3.0, "mode": "lane", "line_y_m": -2.5}
        pos = predict_positions(track, np.array([0.0, 1.0]))
        self.assertAlmostEqual(pos[1, 0], 10.0, places=6)
        self.assertAlmostEqual(pos[1, 1], -2.5, places=6)

    def test_blockage_window(self) -> None:
        losses = np.array([0.0, 2.0, 12.0, 15.0, 4.0, 1.0])
        times = np.arange(6, dtype=np.float64) * 0.1
        start, end = blockage_window(losses, times, 10.0, 3.0)
        self.assertAlmostEqual(start, 0.2)
        self.assertAlmostEqual(end, 0.4)


class PolicyTest(unittest.TestCase):
    def test_handover_needs_the_actionable_window(self) -> None:
        serving = {"start_s": 0.3, "end_s": 1.0, "clear": False}
        other = {"start_s": None, "end_s": None, "clear": True}
        self.assertTrue(should_handover("oru-0", "oru-1", serving, other, tau_e2_s=0.02, tau_ho_s=0.02, horizon_s=1.0, hold_remaining_s=0.0))
        self.assertFalse(should_handover("oru-0", "oru-1", serving, other, tau_e2_s=0.02, tau_ho_s=0.02, horizon_s=1.0, hold_remaining_s=0.2))
        early = {"start_s": 0.01, "end_s": 0.2, "clear": False}
        self.assertFalse(should_handover("oru-0", "oru-1", early, other, tau_e2_s=0.02, tau_ho_s=0.02, horizon_s=1.0, hold_remaining_s=0.0))

    def test_return_after_hold(self) -> None:
        self.assertTrue(should_return({"start_s": 0.4, "end_s": 1.0}, {"clear": True}, 0.0))
        self.assertFalse(should_return({"start_s": 0.4, "end_s": 1.0}, {"clear": True}, 0.3))


class BaselineTest(unittest.TestCase):
    def test_a3_needs_ttt(self) -> None:
        self.assertFalse(a3_trigger(-80.0, -70.0, offset_db=3.0, hysteresis_db=1.0, above_since_s=0.05, ttt_s=0.16))
        self.assertTrue(a3_trigger(-80.0, -70.0, offset_db=3.0, hysteresis_db=1.0, above_since_s=0.2, ttt_s=0.16))

    def test_l3_and_trend(self) -> None:
        self.assertAlmostEqual(l3_filter(-80.0, -90.0, 0.5), -85.0)
        self.assertTrue(trend_trigger([-70.0, -75.0, -80.0, -85.0], 0.1, 5.0, 0.5))


class TimelineTest(unittest.TestCase):
    def test_interpolation_and_loss_lerp(self) -> None:
        values = interpolate_series(np.array([0.0, 10.0]), np.array([0.0, 0.1]), np.array([0.05]))
        self.assertAlmostEqual(float(values[0]), 5.0)
        loss = lerp_loss_db(0.0, 10.0, 0.5)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 10.0)


CACHE_JOB = ROOT / "results" / "cache" / "lamppost" / "high" / "seed_101"
TOL = 1e-5


def _gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class LinkBudgetTest(unittest.TestCase):
    def test_service_snr_and_margin_shift(self) -> None:
        from sim.comm.linkbudget import extra_loss_db, noise_dbm, rate_bps, sensing_overhead, snr_all_margins, snr_req_db

        self.assertAlmostEqual(snr_req_db(4e8, 122.88e6), 9.32, places=2)
        self.assertAlmostEqual(noise_dbm(122.88e6, 13.0), -174.0 + 10 * math.log10(122.88e6) + 13.0, places=12)
        self.assertAlmostEqual(sensing_overhead(1 / 14, 0.32), 0.32 / 14, places=12)
        ref = np.array([[10.0, 40.0]])
        loss = extra_loss_db([0.0, 30.0], 35.9)
        got = snr_all_margins(ref, loss)
        self.assertTrue(np.allclose(got[0], ref - 35.9) and np.allclose(got[1], ref - 5.9))
        self.assertAlmostEqual(float(rate_bps(np.array(0.0), 1e6)), 1e6, places=6)
        if _gpu():
            gpu = snr_all_margins(ref, loss, device="cuda").cpu().numpy()
            self.assertTrue(np.array_equal(gpu, got))
        # rate matches the scalar PHY reference
        for snr_db in (-5.0, 0.0, 9.3, 30.0):
            scalar = shannon_bps(10 ** (snr_db / 10), 122.88e6)
            self.assertAlmostEqual(float(rate_bps(np.array(snr_db), 122.88e6)) / scalar, 1.0, places=9)


@unittest.skipUnless(_gpu() and (CACHE_JOB / "comm_geometry.npz").exists(), "needs GPU and the comm geometry cache")
class TimelineEqualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sim.scenes.config import load_yaml
        from sim.scenes.traffic import prepare_scenario
        from xapp.timeline import actor_tracks, comm_times, held_segments, load_geometry

        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        cls.geom, cls.meta = load_geometry(CACHE_JOB)
        scenario = prepare_scenario(raw, seed=101, mount="lamppost", density="high", duration_s=60.0, dt_s=0.1)
        times = comm_times(int(cls.meta["n_snapshots"]), 0.1, 0.01)[:3001]
        cls.actors = actor_tracks(scenario, times)
        cls.seg = held_segments(cls.geom, times, cls.actors["ue_position_m"], 0.1)

    def test_gpu_matches_numpy(self) -> None:
        from xapp.timeline import model_b_timeline, model_b_timeline_numpy

        wl = float(self.meta["wavelength_m"])
        gpu = model_b_timeline(self.seg, self.actors["blocker_position_m"], self.actors["blocker_size_m"], wl, max_elements=2_000_000)
        steps = np.array([0, 1, 7, 10, 155, 1003, 2047, 2999, 3000])
        ref = model_b_timeline_numpy(self.seg, self.actors["blocker_position_m"], self.actors["blocker_size_m"], wl, steps)
        for key in ("blocked_power", "unblocked_power", "best_alt_power", "los_blocked_power"):
            a, b = gpu[key][steps], ref[key]
            self.assertTrue(np.allclose(a, b, rtol=TOL, atol=0.0), key)
        a, b = gpu["los_loss_db"][steps], ref["los_loss_db"]
        fin = np.isfinite(b)
        self.assertTrue(np.array_equal(np.isfinite(a), fin))
        self.assertTrue(np.allclose(a[fin], b[fin], rtol=0.0, atol=TOL))
        self.assertTrue(np.array_equal(gpu["los_blocker"][steps], ref["los_blocker"]))

    def test_snapshot_steps_reproduce_the_trace(self) -> None:
        from xapp.timeline import model_b_timeline

        wl = float(self.meta["wavelength_m"])
        gpu = model_b_timeline(self.seg, self.actors["blocker_position_m"], self.actors["blocker_size_m"], wl)
        snaps = np.arange(0, 3001, 10)
        ref = self.geom["blocked_power_snapshot"][: len(snaps)]
        # The trace used the float32 device position of the UE, the timeline the
        # float64 analytic one (|diff| ~1.5e-6 m); near a grazing knife edge that
        # moves the power by up to ~2.5e-5 relative (~1e-4 dB). Not an
        # implementation difference: the GPU/NumPy tests above hold at 1e-5.
        self.assertTrue(np.allclose(gpu["blocked_power"][snaps], ref, rtol=1e-4, atol=0.0))

    def test_prediction_gpu_matches_numpy(self) -> None:
        import json

        from xapp.predict_torch import pack_tracks, predict_numpy, predict_torch, ue_fixes

        tracks = json.loads((CACHE_JOB / "tracks_budget_2.json").read_text())
        packed = pack_tracks(tracks, {"bus": (12.0, 2.5, 3.2), "pedestrian": (0.5, 0.5, 1.75)})
        rep = np.arange(0, 3001, 10)
        pos = self.actors["ue_position_m"][rep]
        vel = self.actors["ue_velocity_mps"][rep]
        fix = ue_fixes(pos, vel, 1.0, 7)
        sub = {k: v[: len(rep)] for k, v in packed.items()}
        taus = np.arange(0.0, 3.0 + 1e-9, 0.05)
        oru = self.geom["oru_position_m"][0]
        wl = float(self.meta["wavelength_m"])
        gpu = predict_torch(sub, fix, vel, oru, taus, wl)
        reports = [0, 5, 77, 150, 299]
        ref = predict_numpy(sub, fix, vel, oru, taus, wl, reports)
        a, b = gpu[reports], ref
        fin = np.isfinite(b)
        self.assertTrue(np.array_equal(np.isfinite(a), fin))
        self.assertTrue(np.allclose(a[fin], b[fin], rtol=0.0, atol=TOL))


class SchemeEqualityTest(unittest.TestCase):
    def _lanes(self):
        from xapp.schemes import Lanes

        rng = np.random.default_rng(5)
        n_l, n_t = 9, 1500
        base = np.cumsum(rng.normal(0.0, 0.6, size=(n_l, n_t, 2)), axis=1) + np.array([15.0, 12.0])
        dips = rng.random((n_l, n_t, 2)) < 0.004
        for l, k, c in zip(*np.nonzero(dips)):
            base[l, k : k + 80, c] -= 20.0
        scheme = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        trig = rng.random((n_l, n_t // 10, 2)) < 0.05
        return Lanes(
            snr_db=base,
            scheme=scheme,
            offset_db=np.array([1.0, 3.0, 1.0] * 3),
            hysteresis_db=np.array([1.0, 2.0, 3.0] * 3),
            ttt_steps=np.array([4, 8, 16] * 3),
            filter_a=np.full(n_l, 0.5),
            window_steps=np.array([10, 20, 50] * 3),
            drop_db=np.array([3.0, 5.0, 3.0] * 3),
            trend_horizon_s=np.where(scheme == 2, 0.5, 0.0),
            hold_steps=np.array([20, 50, 100] * 3),
            tau_ho_steps=np.full(n_l, 2),
            e2_delay_steps=np.array([2, 5, 10] * 3),
            overhead=np.where(scheme == 3, 0.32 / 14, 0.0),
            initial_cell=np.array([0, 1, 0] * 3),
            trigger=trig,
            trigger_end_steps=rng.integers(0, 300, size=trig.shape),
        )

    def test_a5_vectorised_matches_scalar(self) -> None:
        from xapp.schemes import REASON, simulate, simulate_scalar

        lanes = self._lanes()
        n = lanes.snr_db.shape[0]
        lanes.scheme = np.array([REASON["a5"]] * 4 + list(lanes.scheme[4:]))
        lanes.a5_thr1 = np.array([5.0, 10.0, 12.0, 8.0] + [0.0] * (n - 4))
        lanes.a5_thr2 = np.array([8.0, 6.0, 12.0, 10.0] + [0.0] * (n - 4))
        kw = {"bandwidth_hz": 122.88e6, "rate_req_bps": 4e8, "max_se": 7.4063}
        vec = simulate(lanes, **kw)
        total = 0
        for lane in range(n):
            ref = simulate_scalar(lanes, lane, **kw)
            sel = vec["handovers"]["lane"] == lane
            got = list(zip(vec["handovers"]["step"][sel], vec["handovers"]["from"][sel], vec["handovers"]["to"][sel], vec["handovers"]["reason"][sel]))
            self.assertEqual([tuple(int(x) for x in g) for g in got], [tuple(int(x) for x in r) for r in ref["handovers"]], f"lane {lane}")
            self.assertTrue(np.array_equal(vec["outage_req"][lane], ref["outage_req"]))
            if lane < 4:
                total += len(ref["handovers"])
        self.assertGreater(total, 3)

    def test_vectorised_matches_scalar(self) -> None:
        from xapp.schemes import simulate, simulate_scalar

        lanes = self._lanes()
        kw = {"bandwidth_hz": 122.88e6, "rate_req_bps": 4e8, "max_se": 7.4063}
        vec = simulate(lanes, **kw)
        total = 0
        for lane in range(lanes.snr_db.shape[0]):
            ref = simulate_scalar(lanes, lane, **kw)
            sel = vec["handovers"]["lane"] == lane
            got = list(zip(vec["handovers"]["step"][sel], vec["handovers"]["from"][sel], vec["handovers"]["to"][sel], vec["handovers"]["reason"][sel]))
            self.assertEqual([tuple(int(x) for x in g) for g in got], [tuple(int(x) for x in r) for r in ref["handovers"]], f"lane {lane}")
            self.assertTrue(np.array_equal(vec["outage_req"][lane], ref["outage_req"]))
            self.assertTrue(np.array_equal(vec["outage0"][lane], ref["outage0"]))
            self.assertAlmostEqual(vec["rate_sum"][lane] / ref["rate_sum"], 1.0, places=9)
            total += len(ref["handovers"])
        self.assertGreater(total, 5)


if __name__ == "__main__":
    unittest.main()
