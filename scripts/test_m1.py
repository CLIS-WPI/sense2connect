"""M1 checks: model B formulas, continuity, and a bit-identical scenario."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.blockage import (  # noqa: E402
    knife_edge_f,
    lateral_offset_sweep,
    loss_db_from_factors,
    screen_blockage,
    sum_blockage_db,
)
from sim.scenes.config import load_scenario  # noqa: E402
from sim.scenes.loop import run_scenario  # noqa: E402
from sim.scenes.motion import states_at  # noqa: E402


def direct_knife_edge_f(excess_m: float, wavelength_m: float, sign: float) -> float:
    """Clause 7.6.4.2 edge factor, written out independently of the module."""
    excess = excess_m if excess_m > 0.0 else 0.0
    return math.atan(sign * (math.pi / 2.0) * math.sqrt(math.pi / wavelength_m * excess)) / math.pi


def direct_loss_db(f_h1: float, f_h2: float, f_w1: float, f_w2: float) -> float:
    """Clause 7.6.4.2 screen loss [dB], written out independently."""
    base = 1.0 - (f_h1 + f_h2) * (f_w1 + f_w2)
    if base <= 1e-15:
        return math.inf
    return -20.0 * math.log10(base)


class ModelBFormulaTest(unittest.TestCase):
    def test_edge_factor_matches_direct_formula(self) -> None:
        wavelength = 299_792_458.0 / 28e9
        for excess in (0.0, 1e-4, 0.01, 0.2, 1.0, 5.0):
            for sign in (-1.0, 1.0):
                got = knife_edge_f(excess, wavelength, sign)
                exp = direct_knife_edge_f(excess, wavelength, sign)
                self.assertAlmostEqual(got, exp, places=12)

    def test_screen_loss_matches_direct_formula(self) -> None:
        wavelength = 299_792_458.0 / 28e9
        samples = (
            (0.01, 0.02, 0.03, 0.04),
            (0.2, 0.2, 0.1, -0.05),
            (0.4, 0.4, 0.4, 0.4),
        )
        for factors in samples:
            got = loss_db_from_factors(*factors)
            exp = direct_loss_db(*factors)
            self.assertTrue(math.isclose(got, exp, rel_tol=1e-12, abs_tol=1e-12))

    def test_two_screens_add_in_db(self) -> None:
        self.assertAlmostEqual(sum_blockage_db([1.5, 2.25]), 3.75)

    def test_centered_screen_exceeds_a_far_screen(self) -> None:
        wavelength = 299_792_458.0 / 28e9
        tx = np.array([0.0, 0.0, 1.5])
        rx = np.array([20.0, 0.0, 1.5])
        center = screen_blockage(tx, rx, np.array([10.0, 0.0, 0.8]), 5.0, 2.0, 1.6, wavelength)
        far = screen_blockage(tx, rx, np.array([10.0, 8.0, 0.8]), 5.0, 2.0, 1.6, wavelength)
        self.assertGreater(center.loss_db, 3.0)
        self.assertLess(abs(far.loss_db), 1.0)
        self.assertTrue(center.intersects)
        self.assertFalse(far.intersects)

    def test_loss_is_continuous_across_the_screen_edge(self) -> None:
        wavelength = 299_792_458.0 / 28e9
        tx = np.array([0.0, 0.0, 1.5])
        rx = np.array([20.0, 0.0, 1.5])
        # Projected half-width of the 2 m wide cuboid, viewed along x, is 1 m.
        edge = 1.0
        inside = screen_blockage(
            tx, rx, np.array([10.0, edge - 1e-4, 0.8]), 5.0, 2.0, 1.6, wavelength
        ).loss_db
        outside = screen_blockage(
            tx, rx, np.array([10.0, edge + 1e-4, 0.8]), 5.0, 2.0, 1.6, wavelength
        ).loss_db
        self.assertTrue(math.isfinite(inside) and math.isfinite(outside))
        self.assertLess(abs(inside - outside), 0.05)

        offsets = np.linspace(-8.0, 8.0, 801)
        losses = lateral_offset_sweep(offsets, wavelength)
        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertGreater(losses[len(losses) // 2], losses[0])


class ScenarioTest(unittest.TestCase):
    def test_same_config_and_seed_match_and_report_events(self) -> None:
        scenario = load_scenario(ROOT / "configs" / "m1_scenario.yaml")
        official = ROOT / "results" / "M1"
        replica = ROOT / "results" / "M1_replica"
        first = run_scenario(scenario, official, render=True)
        second = run_scenario(scenario, replica, render=False)

        with np.load(official / "channels.npz", allow_pickle=True) as left_file:
            left = list(left_file["snapshots"])
        with np.load(replica / "channels.npz", allow_pickle=True) as right_file:
            right = list(right_file["snapshots"])
        # Interaction types are discrete. Coefficients, delays, Doppler and
        # vertices come from OptiX and match within the tolerances below;
        # the residuals are a few micrometres or about 1e-9 on field values.
        # OptiX reductions on this GPU repeat to about 1e-3 relative on the
        # weaker coefficients and to 1e-4 m on vertices. Interaction types match.
        float_tol = {
            "a": (1e-2, 1e-7),
            "tau": (1e-5, 1e-10),
            "doppler": (1e-4, 1e-3),
            "vertices": (0.0, 1e-4),
        }
        self.assertEqual(len(left), len(right))
        for left_snap, right_snap in zip(left, right):
            self.assertEqual(left_snap["t_s"], right_snap["t_s"])
            for key in ("sensing", "background", "comm_geometric"):
                for field in ("interactions", "valid"):
                    np.testing.assert_array_equal(
                        left_snap[key][field], right_snap[key][field]
                    )
                for field, (rtol, atol) in float_tol.items():
                    np.testing.assert_allclose(
                        left_snap[key][field],
                        right_snap[key][field],
                        rtol=rtol,
                        atol=atol,
                    )
            np.testing.assert_allclose(
                left_snap["comm_model_b"], right_snap["comm_model_b"], rtol=1e-2, atol=1e-7
            )

        self.assertEqual(first["events"], second["events"])
        self.assertTrue((official / "scenario.gif").is_file())
        self.assertTrue((official / "blockage_lateral_offset.png").is_file())
        self.assertGreaterEqual(len(list((official / "frames").glob("frame_*.png"))), 1)

        los_events = [event for event in first["events"] if event["path"] == "los"]
        self.assertTrue(los_events, "expected a LoS blockage event")
        for event in los_events:
            self.assertIn("blocker_id", event)
            self.assertLessEqual(event["start_s"], event["end_s"])
            self.assertIsInstance(event["blocker_id"], str)
            self.assertTrue(event["blocker_id"])

        # Scene read-back matches the analytic trajectory, including velocity.
        for snap in first["states"]:
            expected = states_at(scenario, snap["t_s"])
            for name, state in expected.items():
                got_pos = np.asarray(snap["actors"][name]["position_m"])
                got_vel = np.asarray(snap["actors"][name]["velocity_mps"])
                np.testing.assert_allclose(got_pos, state["position_m"], atol=1e-4)
                np.testing.assert_allclose(got_vel, state["velocity_mps"], atol=1e-6)

        # The comm LoS path is still present when a blocker crosses it.
        # Model B reduces it; the absorber is not in that scene.
        blocked_times = {event["start_s"] for event in los_events}
        saw_reduction = False
        with np.load(official / "channels.npz", allow_pickle=True) as handle:
            snaps = list(handle["snapshots"])
        for snap in snaps:
            geometric = snap["comm_geometric"]
            interactions = geometric["interactions"]
            valid = geometric["valid"]
            for path_index in range(valid.shape[-1]):
                if not bool(valid[0, 0, path_index]):
                    continue
                if not np.all(interactions[:, 0, 0, path_index] == 0):
                    continue
                geo = geometric["a"][0, :, 0, :, path_index, :]
                modelled = snap["comm_model_b"][0, :, 0, :, path_index, :]
                self.assertGreater(np.max(np.abs(geo)), 0.0)
                if snap["t_s"] in blocked_times or any(
                    event["start_s"] <= snap["t_s"] <= event["end_s"] for event in los_events
                ):
                    if np.max(np.abs(modelled)) < 0.9 * np.max(np.abs(geo)):
                        saw_reduction = True
        self.assertTrue(saw_reduction)

        written = json.loads((official / "ground_truth.json").read_text(encoding="utf-8"))
        self.assertEqual(written["events"], first["events"])


if __name__ == "__main__":
    unittest.main()
