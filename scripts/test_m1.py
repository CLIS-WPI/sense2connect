"""Theory checks for blockage model B, and a repeatable short scenario."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.blockage import (  # noqa: E402
    available_power,
    headroom_db,
    path_blocker_loss,
    power_ratio_loss_db,
    screen_blockage,
)
from sim.comm.events import annotate_los_events, apply_hysteresis, entry_warning_s, merge_ue_events  # noqa: E402
from sim.scenes.config import load_scenario  # noqa: E402
from sim.scenes.loop import run_scenario  # noqa: E402
from sim.scenes.motion import states_at, track_identity  # noqa: E402
from sim.scenes.traffic import prepare_scenario  # noqa: E402

WAVELENGTH_M = 299_792_458.0 / 28e9


def _wide_screen(center_y_m: float):
    """Tall, wide screen beside a horizontal 80 m link at z = 5 m."""
    tx = np.array([0.0, 0.0, 5.0])
    rx = np.array([80.0, 0.0, 5.0])
    center = np.array([40.0, center_y_m, 5.0])
    # 4 m along the link, 40 m across it, 40 m tall.
    return screen_blockage(tx, rx, center, 4.0, 40.0, 40.0, WAVELENGTH_M)


class ModelBTheoryTest(unittest.TestCase):
    def test_grazing_edge_of_a_wide_screen_is_about_6_db(self) -> None:
        # Half-width is 20 m, so a center at y = 20 m puts one vertical edge on the link.
        # A single knife edge at the shadow boundary is 6 dB. The other three edges of
        # this screen are far, so model B should land on that value.
        loss = _wide_screen(20.0).loss_db
        self.assertAlmostEqual(loss, 6.0, delta=0.5)

    def test_screen_far_from_the_path_is_about_0_db(self) -> None:
        loss = _wide_screen(80.0).loss_db
        self.assertLess(abs(loss), 0.5)
        self.assertFalse(_wide_screen(80.0).intersects)

    def test_large_screen_centered_on_the_path_is_a_deep_loss(self) -> None:
        result = _wide_screen(0.0)
        self.assertTrue(result.intersects)
        self.assertGreater(result.loss_db, 20.0)

    def test_loss_increases_as_the_screen_moves_into_the_path(self) -> None:
        offsets = np.linspace(30.0, 0.0, 31)
        losses = np.array([_wide_screen(float(offset)).loss_db for offset in offsets])
        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertTrue(np.all(np.diff(losses) >= -0.05))
        self.assertGreater(losses[-1], losses[0] + 10.0)

    def test_every_segment_contributes(self) -> None:
        wavelength = WAVELENGTH_M
        # Two colinear hops. A screen sits on the first hop only.
        first = (np.array([0.0, 0.0, 5.0]), np.array([40.0, 0.0, 5.0]))
        second = (np.array([40.0, 0.0, 5.0]), np.array([80.0, 0.0, 5.0]))
        center = np.array([20.0, 0.0, 5.0])
        on_first, _ = path_blocker_loss([first], center, 4.0, 40.0, 40.0, wavelength)
        on_both, parts = path_blocker_loss([first, second], center, 4.0, 40.0, 40.0, wavelength)
        self.assertGreater(on_first, 20.0)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(on_both, on_first, delta=1.0)

    def test_total_power_loss_matches_the_ratio(self) -> None:
        # Two equal paths. One is unattenuated, one loses 10 dB, so the sum drops by about 2.6 dB.
        loss = power_ratio_loss_db([1.0, 1.0], [0.0, 10.0])
        expected = 10.0 * np.log10(2.0 / (1.0 + 0.1))
        self.assertAlmostEqual(loss, expected, delta=1e-9)
        self.assertEqual(power_ratio_loss_db([1.0], [float("inf")]), float("inf"))
        self.assertAlmostEqual(headroom_db(1.0, 0.1), -10.0, delta=1e-9)

    def test_short_gaps_merge_and_ten_db_does_not_exceed_three(self) -> None:
        # One 3 dB outage with a 0.2 s dip below 10 dB. That dip used to make
        # two 10 dB events and one 3 dB event.
        def event(metric: str, threshold: float, start: int, end: int, peak: float) -> dict:
            return {
                "ue": "ue-0",
                "oru": "oru-0",
                "metric": metric,
                "threshold_db": threshold,
                "blocker_id": "bus-0",
                "blocker_kind": "bus",
                "start_snapshot": start,
                "end_snapshot": end,
                "start_s": start * 0.1,
                "end_s": end * 0.1,
                "max_loss_db": peak,
            }

        raw = [
            event("los", 3.0, 0, 10, 22.0),
            event("los", 10.0, 0, 4, 22.0),
            event("los", 10.0, 7, 10, 18.0),
        ]
        self.assertEqual(len(merge_ue_events(raw, 0.0, 0.1)), 3)
        # The dip below 10 dB is 0.2 s, so a 0.5 s gap joins the two 10 dB pieces.
        merged = apply_hysteresis(raw, 0.5, 0.1)
        los3 = [item for item in merged if item["threshold_db"] == 3.0]
        los10 = [item for item in merged if item["threshold_db"] == 10.0]
        self.assertEqual(len(los3), 1)
        self.assertEqual(len(los10), 1)
        self.assertLessEqual(len(los10), len(los3))
        self.assertEqual(los10[0]["start_snapshot"], 0)
        self.assertEqual(los10[0]["end_snapshot"], 10)
        self.assertEqual(los10[0]["blocker_id"], "bus-0")
        # A 2 s dip that stays inside the 3 dB event is still one 10 dB event.
        long_dip = [
            event("los", 3.0, 0, 30, 22.0),
            event("los", 10.0, 0, 4, 22.0),
            event("los", 10.0, 25, 30, 18.0),
        ]
        nested = apply_hysteresis(long_dip, 0.5, 0.1)
        self.assertEqual(len([item for item in nested if item["threshold_db"] == 10.0]), 1)
        self.assertEqual(len([item for item in nested if item["threshold_db"] == 3.0]), 1)

    def test_warning_is_the_time_since_entering_range(self) -> None:
        distances = [50.0, 45.0, 30.0, 20.0, 10.0]
        inside = entry_warning_s(distances, 4, 0.1, 40.0)
        self.assertAlmostEqual(inside["warning_s"], 0.2, delta=1e-9)
        self.assertFalse(inside["censored"])
        outside = entry_warning_s(distances, 1, 0.1, 40.0)
        self.assertIsNone(outside["warning_s"])
        self.assertFalse(outside["in_range"])
        already = entry_warning_s([10.0, 12.0, 11.0], 2, 0.1, 40.0)
        self.assertTrue(already["censored"])
        self.assertAlmostEqual(already["warning_s"], 0.2, delta=1e-9)
        born = entry_warning_s([float("inf"), float("inf"), 20.0, 15.0], 3, 0.1, 40.0)
        self.assertTrue(born["censored"])
        self.assertTrue(born["upper_bound"])
        self.assertAlmostEqual(born["warning_s"], 0.1, delta=1e-9)

    def test_available_headroom_uses_power_after_model_b(self) -> None:
        self.assertAlmostEqual(available_power(0.1, 10.0), 0.01, delta=1e-12)
        events = [
            {
                "metric": "los",
                "threshold_db": 10.0,
                "ue": "ue-0",
                "oru": "oru-0",
                "start_snapshot": 0,
                "end_snapshot": 0,
                "blocker_id": "bus-0#0",
            }
        ]
        trace = [
            {
                "ue": "ue-0",
                "oru": "oru-0",
                "snapshot": 0,
                "los_loss_db": 12.0,
                "los_power": 1.0,
                "alt_power": 0.1,
                "alt_available_power": 0.01,
                "alt_blocked_by_los_blocker": "bus-0#0",
            }
        ]
        annotate_los_events(events, trace, {}, 0.1, 40.0)
        self.assertAlmostEqual(events[0]["headroom_db"], -10.0, delta=1e-9)
        self.assertAlmostEqual(events[0]["headroom_available_db"], -20.0, delta=1e-9)
        self.assertTrue(events[0]["alt_same_blocker"])

    def test_a_wrap_starts_a_new_identity(self) -> None:
        scenario = {
            "lanes": [{"name": "east", "length_m": 80.0}],
            "sidewalks": [],
        }
        east = {"name": "car-0", "lane": "east", "s0_m": 70.0, "speed_mps": 10.0}
        self.assertEqual(track_identity(scenario, east, 0.9), "car-0#0")
        self.assertEqual(track_identity(scenario, east, 1.0), "car-0#1")
        west = {"name": "car-1", "lane": "east", "s0_m": 5.0, "speed_mps": -10.0}
        self.assertEqual(track_identity(scenario, west, 0.4), "car-1#0")
        self.assertEqual(track_identity(scenario, west, 0.6), "car-1#-1")


class DeploymentTest(unittest.TestCase):
    def test_mounts_and_fleet_follow_the_config(self) -> None:
        raw = load_scenario(ROOT / "configs" / "m1_scenario.yaml")
        lamp = prepare_scenario(raw, mount="lamppost", density="low", seed=101)
        facade = prepare_scenario(raw, mount="facade", density="high", seed=101)
        self.assertEqual(lamp["oru_height_m"], 5.0)
        self.assertEqual(facade["oru_height_m"], 8.0)
        self.assertEqual(lamp["orus"][0]["position_m"][2], 5.0)
        self.assertEqual(facade["orus"][0]["position_m"][2], 8.0)
        self.assertEqual(len(facade["orus"]), 1)
        m2 = prepare_scenario(load_scenario(ROOT / "configs" / "m2_scenario.yaml"), seed=101)
        self.assertEqual(len(m2["orus"]), 2)
        second = m2["orus"][1]
        self.assertEqual(second["height_m"], 5.0)
        self.assertEqual(second["mount"], "lamppost")
        self.assertLess(second["position_m"][1], m2["orus"][0]["position_m"][1])
        self.assertAlmostEqual(abs(second["position_m"][0] - m2["orus"][0]["position_m"][0]), 40.0, delta=1e-6)
        self.assertEqual(m2["array"]["oru"]["num_rows"], 8)
        self.assertEqual(m2["array"]["ue"]["num_cols"], 1)
        self.assertTrue(m2["array"]["synthetic_array"])
        buses = [item for item in facade["vehicles"] if item["kind"] == "bus"]
        trucks = [item for item in facade["vehicles"] if item["kind"] == "truck"]
        self.assertTrue(buses)
        self.assertEqual(buses[0]["length_m"], 12.0)
        self.assertEqual(buses[0]["width_m"], 2.5)
        self.assertEqual(buses[0]["height_m"], 3.2)
        self.assertEqual(buses[0]["object_type"], "vehicle-multi-sp")
        self.assertEqual(trucks[0]["length_m"], 8.0)
        self.assertEqual(trucks[0]["height_m"], 3.5)
        sidewalk = [item for item in facade["pedestrians"] if item["motion"] == "sidewalk"]
        crossing = [item for item in facade["pedestrians"] if item["motion"] == "crossing"]
        self.assertEqual(len(sidewalk), 12)
        self.assertEqual(len(crossing), 4)
        self.assertEqual({item["sidewalk"] for item in sidewalk}, {"south", "north"})
        self.assertTrue(any(item["speed_mps"] > 0.0 for item in sidewalk))
        self.assertTrue(any(item["speed_mps"] < 0.0 for item in sidewalk))
        self.assertTrue(any(item["direction"] > 0.0 for item in crossing))
        self.assertTrue(any(item["direction"] < 0.0 for item in crossing))
        again = prepare_scenario(raw, mount="facade", density="high", seed=101)
        self.assertEqual(
            [item["s0_m"] for item in facade["vehicles"]],
            [item["s0_m"] for item in again["vehicles"]],
        )
        self.assertEqual(
            [item["s0_m"] for item in facade["pedestrians"]],
            [item["s0_m"] for item in again["pedestrians"]],
        )

    def test_a_crossing_passes_between_the_sidewalks(self) -> None:
        # The station is an arbitrary point on the street, not a radio coordinate.
        scenario = {
            "lanes": [],
            "sidewalks": [
                {
                    "name": "south",
                    "origin_m": [-40.0, -7.0, 0.0],
                    "direction": [1.0, 0.0, 0.0],
                    "length_m": 80.0,
                },
                {
                    "name": "north",
                    "origin_m": [-40.0, 6.5, 0.0],
                    "direction": [1.0, 0.0, 0.0],
                    "length_m": 80.0,
                },
            ],
            "ues": [],
            "vehicles": [],
            "pedestrians": [
                {
                    "name": "ped-cross",
                    "motion": "crossing",
                    "sidewalk": "south",
                    "other_sidewalk": "north",
                    "s0_m": 10.0,
                    "s_cross_m": 10.0,
                    "speed_mps": 1.5,
                    "direction": 1.0,
                    "center_height_m": 0.875,
                }
            ],
        }
        ys = [float(states_at(scenario, t)["ped-cross"]["position_m"][1]) for t in np.linspace(0.0, 12.0, 25)]
        self.assertLess(min(ys), -6.0)
        self.assertGreater(max(ys), 6.0)
        self.assertTrue(any(-6.0 < y < 6.0 for y in ys))


class ScenarioTest(unittest.TestCase):
    def test_repeatable_paths_and_per_path_blockage(self) -> None:
        raw = load_scenario(ROOT / "configs" / "m1_scenario.yaml")
        scenario = prepare_scenario(raw)
        official = ROOT / "results" / "M1"
        replica = ROOT / "results" / "M1_replica"
        first = run_scenario(scenario, official, render=True)
        second = run_scenario(scenario, replica, render=False)

        # PathSolver and RCSSolver use deterministic=True. Residual differences
        # are OptiX/Dr.Jit reduction order on weak coefficients, not the trajectory.
        float_tol = {
            "a": (1e-2, 1e-7),
            "tau": (1e-5, 1e-10),
            "doppler": (1e-4, 1e-3),
            "vertices": (0.0, 1e-4),
        }
        with np.load(official / "channels.npz", allow_pickle=True) as left_file:
            left = list(left_file["snapshots"])
        with np.load(replica / "channels.npz", allow_pickle=True) as right_file:
            right = list(right_file["snapshots"])
        self.assertEqual(len(left), len(right))
        max_rel = 0.0
        for left_snap, right_snap in zip(left, right):
            for key in ("sensing", "background", "comm_geometric"):
                for field in ("interactions", "valid"):
                    np.testing.assert_array_equal(left_snap[key][field], right_snap[key][field])
                for field, (rtol, atol) in float_tol.items():
                    np.testing.assert_allclose(
                        left_snap[key][field], right_snap[key][field], rtol=rtol, atol=atol
                    )
                    diff = np.abs(left_snap[key][field] - right_snap[key][field])
                    scale = np.maximum(np.abs(left_snap[key][field]), np.abs(right_snap[key][field]))
                    rel = np.divide(
                        np.abs(diff).astype(float),
                        np.abs(scale).astype(float),
                        out=np.zeros(np.shape(diff), dtype=float),
                        where=np.abs(scale).astype(float) > 1e-12,
                    )
                    if rel.size:
                        max_rel = max(max_rel, float(np.max(rel)))
            np.testing.assert_allclose(
                left_snap["comm_model_b"], right_snap["comm_model_b"], rtol=1e-2, atol=1e-7
            )
        (official / "repeatability.txt").write_text(
            f"max_relative_coefficient_or_geometry={max_rel:.6e}\n",
            encoding="utf-8",
        )

        def _by_shape(events: list[dict]) -> list[dict]:
            comparable = []
            for event in events:
                item = dict(event)
                item["path"] = item["path_key"]
                comparable.append(item)
            # Raw Paths.objects integers change across scene loads, so sort on the stable key.
            comparable.sort(key=lambda item: (item["ue"], item["path_key"], item["blocker_id"], item["start_s"]))
            return comparable

        self.assertEqual(_by_shape(first["events"]), _by_shape(second["events"]))
        self.assertEqual(first["ue_events"], second["ue_events"])
        for event in first["events"]:
            if event["path"] == "los":
                self.assertEqual(event["path_class"], "los")
            else:
                self.assertRegex(event["path"], r"^nlos-[0-9.]+-[0-9.]+$")
                self.assertIn(event["path_class"], ("ground", "wall", "double"))
        stability = first["path_id_stability"]["mean_persistence"]
        self.assertIsNotNone(stability)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        self.assertEqual(first["oru_height_m"], 8.0)
        los_rows = [row for row in first["per_snapshot_blockage"] if row["path"] == "los"]
        self.assertTrue(los_rows)
        self.assertTrue(all(len(row["segments"]) >= 1 for row in los_rows))
        reflected = [row for row in first["per_snapshot_blockage"] if row["path"] != "los"]
        self.assertTrue(reflected, "expected reflected comm paths")
        self.assertTrue(any(len(row["segments"]) >= 2 for row in reflected))
        for event in first["events"]:
            self.assertLessEqual(event["start_s"], event["end_s"])
            self.assertIn(event["blocker_kind"], ("car", "bus", "truck", "pedestrian"))

        for snap in first["states"]:
            expected = states_at(scenario, snap["t_s"])
            for name, state in expected.items():
                got_pos = np.asarray(snap["actors"][name]["position_m"])
                got_vel = np.asarray(snap["actors"][name]["velocity_mps"])
                np.testing.assert_allclose(got_pos, state["position_m"], atol=1e-4)
                np.testing.assert_allclose(got_vel, state["velocity_mps"], atol=1e-6)

        written = json.loads((official / "ground_truth.json").read_text(encoding="utf-8"))
        self.assertEqual(written["events"], first["events"])
        self.assertTrue((official / "scenario.gif").is_file())


if __name__ == "__main__":
    unittest.main()
