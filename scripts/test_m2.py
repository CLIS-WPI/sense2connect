"""Waveform, CFAR, clustering, and one monostatic CPI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml  # noqa: E402
from sim.scenes.traffic import prepare_scenario  # noqa: E402
from sim.sensing.cfar import ca_cfar  # noqa: E402
from sim.sensing.channel import rotation_matrix  # noqa: E402
from sim.sensing.cluster import dbscan  # noqa: E402
from sim.sensing.ghost import mirror_y, reject_ghosts  # noqa: E402
from sim.sensing.noise import noise_power_w, subcarrier_noise_power_w  # noqa: E402
from sim.sensing.radar import closest_bin, scan_scenario  # noqa: E402
from sim.sensing.waveform import waveform_from_config  # noqa: E402


class WaveformTest(unittest.TestCase):
    def test_numerology_3_matches_the_radar_budget(self) -> None:
        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        wave = waveform_from_config(raw)
        self.assertEqual(wave.numerology, 3)
        self.assertAlmostEqual(wave.subcarrier_spacing_hz, 120e3, delta=1e-6)
        self.assertAlmostEqual(wave.prf_hz, 8e3, delta=1e-6)
        self.assertAlmostEqual(wave.cpi_s, 0.032, delta=1e-12)
        self.assertAlmostEqual(wave.bandwidth_hz, 122.88e6, delta=1.0)
        self.assertAlmostEqual(wave.v_max_mps, 21.4, delta=0.2)
        self.assertAlmostEqual(wave.v_res_mps, 0.167, delta=0.01)
        self.assertAlmostEqual(wave.overhead, 1.0 / 14.0, delta=1e-12)
        wide = waveform_from_config(raw, n_subcarriers=2048)
        self.assertAlmostEqual(wide.bandwidth_hz, 245.76e6, delta=1.0)

    def test_noise_uses_bandwidth_and_noise_figure(self) -> None:
        per_bin = subcarrier_noise_power_w(120e3, 7.0, 290.0)
        full = noise_power_w(1024 * 120e3, 7.0, 290.0)
        self.assertAlmostEqual(full, 1024 * per_bin, delta=per_bin * 1e-9)

    def test_cfar_refuses_a_noiseless_map(self) -> None:
        power = np.zeros((32, 32))
        power[16, 16] = 10.0
        with self.assertRaises(RuntimeError):
            ca_cfar(power, guard=2, train=4, pfa=1e-3, noise_applied=False)

    def test_dbscan_groups_nearby_scatterers(self) -> None:
        points = np.array([[0.0, 0.0], [1.5, 0.2], [30.0, 0.0]])
        labels = dbscan(points, eps_m=4.0, min_samples=1)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_a_mirror_across_the_facade_is_rejected(self) -> None:
        image = mirror_y(np.array([0.0, 0.0]), 9.6)
        self.assertAlmostEqual(image[1], 19.2, delta=1e-9)
        strong = {"x_m": 0.0, "y_m": 0.0, "power": 10.0}
        ghost = {"x_m": float(image[0]), "y_m": float(image[1]), "power": 1.0}
        kept = reject_ghosts([strong, ghost], [9.6, -8.6], gate_m=2.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["power"], 10.0)

    def test_rotation_matches_sionna(self) -> None:
        import mitsuba as mi
        from sionna.rt.utils.geometry import rotation_matrix as sionna_rotation

        angles = (0.3, -0.2, 0.1)
        reference = np.array(sionna_rotation(mi.Point3f(*angles))).reshape(3, 3)
        got = rotation_matrix(*angles)
        self.assertTrue(np.allclose(got, reference, atol=1e-5))


class RadarFrameTest(unittest.TestCase):
    def test_one_cpi_places_a_vehicle_near_its_range_and_doppler(self) -> None:
        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        wave = waveform_from_config(raw)
        scenario = prepare_scenario(raw, seed=101, mount="lamppost", density="low", duration_s=wave.cpi_s, dt_s=wave.cpi_s)
        frames = scan_scenario(
            scenario,
            wave,
            [{"guard": 2, "train": 4, "pfa": 1e-3}],
            max_frames=1,
        )
        targets = frames[0]["ground_truth"]
        self.assertTrue(targets)
        radar = np.asarray(scenario["orus"][0]["position_m"], dtype=np.float64)
        target = max(targets, key=lambda item: abs(closest_bin(item, radar, wave)[1]))
        distance, approaching = closest_bin(target, radar, wave)
        hits = frames[0]["detections"][(wave.n_subcarriers, "twin", 4, 1e-3)]
        self.assertTrue(hits)
        nearest = min(hits, key=lambda item: abs(float(item["range_m"]) - distance))
        self.assertLess(abs(float(nearest["range_m"]) - distance), 3.0 * wave.range_resolution_m)
        # Scatterers sit off the target center, so the radial speed is not the center's.
        self.assertLess(abs(float(nearest["radial_velocity_mps"]) - approaching), 1.2)
        error = np.hypot(float(nearest["x_m"]) - float(target["x_m"]), float(nearest["y_m"]) - float(target["y_m"]))
        self.assertLess(error, 15.0)


if __name__ == "__main__":
    unittest.main()
