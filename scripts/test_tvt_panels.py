"""Tests of the back-to-back panel model (sim/tvt/panels.py; configs/tvt.yaml panels:).

PatternTest / SelectionTest / RadarTransformTest are pure NumPy. SionnaExactTest (GPU, ~1-2 min) traces one
radar snapshot of the paper-1 canyon twice - with the unchanged isotropic paper-1 radar (look_at the street
centre) and with a TR 38.901 panel at yaw 0 / 180 deg - and checks that the pattern applied to the traced path
angles of the isotropic trace (panels.radar_panel_paths) reproduces the panel trace: same paths, same per-element
coefficients. It also checks a communication link (O-RU 8x8 TX -> UE) the same way.
Run: python scripts/test_tvt_panels.py [SionnaExactTest]
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.tvt import panels as PN  # noqa: E402


def db(x):
    return 20.0 * np.log10(x)


def u_az_el(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


class PatternTest(unittest.TestCase):
    def test_tr38901_values(self):
        """Boresight 8 dBi, -3 dB at +-32.5 deg (65 deg HPBW) in both cuts, 30 dB front-to-back, floor 8 - 30 dBi."""
        self.assertAlmostEqual(db(PN.amplitude(u_az_el(0, 0), 1)), 8.0, places=9)
        for az, el in ((32.5, 0), (-32.5, 0), (0, 32.5), (0, -32.5)):
            self.assertAlmostEqual(db(PN.amplitude(u_az_el(az, el), 1)), 5.0, places=9)
        self.assertAlmostEqual(db(PN.amplitude(u_az_el(180, 0), 1)), -22.0, places=9)
        self.assertAlmostEqual(db(PN.amplitude(u_az_el(90, 0), 1)), 8.0 - 12 * (90 / 65) ** 2, places=9)
        self.assertAlmostEqual(db(PN.amplitude(u_az_el(0, 90), 1)), 8.0 - 12 * (90 / 65) ** 2, places=9)  # SLA_V not reached
        self.assertAlmostEqual(db(PN.amplitude(u_az_el(30, 20), 1)), 8.0 - 12 * (30 / 65) ** 2 - 12 * (20 / 65) ** 2, places=9)

    def test_back_panel_is_mirrored(self):
        """The -x panel sees direction (az, el) as the +x panel sees (az + 180, el)."""
        rng = np.random.default_rng(1)
        az, el = rng.uniform(-180, 180, 200), rng.uniform(-80, 80, 200)
        u = np.stack([u_az_el(a, e) for a, e in zip(az, el)])
        u_m = np.stack([u_az_el(a + 180, e) for a, e in zip(az, el)])
        np.testing.assert_allclose(PN.amplitude(u, -1), PN.amplitude(u_m, 1), rtol=1e-12)
        self.assertAlmostEqual(db(PN.amplitude(u_az_el(180, 0), -1)), 8.0, places=9)
        self.assertAlmostEqual(db(PN.amplitude(u_az_el(0, 0), -1)), -22.0, places=9)

    def test_front_to_back_everywhere(self):
        """For every direction one panel is >= the other; on the x = 0 plane both are equal."""
        u = np.stack([u_az_el(90, e) for e in (-40, 0, 40)] + [u_az_el(-90, 10)])
        np.testing.assert_allclose(PN.amplitude(u, 1), PN.amplitude(u, -1), rtol=1e-12)


class SelectionTest(unittest.TestCase):
    def test_facing(self):
        np.testing.assert_array_equal(PN.facing(np.array([13.0, 11.0, 12.0, -50.0]), 12.0), [1, -1, 1, -1])

    def test_link_amplitude_uses_the_facing_panel(self):
        oru = np.array([[12.0, 6.5, 5.0], [52.0, -7.0, 5.0]])
        ue = np.array([[[20.0, -6.0, 1.5], [30.0, 7.5, 1.5]]])  # T=1, U=2: east of O-RU 0, west of O-RU 1 / east of 0
        d = ue[:, :, None, :] - oru[None, None]  # LoS departure directions [1, 2, 2, 3]
        u = (d / np.linalg.norm(d, axis=-1, keepdims=True))[:, :, :, None, :]  # P = 1
        amp, sx = PN.link_amplitude(u, ue, oru)
        np.testing.assert_array_equal(sx, [[[1, -1], [1, -1]]])
        for i in range(2):
            for c in range(2):
                self.assertAlmostEqual(amp[0, i, c, 0], PN.amplitude(u[0, i, c, 0], sx[0, i, c]))
                self.assertGreaterEqual(amp[0, i, c, 0], PN.amplitude(u[0, i, c, 0], -sx[0, i, c]))

    def test_keep_facing(self):
        det = [{"x_m": 20.0}, {"x_m": 5.0}, {"x_m": 12.0}]
        self.assertEqual([d["x_m"] for d in PN.keep_facing(det, 1, 12.0)], [20.0, 12.0])
        self.assertEqual([d["x_m"] for d in PN.keep_facing(det, -1, 12.0)], [5.0])

    def test_config(self):
        self.assertTrue(PN.enabled())
        c = PN.config()
        self.assertEqual(c["pattern"], "tr38901")
        self.assertEqual(c["yaw_deg"], [0.0, 180.0])


class RadarTransformTest(unittest.TestCase):
    def test_resteer_matches_direct_synthesis(self):
        """Per-element coefficients of an array at an arbitrary orientation -> panel sx: equal to direct synthesis."""
        from sim.positioning.array import element_positions
        from sim.sensing.channel import rotation_matrix

        wl = 299_792_458.0 / 28e9
        r = element_positions(wl)
        rot_old = rotation_matrix(-math.pi / 2, 0.45, 0.0)
        rng = np.random.default_rng(3)
        P = 7
        th_r, ph_r = rng.uniform(0.3, 2.8, P), rng.uniform(-math.pi, math.pi, P)
        th_t, ph_t = rng.uniform(0.3, 2.8, P), rng.uniform(-math.pi, math.pi, P)
        c = rng.standard_normal(P) + 1j * rng.standard_normal(P)
        u_r = np.stack([np.sin(th_r) * np.cos(ph_r), np.sin(th_r) * np.sin(ph_r), np.cos(th_r)], -1)
        u_t = np.stack([np.sin(th_t) * np.cos(ph_t), np.sin(th_t) * np.sin(ph_t), np.cos(th_t)], -1)
        k = 2 * math.pi / wl
        a_old = c[None] * np.exp(1j * k * (r @ rot_old.T) @ u_r.T)  # [M, P]
        packed = {"a": a_old[None, :, None, None, :, None].astype(np.complex64), "theta_r": th_r[None, None].astype(np.float32),
                  "phi_r": ph_r[None, None].astype(np.float32), "theta_t": th_t[None, None].astype(np.float32),
                  "phi_t": ph_t[None, None].astype(np.float32)}
        for sx in (1, -1):
            out, resid = PN.radar_panel_paths(packed, sx, r, rot_old, wl)
            self.assertLess(resid, 2e-3)  # float32 angles and coefficients
            want = c[None] * PN.amplitude(u_t, sx)[None] * PN.amplitude(u_r, sx)[None] * np.exp(1j * k * (r @ PN.rotation(sx).T) @ u_r.T)
            got = out["a"][0, :, 0, 0, :, 0]
            np.testing.assert_allclose(got, want, rtol=0, atol=2e-3 * np.abs(want).max())


class TrackerGatingTest(unittest.TestCase):
    def test_search_cost_uses_the_panel_half_space(self):
        """Two candidates mirrored across the O-RU's y-z plane have the same (u_y, u_z); the panel sign picks the right one."""
        from sim.tvt.tracker import MultipathTracker

        tr = object.__new__(MultipathTracker)
        tr.C, tr.oru = 1, np.array([[12.0, 6.5, 5.0]])
        g = np.array([[20.0, -6.0, 1.5], [4.0, -6.0, 1.5]])  # x - x_O-RU = +8 / -8 m, identical (u_y, u_z)
        d = g[0] - tr.oru[0]
        u = d / np.linalg.norm(d)
        base = {"uy": np.array([u[1]]), "uz": np.array([u[2]]), "tau_ns": np.array([0.0]), "valid": np.array([True])}
        c_iso = tr.search_cost([base], g)
        self.assertAlmostEqual(c_iso[0], c_iso[1], places=9)  # one isotropic UPA: mirror-ambiguous
        c_p = tr.search_cost([base | {"sx": np.array([1])}], g)
        c_m = tr.search_cost([base | {"sx": np.array([-1])}], g)
        self.assertLess(c_p[0], 1.0)
        self.assertGreater(c_p[1], 1e11)
        self.assertLess(c_m[1], 1.0)
        self.assertGreater(c_m[0], 1e11)


class SionnaExactTest(unittest.TestCase):
    """Pattern on the traced angles == Sionna trace with TR 38.901 panels (GPU)."""

    @classmethod
    def setUpClass(cls):
        import mitsuba as mi  # noqa: F401
        from sim.scenes.config import load_yaml
        from sim.scenes.traffic import prepare_scenario

        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        cls.raw = raw
        cls.scenario = prepare_scenario(raw, seed=101, mount="lamppost", density="high", duration_s=0.1, dt_s=0.1)

    def _radar(self, arr):
        """One snapshot: arr = "paper1" (cached radar: isotropic, look_at the street centre), "iso0" (isotropic, untilted,
        orientation 0) or a panel sign (TR 38.901 elements, yaw 0 / 180 deg)."""
        from sionna.rt import PlanarArray

        from sim.scenes.motion import states_at
        from sim.sensing.cache import labels_from_scene
        from sim.sensing.radar import _radar_kwargs, build_radar, move_targets  # noqa: F401
        from sim.sensing.trace import pack_radar_paths

        radar = build_radar(self.scenario)
        live = radar["live"]
        if arr != "paper1":
            pat = "iso" if arr == "iso0" else "tr38901"
            live.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern=pat, polarization="V")
            live.rx_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern=pat, polarization="V")
            o = [0.0, 0.0, 0.0] if arr == "iso0" else [float(v) for v in PN.orientation(arr)]
            for name in ("radar-tx", "radar-rx"):
                live.get(name).orientation = o
        states = states_at(self.scenario, 0.0)
        move_targets(live, radar["targets"], states)
        pos = {a["name"]: states[a["name"]]["position_m"] for a in list(self.scenario["vehicles"]) + list(self.scenario["pedestrians"])}
        labels = labels_from_scene(live, pos)
        rcs = pack_radar_paths(radar["rcs"](live, **_radar_kwargs(self.scenario, "rcs")), labels)
        bg = pack_radar_paths(radar["background"](live, **_radar_kwargs(self.scenario, "background")), labels)
        return radar, rcs, bg

    @staticmethod
    def _compare(p0, p1, sx, positions, rot_old, wl):
        """max |a_panel - transform(a_base)| / max |a_panel| over the paths that do not leave / arrive at a pole
        (theta = 0 or 180 deg, where the GCS-LCS polarization angle is undefined); returns (err, n_paths, n_pole)."""
        np.testing.assert_allclose(p1["tau"], p0["tau"], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(p1["theta_r"], p0["theta_r"], atol=1e-5)
        np.testing.assert_allclose(p1["phi_t"], p0["phi_t"], atol=1e-5)
        got, resid = PN.radar_panel_paths(p0, sx, positions, rot_old, wl)
        assert resid < 1e-3, resid
        a1 = p1["a"][0, :, 0, 0, :, 0].astype(np.complex128)
        g = got["a"][0, :, 0, 0, :, 0]
        pole = (np.abs(np.cos(p0["theta_t"][0, 0])) > 1 - 1e-6) | (np.abs(np.cos(p0["theta_r"][0, 0])) > 1 - 1e-6)
        live = (np.abs(a1).max(0) > 0) & ~pole
        err = float(np.abs(g[:, live] - a1[:, live]).max() / np.abs(a1[:, live]).max())
        return err, int(live.sum()), int(pole.sum())

    def test_radar(self):
        """Untilted isotropic trace + pattern == panel trace (both panels); the tilted paper-1 trace is NOT enough
        (its V polarization differs: polarized solver), hence the radar is re-traced untilted."""
        from sim.sensing.channel import rotation_matrix

        wl = 299_792_458.0 / float(self.scenario["carrier_hz"])
        radar, rcs0, bg0 = self._radar("iso0")
        self.assertEqual(rcs0["a"].shape[1], 64)
        rp, rcsp, _ = self._radar("paper1")
        for sx in (1, -1):
            _, rcs1, bg1 = self._radar(sx)
            for name, p0, p1 in (("rcs", rcs0, rcs1), ("background", bg0, bg1)):
                with self.subTest(sx=sx, kind=name):
                    err, n, n_pole = self._compare(p0, p1, sx, radar["positions_m"], np.eye(3), wl)
                    print(f"radar {name} sx={sx:+d}: {n} paths (+{n_pole} at a pole, excluded), max |diff| / max |a| = {err:.2e}")
                    self.assertLess(err, 1e-3)
            err_tilt, _, _ = self._compare(rcsp, rcs1, sx, rp["positions_m"], rotation_matrix(*rp["orientation_rad"].tolist()), wl)
            print(f"radar rcs sx={sx:+d} from the tilted paper-1 trace: max |diff| / max |a| = {err_tilt:.2e} (not exact)")
            self.assertGreater(err_tilt, 1e-2)

    def test_comm_link(self):
        """O-RU 8x8 TX with TR 38.901 panel at yaw 0 / 180 -> single-element UE: a = a_c A(u_dep) exp(j k u_dep . R r_m)."""
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

        from sim.positioning.array import element_positions
        from sim.sensing.radar import _scene_by_name

        oru = [float(v) for v in self.scenario["orus"][0]["position_m"]]
        ues = [[20.0, -6.0, 1.5], [2.0, 7.5, 1.5]]
        wl = 299_792_458.0 / float(self.scenario["carrier_hz"])
        r = element_positions(wl)
        k = 2 * math.pi / wl
        for ue in ues:
            out = {}
            for sx in (None, 1, -1):
                scene = load_scene(_scene_by_name(str(self.scenario["scene"])))
                scene.frequency = float(self.scenario["carrier_hz"])
                pat = "iso" if sx is None else "tr38901"
                scene.tx_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern=pat, polarization="V")
                scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
                o = [0.0, 0.0, 0.0] if sx is None else [float(v) for v in PN.orientation(sx)]
                scene.add(Transmitter("oru", position=oru, orientation=o))
                scene.add(Receiver("ue", position=ue))
                paths = PathSolver()(scene, max_depth=2, synthetic_array=True, los=True, specular_reflection=True, diffuse_reflection=False,
                                     refraction=False)
                a, tau = paths.cir(sampling_frequency=1.0, num_time_steps=1, normalize_delays=False, out_type="numpy")
                out[sx] = (np.asarray(a)[0, 0, 0, :, :, 0], np.asarray(tau), np.asarray(paths.theta_t.numpy())[0, 0], np.asarray(paths.phi_t.numpy())[0, 0])
            a0, tau0, th, ph = out[None]  # a [tx_ant, P]
            u = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)  # [P, 3]
            live = np.abs(a0).max(axis=0) > 0
            centre = (a0 * np.exp(1j * k * r @ u.T).conj()).mean(axis=0)
            for sx in (1, -1):
                a1, tau1, _, _ = out[sx]
                perm = np.array([int(np.argmin(np.abs(tau1.ravel() - t))) for t in tau0.ravel()])  # path order may differ
                np.testing.assert_allclose(tau1.ravel()[perm], tau0.ravel(), rtol=1e-6, atol=1e-12)
                a1 = a1[:, perm]
                want = centre[None] * PN.amplitude(u, sx)[None] * np.exp(1j * k * (r @ PN.rotation(sx).T) @ u.T)
                err = np.abs(a1[:, live] - want[:, live]).max() / np.abs(a1[:, live]).max()
                print(f"comm ue {ue} sx={sx:+d}: {int(live.sum())} paths, max |diff| / max |a| = {err:.2e}")
                self.assertLess(err, 1e-3)


if __name__ == "__main__":
    unittest.main()
