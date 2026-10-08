"""Tests of the back-to-back panel model (sim/tvt/panels.py; configs/tvt.yaml panels:).

PatternTest / SelectionTest / RadarTransformTest are pure NumPy. SionnaExactTest (GPU, ~1-2 min) traces one
radar snapshot of the paper-1 canyon twice - with the unchanged isotropic paper-1 radar (look_at the street
centre) and with a TR 38.901 panel at yaw 0 / 180 deg (and 90 / 270 deg, the intersection's four panels) - and checks that the pattern applied to the traced path
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


class YawPanelTest(unittest.TestCase):
    """More than two panels (configs/tvt.yaml panels.yaw_deg_by_mount; the intersection: yaw 0, 180, 90, 270 deg)."""

    def test_config_by_mount(self):
        self.assertEqual(PN.yaws_deg("lamppost"), [0.0, 180.0])
        self.assertEqual(PN.yaws_deg("facade_ueslow"), [0.0, 180.0])
        self.assertEqual(PN.yaws_deg("corner"), [0.0, 180.0, 90.0, 270.0])
        self.assertTrue(PN.is_pair(PN.yaws_deg("lamppost")))
        self.assertFalse(PN.is_pair(PN.yaws_deg("corner")))
        with self.assertRaises(ValueError):
            PN.yaws_deg("x", {"yaw_deg": [0.0, 45.0]})

    def test_rotations_exact(self):
        for y in (0.0, 90.0, 180.0, 270.0):
            R = PN.rot_z(y)
            np.testing.assert_array_equal(R @ R.T, np.eye(3))
            np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [math.cos(math.radians(y)), math.sin(math.radians(y)), 0.0], atol=1e-15)
        np.testing.assert_array_equal(PN.rot_z(180.0), PN.rotation(-1))
        self.assertEqual(PN.split(270.0), (90.0, -1))
        self.assertEqual(PN.split(90.0), (90.0, 1))
        self.assertEqual(PN.split(180.0), (0.0, -1))

    def test_amplitude_yaw(self):
        """yaw 0 / 180 = the third-round panels exactly; yaw 90 / 270: boresight +y / -y, mirrored pattern."""
        rng = np.random.default_rng(5)
        u = np.stack([u_az_el(a, e) for a, e in zip(rng.uniform(-180, 180, 300), rng.uniform(-80, 80, 300))])
        np.testing.assert_array_equal(PN.amplitude_yaw(u, 0.0), PN.amplitude(u, 1))
        np.testing.assert_array_equal(PN.amplitude_yaw(u, 180.0), PN.amplitude(u, -1))
        u90 = u @ PN.rot_z(90.0).T  # direction rotated by +90 deg: seen by the yaw-90 panel as u by the +x panel
        np.testing.assert_allclose(PN.amplitude_yaw(u90, 90.0), PN.amplitude(u, 1), rtol=1e-12)
        np.testing.assert_allclose(PN.amplitude_yaw(u90, 270.0), PN.amplitude(u, -1), rtol=1e-12)
        self.assertAlmostEqual(db(PN.amplitude_yaw(u_az_el(90, 0), 90.0)), 8.0, places=9)
        self.assertAlmostEqual(db(PN.amplitude_yaw(u_az_el(-90, 0), 270.0)), 8.0, places=9)
        self.assertAlmostEqual(db(PN.amplitude_yaw(u_az_el(-90, 0), 90.0)), -22.0, places=9)

    def test_select(self):
        """Nearest boresight azimuth; for the +-x pair identical to ``facing`` (+x on a tie); first listed on a tie."""
        rng = np.random.default_rng(6)
        t = rng.uniform(-40, 40, (500, 2))
        o = np.array([7.5, 7.0])
        np.testing.assert_array_equal(PN.select(t, o, [0.0, 180.0]), np.where(PN.facing(t[:, 0], o[0]) > 0, 0, 1))
        ys = [0.0, 180.0, 90.0, 270.0]
        k = PN.select(t, o, ys)
        az = np.degrees(np.arctan2(t[:, 1] - o[1], t[:, 0] - o[0]))
        want = [int(np.argmin([abs((a - y + 180) % 360 - 180) for y in ys])) for a in az]
        np.testing.assert_array_equal(k, want)
        self.assertEqual(int(PN.select(np.array([7.0, -20.0]), o, ys)), 3)  # UE 1 on the cross street: the south panel
        self.assertEqual(int(PN.select(np.array([8.5, 8.0]), o, ys)), 0)  # exactly 45 deg: first listed (east)
        self.assertEqual(int(PN.select(np.array([7.5, 7.0]), o, ys)), 0)

    def test_link_amplitude_yaws(self):
        """Pair: equal to link_amplitude; four panels: amplitude of the selected panel and directions in its pair-axis frame."""
        rng = np.random.default_rng(7)
        T, U, C, P = 3, 2, 2, 5
        u = rng.standard_normal((T, U, C, P, 3))
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        ue = np.concatenate([rng.uniform(-30, 30, (T, U, 2)), np.full((T, U, 1), 1.5)], -1)
        oru = np.array([[7.5, 7.0, 5.0], [-7.5, -7.0, 5.0]])
        a0, sx = PN.link_amplitude(u, ue, oru)
        a1, yaw, u1 = PN.link_amplitude_yaws(u, ue, oru, [0.0, 180.0])
        np.testing.assert_array_equal(a0, a1)
        np.testing.assert_array_equal(yaw, np.where(sx > 0, 0.0, 180.0))
        ys = [0.0, 180.0, 90.0, 270.0]
        a4, yaw4, u4 = PN.link_amplitude_yaws(u, ue, oru, ys)
        for t in range(T):
            for i in range(U):
                for c in range(C):
                    y = ys[int(PN.select(ue[t, i, :2], oru[c, :2], ys))]
                    self.assertEqual(yaw4[t, i, c], y)
                    np.testing.assert_allclose(a4[t, i, c], PN.amplitude_yaw(u[t, i, c], y), rtol=1e-12)
                    np.testing.assert_allclose(u4[t, i, c], u[t, i, c] @ PN.rot_z(y % 180.0), rtol=0, atol=1e-15)

    def test_codebook_gain_of_rotated_panel(self):
        """|w^H s|^2 with the yaw-90 grid (world positions R r) equals the unrotated grid with R_90^T u (and the
        yaw-270 grid the same set of codebook gains)."""
        from sim.positioning.array import element_positions
        from sim.tvt.service import codebook

        wl = 299_792_458.0 / 28e9
        r = element_positions(wl)
        k = 2 * math.pi / wl
        W = codebook(wl, 2)
        u = u_az_el(70.0, -15.0)
        for y in (90.0, 270.0):
            s_world = np.exp(1j * k * (r @ PN.rot_z(y).T) @ u)
            s_frame = np.exp(1j * k * r @ PN.to_frame(u, 90.0))
            np.testing.assert_allclose(np.sort(np.abs(W.conj() @ s_world)), np.sort(np.abs(W.conj() @ s_frame)), rtol=1e-9, atol=1e-9)

    def test_stored_frame(self):
        """A panel's local measurement times its sign is the stored (frame R_a) u_y; the true direction is one of the
        two stored candidates."""
        rng = np.random.default_rng(8)
        for y in (0.0, 180.0, 90.0, 270.0):
            a, sx = PN.split(y)
            for _ in range(50):
                u = rng.standard_normal(3)
                u /= np.linalg.norm(u)
                local = PN.to_frame(u, y)
                ty, fs = PN.transverse(u, a)
                self.assertAlmostEqual(sx * local[1], float(ty), places=12)
                self.assertEqual(int(fs), 1 if sx * local[0] >= 0 else -1)
                cand = PN.stored_candidates(np.array(ty), np.array(u[2]), a)
                self.assertLess(min(np.linalg.norm(cand[i] - u) for i in range(2)), 1e-9)

    def test_keep_sector(self):
        det = [{"x_m": 30.0, "y_m": 7.0}, {"x_m": 7.0, "y_m": -30.0}, {"x_m": -20.0, "y_m": 7.5}, {"x_m": 8.0, "y_m": 40.0}]
        ys = [0.0, 180.0, 90.0, 270.0]
        o = np.array([7.5, 7.0, 5.0])
        got = [[d["x_m"] for d in PN.keep_sector(det, i, ys, o)] for i in range(4)]
        self.assertEqual(got, [[30.0], [-20.0], [8.0], [7.0]])
        self.assertEqual(PN.keep_sector(det, 0, [0.0, 180.0], o), PN.keep_facing(det, 1, 7.5))


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
        for y in (0.0, 180.0, 90.0, 270.0):
            out, resid = PN.radar_panel_paths_yaw(packed, y, r, rot_old, wl)
            self.assertLess(resid, 2e-3)
            want = c[None] * PN.amplitude_yaw(u_t, y)[None] * PN.amplitude_yaw(u_r, y)[None] * np.exp(1j * k * (r @ PN.rot_z(y).T) @ u_r.T)
            np.testing.assert_allclose(out["a"][0, :, 0, 0, :, 0], want, rtol=0, atol=2e-3 * np.abs(want).max())
            if y in (0.0, 180.0):
                np.testing.assert_array_equal(out["a"], PN.radar_panel_paths(packed, 1 if y == 0.0 else -1, r, rot_old, wl)[0]["a"])


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
        c_a0 = tr.search_cost([base | {"sx": np.array([1]), "pa": np.array([0.0])}], g)  # pair axis 0 = the third-round rule
        np.testing.assert_array_equal(c_a0, c_p)

    def test_search_cost_perpendicular_panel(self):
        """A +-y panel component (pair axis 90: stored u_y = -u_x) explains the LoS of the right position only."""
        from sim.tvt.tracker import MultipathTracker

        tr = object.__new__(MultipathTracker)
        tr.C, tr.oru = 1, np.array([[7.5, 7.0, 5.0]])
        g = np.array([[7.0, -20.0, 1.5], [7.0, 34.0, 1.5], [-20.0, 7.5, 1.5]])  # south, north (mirror across y = 7), west
        d = g[0] - tr.oru[0]
        u = d / np.linalg.norm(d)
        ty, fs = PN.transverse(u, 90.0)
        base = {"uy": np.array([float(ty)]), "uz": np.array([u[2]]), "tau_ns": np.array([0.0]), "valid": np.array([True]), "pa": np.array([90.0])}
        c = tr.search_cost([base | {"sx": np.array([int(fs)])}], g)
        self.assertEqual(int(fs), -1)  # the south panel (yaw 270)
        self.assertLess(c[0], 1e-9)
        self.assertGreater(c[1], 1e11)  # mirror image in front of the north panel: excluded by the half-space
        self.assertGreater(c[2], 1.0)

    def test_predict_meas_rotated_frame(self):
        """predict_meas with pair axis 90: u_y = -u_x of the world direction, same delay, Jacobian = finite differences."""
        from sim.tvt.sources import faces_from_geometry
        from sim.tvt.tracker import MultipathTracker, TrackerParams

        faces = faces_from_geometry(ROOT / "configs" / "scenes" / "tvt_intersection" / "geometry.json")
        oru = np.array([[7.5, 7.0, 5.0], [-7.5, -7.0, 5.0]])
        tr = MultipathTracker(faces, oru, 1e9 / 120e3, {"rects": [[-40.0, 40.0, -8.6, 9.6]]}, TrackerParams())
        x = np.zeros(tr.n)
        x[0], x[1] = 3.0, -6.0
        idx = np.arange(min(12, len(tr.sources)))
        for c in range(2):
            h0, H0 = tr.predict_meas(x, c, idx)
            sx0 = tr._pred_sx.copy()
            h9, H9 = tr.predict_meas(x, c, idx, 90.0)
            sx9 = tr._pred_sx.copy()
            np.testing.assert_allclose(h9[:, [0, 2]], h0[:, [0, 2]], atol=1e-9)
            np.testing.assert_allclose(H9[:, [0, 2]], H0[:, [0, 2]], atol=1e-9)
            from sim.tvt.sources import predict

            N, A, V = tr._planes(x)
            pr = predict(oru[c], np.array([x[0], x[1], tr.z]), N[idx], A[idx], V[idx])
            np.testing.assert_allclose(h9[:, 1], -pr["u"][:, 0], atol=1e-12)
            np.testing.assert_array_equal(sx9, np.where(pr["u"][:, 1] >= 0, 1, -1))
            np.testing.assert_array_equal(sx0, np.where(pr["u"][:, 0] >= 0, 1, -1))
            eps = 1e-6
            for j in range(2):
                xp, xm = x.copy(), x.copy()
                xp[j] += eps
                xm[j] -= eps
                num = (tr.predict_meas(xp, c, idx, 90.0)[0] - tr.predict_meas(xm, c, idx, 90.0)[0]) / (2 * eps)
                np.testing.assert_allclose(H9[:, :, j], num, atol=1e-5)


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
            if arr == "iso0":
                o = [0.0, 0.0, 0.0]
            elif isinstance(arr, tuple):  # ("yaw", deg): a panel of the intersection's four-panel set
                o = [math.radians(arr[1]), 0.0, 0.0]
            else:
                o = [float(v) for v in PN.orientation(arr)]
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
    def _compare(p0, p1, sx, positions, rot_old, wl, yaw=None):
        """max |a_panel - transform(a_base)| / max |a_panel| over the paths that do not leave / arrive at a pole
        (theta = 0 or 180 deg, where the GCS-LCS polarization angle is undefined); returns (err, n_paths, n_pole)."""
        np.testing.assert_allclose(p1["tau"], p0["tau"], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(p1["theta_r"], p0["theta_r"], atol=1e-5)
        np.testing.assert_allclose(p1["phi_t"], p0["phi_t"], atol=1e-5)
        got, resid = PN.radar_panel_paths(p0, sx, positions, rot_old, wl) if yaw is None else PN.radar_panel_paths_yaw(p0, yaw, positions, rot_old, wl)
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
        for y in (90.0, 270.0):  # the intersection's +y / -y panels (fourth round)
            _, rcs1, bg1 = self._radar(("yaw", y))
            for name, p0, p1 in (("rcs", rcs0, rcs1), ("background", bg0, bg1)):
                with self.subTest(yaw=y, kind=name):
                    err, n, n_pole = self._compare(p0, p1, None, radar["positions_m"], np.eye(3), wl, yaw=y)
                    print(f"radar {name} yaw={y:g}: {n} paths (+{n_pole} at a pole, excluded), max |diff| / max |a| = {err:.2e}")
                    self.assertLess(err, 1e-3)

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
            for sx in (None, 1, -1, 90.0, 270.0):  # None: isotropic; +-1: third-round panels; 90 / 270: yaw [deg]
                scene = load_scene(_scene_by_name(str(self.scenario["scene"])))
                scene.frequency = float(self.scenario["carrier_hz"])
                pat = "iso" if sx is None else "tr38901"
                scene.tx_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern=pat, polarization="V")
                scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
                o = [0.0, 0.0, 0.0] if sx is None else ([math.radians(sx), 0.0, 0.0] if isinstance(sx, float) else [float(v) for v in PN.orientation(sx)])
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
            for sx in (1, -1, 90.0, 270.0):
                a1, tau1, _, _ = out[sx]
                perm = np.array([int(np.argmin(np.abs(tau1.ravel() - t))) for t in tau0.ravel()])  # path order may differ
                np.testing.assert_allclose(tau1.ravel()[perm], tau0.ravel(), rtol=1e-6, atol=1e-12)
                a1 = a1[:, perm]
                if isinstance(sx, float):
                    want = centre[None] * PN.amplitude_yaw(u, sx)[None] * np.exp(1j * k * (r @ PN.rot_z(sx).T) @ u.T)
                else:
                    want = centre[None] * PN.amplitude(u, sx)[None] * np.exp(1j * k * (r @ PN.rotation(sx).T) @ u.T)
                err = np.abs(a1[:, live] - want[:, live]).max() / np.abs(a1[:, live]).max()
                print(f"comm ue {ue} panel {sx}: {int(live.sum())} paths, max |diff| / max |a| = {err:.2e}")
                self.assertLess(err, 1e-3)


if __name__ == "__main__":
    unittest.main()
