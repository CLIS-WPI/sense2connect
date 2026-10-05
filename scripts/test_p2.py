"""Paper 2 tests (unittest). Run in the container: python scripts/test_p2.py [-v].

Tests that need cached data (paper-1 geometry, results/P2) are skipped when it is absent.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sim.positioning.array import element_positions, steering, unit_direction  # noqa: E402
from sim.positioning.geometry import C0, bounce_planes, first_segment_direction, path_geometry, polyline_length  # noqa: E402

WL = C0 / 28e9
JOB = ROOT / "results" / "cache" / "lamppost" / "low" / "seed_1001"
F32_EPS = float(np.finfo(np.float32).eps)


def _known_planes() -> dict:
    from sim.scenes.config import load_yaml

    return load_yaml(ROOT / "configs" / "p2.yaml")["known_planes"]


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class GeometryTest(unittest.TestCase):
    def test_los_derivative_analytic_vs_fd(self):
        rng = np.random.default_rng(1)
        oru = np.array([12.0, 6.5, 5.0])
        for _ in range(50):
            ue = np.array([rng.uniform(-40, 40), rng.uniform(-8, -6), 1.5])
            N = np.zeros((2, 3))
            A = np.zeros((2, 3))
            V = np.zeros(2, dtype=bool)
            g = path_geometry(oru, ue, N, A, V)
            for k in range(2):
                e = np.zeros(3)
                e[k] = 1e-6
                gp, gm = path_geometry(oru, ue + e, N, A, V), path_geometry(oru, ue - e, N, A, V)
                for q, dq in (("tau", "dtau"), ("el", "del")):
                    fd = (gp[q] - gm[q]) / 2e-6
                    self.assertLess(abs(fd - g[dq][k]), 1e-4 * abs(g[dq][k]) + 1e-15)
                fd = np.angle(np.exp(1j * (gp["az"] - gm["az"]))) / 2e-6
                self.assertLess(abs(fd - g["daz"][k]), 1e-4 * abs(g["daz"][k]) + 1e-12)

    def test_image_method_reproduces_cached_paths(self):
        if not (JOB / "comm_geometry.npz").exists():
            self.skipTest("paper-1 geometry cache absent")
        d = np.load(JOB / "comm_geometry.npz")
        P, n, cls = d["points_m"], d["n_points"].astype(int), d["path_class"]
        ue = np.broadcast_to(d["ue_position_m"][:, :, None, None, :], P.shape[:-2] + (3,))
        oru = np.broadcast_to(d["oru_position_m"][:, None, :, None, :], P.shape[:-2] + (3,))
        N, A, V = bounce_planes(P, n, _known_planes())
        g = path_geometry(oru, ue, N, A, V)
        ok = cls >= 0
        # snapped planes vs mm-rounded bounce points: delay within 2 mm, direction within 1e-4
        self.assertLess(np.abs(g["tau"] * C0 - polyline_length(P, n))[ok].max(), 2e-3)
        # cached bounce points are rounded to 1 mm: direction tolerance 1 mm / range to the first point
        rng1 = np.linalg.norm(np.nan_to_num(P[..., 1, :]) - oru, axis=-1)
        dev = np.linalg.norm(g["u"] - first_segment_direction(P), axis=-1)
        self.assertTrue(np.all(dev[ok] <= 1e-3 / rng1[ok]), f"{np.max(dev[ok] * rng1[ok]):.2e} m")

    def test_nlos_derivative_vs_retrace(self):
        """+/-1 cm re-trace (scripts/p2_trace.py --fd) vs analytic image-method derivatives.

        Criterion: |fd - analytic| <= 1e-3 |analytic| + 3 x the single-precision floor of
        the Sionna finite difference: sqrt(2) eps32 |tau| / (2 h) for the delay and
        sqrt(2) v / (2 h r) for the angles, with v = 2e-5 m the precision of Sionna's
        float32 interaction points (ray-surface intersections; empirical) and r the
        horizontal range (azimuth) or the range (elevation) to the first point. The share
        meeting 1e-3 relative WITHOUT the floor is printed (documented deviation: the
        roadmap's 1e-3 criterion cannot be met everywhere at single precision).
        """
        files = sorted((ROOT / "results" / "P2" / "cache").glob("*/*/seed_*/p2_fd.npz"))
        if not files:
            self.skipTest("no p2_fd.npz (run p2_trace.py --fd)")
        n_strict = n_all = 0
        for fp in files:
            mount, density, sd = fp.parts[-4], fp.parts[-3], fp.parts[-2]
            d = np.load(ROOT / "results" / "cache" / mount / density / sd / "comm_geometry.npz")
            f = np.load(fp)
            h = float(np.abs(f["offsets"]).max())
            ns = f["fd_tau"].shape[0]
            P, n, cls = d["points_m"][:ns], d["n_points"][:ns].astype(int), d["path_class"][:ns]
            ue = np.broadcast_to(d["ue_position_m"][:ns, :, None, None, :], P.shape[:-2] + (3,))
            oru = np.broadcast_to(d["oru_position_m"][:ns, None, :, None, :], P.shape[:-2] + (3,))
            N, A, V = bounce_planes(P, n, _known_planes())
            g = path_geometry(oru, ue, N, A, V)
            first = np.nan_to_num(P[..., 1, :])
            rng_first = np.linalg.norm(first - oru, axis=-1)
            rng_h = np.linalg.norm((first - oru)[..., :2], axis=-1)  # azimuth noise scales with the horizontal range
            vprec = 2e-5
            for k, (ip, im) in enumerate(((0, 1), (2, 3))):
                uu = f["fd_u"]
                az = np.arctan2(uu[..., 1], uu[..., 0])
                el = np.arcsin(np.clip(uu[..., 2], -1, 1))
                cand = {
                    "tau": ((f["fd_tau"][..., ip] - f["fd_tau"][..., im]) / (2 * h), g["dtau"][..., k], math.sqrt(2) * F32_EPS * g["tau"] / (2 * h)),
                    "az": (np.angle(np.exp(1j * (az[..., ip] - az[..., im]))) / (2 * h), g["daz"][..., k], math.sqrt(2) * vprec / (2 * h * rng_h)),
                    "el": ((el[..., ip] - el[..., im]) / (2 * h), g["del"][..., k], math.sqrt(2) * vprec / (2 * h * rng_first)),
                }
                for name, (fd, an, floor) in cand.items():
                    ok = (cls >= 0) & np.isfinite(fd)
                    err = np.abs(fd - an)[ok]
                    tol = (1e-3 * np.abs(an) + 3 * floor)[ok]
                    self.assertTrue(np.all(err <= tol), f"{fp} {name}: {np.max(err / tol):.2f} x tolerance")
                    n_strict += int(np.sum(err <= 1e-3 * np.abs(an[ok])))
                    n_all += int(ok.sum())
        print(f"\n  FD validation: {n_all} derivatives, {100 * n_strict / max(n_all, 1):.1f} % within 1e-3 relative without the float32 floor", file=sys.stderr)


class ArrayTest(unittest.TestCase):
    def test_positions_match_sionna_layout(self):
        r = element_positions(WL) / WL
        self.assertTrue(np.allclose(r[0], [0, -1.75, 1.75]) and np.allclose(r[1], [0, -1.75, 1.25]) and np.allclose(r[8], [0, -1.25, 1.75]))

    def test_steering_convention_against_sionna(self):
        metas = sorted((ROOT / "results" / "P2" / "cache").glob("*/*/seed_*/p2_paths_meta.json"))
        if not metas:
            self.skipTest("no p2_paths caches")
        for m in metas:
            meta = json.loads(m.read_text())
            self.assertLess(meta["max_steering_residual"], 1e-5, m)
            self.assertLess(meta["max_power_rel_err"], 1.5e-5, m)
            self.assertEqual(meta["key_mismatch"], 0, m)


@unittest.skipUnless(_cuda(), "needs the GPU")
class FisherTest(unittest.TestCase):
    def _toy(self, P=3, N=48, seed=0):
        rng = np.random.default_rng(seed)
        beta = (rng.normal(size=P) + 1j * rng.normal(size=P)) * 20
        tau = rng.uniform(50, 120, P)
        az = rng.uniform(-np.pi, np.pi, P)
        el = rng.uniform(-0.4, 0.2, P)
        f = (np.arange(N) - (N - 1) / 2) * 120e3 * 8
        return beta, tau, az, el, f

    def test_gram_torch_equals_numpy(self):
        import torch

        from sim.positioning.fim import gram_numpy, gram_torch

        r = element_positions(WL)
        for seed in range(3):
            beta, tau, az, el, f = self._toy(seed=seed)
            Jn = gram_numpy(beta, tau, az, el, f, r, WL)
            Jt = gram_torch(*(torch.tensor(x)[None] for x in (beta, tau, az, el)), f, r, WL)[0].numpy()
            self.assertLess(np.abs(Jn - Jt).max(), 1e-9 * np.abs(Jn).max())

    def test_efim_vs_bruteforce_numeric_fim(self):
        """Toy: 2 O-RUs x (LoS + one facade reflection); unknowns x, y, gains, delta_c, b, 64 element phases per O-RU."""
        import torch

        from sim.positioning.fim import calibrated_efim, geometric_efim, gram_torch, peb_from_theta, theta_information

        r = element_positions(WL)
        ue = np.array([5.0, -7.0, 1.5])
        orus = [np.array([12.0, 6.5, 5.0]), np.array([52.0, -7.0, 5.0])]
        wall = np.array([0.0, 1.0, 0.0]), np.array([0.0, 9.571564, 0.0])
        N = 32
        df = 120e3 * 8
        f = (np.arange(N) - (N - 1) / 2) * df
        sig_sync, sig_phi = 0.5, math.radians(3.0)
        gains = [[30 + 10j, 8 - 4j], [20 - 5j, 6 + 6j]]

        def path_params(pos):
            out = []
            for c, o in enumerate(orus):
                rows = []
                for pth in range(2):
                    Nn = np.zeros((2, 3))
                    A = np.zeros((2, 3))
                    V = np.zeros(2, dtype=bool)
                    if pth == 1:
                        Nn[0], A[0], V[0] = wall[0], wall[1], True
                    g = path_geometry(o, pos, Nn, A, V)
                    rows.append(g)
                out.append(rows)
            return out

        # theta layout: x, y, delta0, b, delta1, gains (2 O-RU x 2 paths x Re/Im), psi0[64], psi1[64]
        n_th = 2 + 3 + 8 + 128
        theta0 = np.zeros(n_th)
        theta0[0:2] = ue[:2]
        for c in range(2):
            for pth in range(2):
                theta0[5 + 4 * c + 2 * pth] = gains[c][pth].real
                theta0[5 + 4 * c + 2 * pth + 1] = gains[c][pth].imag

        def signal(th):
            pos = np.array([th[0], th[1], 1.5])
            geo = path_params(pos)
            delta = [th[2], th[4]]
            b = th[3]
            out = []
            for c in range(2):
                psi = th[13 + 64 * c: 13 + 64 * (c + 1)]
                m = np.zeros((N, 64), dtype=complex)
                for pth in range(2):
                    g = geo[c][pth]
                    beta = th[5 + 4 * c + 2 * pth] + 1j * th[5 + 4 * c + 2 * pth + 1]
                    tau = g["tau"] * 1e9 + delta[c] + b
                    m += beta * np.outer(np.exp(-2j * np.pi * f * tau * 1e-9), steering(g["u"], r, WL))
                out.append((m * np.exp(1j * psi)[None, :]).reshape(-1))
            return np.concatenate(out)

        # numeric Jacobian (central differences, step scaled per parameter)
        steps = np.full(n_th, 1e-6)
        steps[2:5] = 1e-6
        D = np.zeros((2 * N * 64, n_th), dtype=complex)
        for i in range(n_th):
            e = np.zeros(n_th)
            e[i] = steps[i]
            D[:, i] = (signal(theta0 + e) - signal(theta0 - e)) / (2 * steps[i])
        J = 2 * np.real(D.conj().T @ D)
        J[2, 2] += 1 / sig_sync ** 2
        J[4, 4] += 1 / sig_sync ** 2
        J[13:, 13:] += np.eye(128) / sig_phi ** 2
        keep = [0, 1]
        drop = [i for i in range(n_th) if i not in keep]
        Jp = J[np.ix_(keep, keep)] - J[np.ix_(keep, drop)] @ np.linalg.solve(J[np.ix_(drop, drop)], J[np.ix_(drop, keep)])
        peb_brute = math.sqrt(np.trace(np.linalg.inv(Jp)))
        # pipeline
        geo = path_params(ue)
        Ks, Hs = [], []
        for c in range(2):
            beta = np.array(gains[c])
            tau = np.array([geo[c][p]["tau"] * 1e9 for p in range(2)])
            az = np.array([geo[c][p]["az"] for p in range(2)])
            el = np.array([geo[c][p]["el"] for p in range(2)])
            Jf = gram_torch(*(torch.tensor(x)[None] for x in (beta, tau, az, el)), f, r, WL)
            K = geometric_efim(Jf, 2)
            Ks.append(calibrated_efim(K, torch.zeros((1, 6), dtype=torch.bool), sig_phi, 6))
            H = np.zeros((6, 5))
            for p in range(2):
                H[3 * p, 0:2] = geo[c][p]["dtau"] * 1e9
                H[3 * p, 2 + c] = 1.0
                H[3 * p, 4] = 1.0
                H[3 * p + 1, 0:2] = geo[c][p]["daz"]
                H[3 * p + 2, 0:2] = geo[c][p]["del"]
            Hs.append(H)
        Keff = torch.stack([k[0] for k in Ks])[None]
        J0 = theta_information(Keff, torch.tensor(np.stack(Hs))[None])
        prior = torch.zeros((1, 5, 5), dtype=torch.float64)
        prior[0, 2, 2] = prior[0, 3, 3] = 1 / sig_sync ** 2
        present = torch.tensor([[True, True, True, True, True]])
        peb = float(peb_from_theta(J0, prior, present)[0])
        self.assertLess(abs(peb - peb_brute) / peb_brute, 1e-3, f"pipeline {peb} vs brute force {peb_brute}")

    def test_peb_scales_with_snr_and_bandwidth(self):
        import torch

        from sim.positioning.fim import geometric_efim, gram_torch

        r = element_positions(WL)
        beta, tau, az, el, _ = self._toy(P=1)
        ks = {}
        for n in (32, 128):
            f = (np.arange(n) - (n - 1) / 2) * 120e3
            b = beta * math.sqrt(32 / n)  # same total energy
            K = geometric_efim(gram_torch(*(torch.tensor(x)[None] for x in (b, tau, az, el)), f, r, WL), 1)[0].numpy()
            ks[n] = K
        # delay information ~ bandwidth^2 at equal energy (rms bandwidth ratio 4 -> information ratio 16)
        ratio = ks[128][0, 0] / ks[32][0, 0]
        brms = lambda n: np.mean(((np.arange(n) - (n - 1) / 2)) ** 2)  # noqa: E731
        self.assertLess(abs(ratio - brms(128) / brms(32)) / ratio, 1e-6)
        # information ~ SNR: doubling the amplitude multiplies K by 4 (all parameters)
        f = (np.arange(32) - 15.5) * 120e3
        K1 = geometric_efim(gram_torch(*(torch.tensor(x)[None] for x in (beta, tau, az, el)), f, r, WL), 1)[0].numpy()
        K2 = geometric_efim(gram_torch(*(torch.tensor(x)[None] for x in (2 * beta, tau, az, el)), f, r, WL), 1)[0].numpy()
        self.assertLess(np.abs(K2[:3, :3] - 4 * K1[:3, :3]).max(), 1e-8 * np.abs(K2[:3, :3]).max())


@unittest.skipUnless(_cuda(), "needs the GPU")
class EstimatorTest(unittest.TestCase):
    def _case(self, N=96, B=5, seed=0):
        rng = np.random.default_rng(seed)
        P = 3
        beta = (rng.normal(size=(B, P)) + 1j * rng.normal(size=(B, P))) * np.array([60.0, 15.0, 8.0])
        tau = rng.uniform(40, 160, (B, P))
        az = rng.uniform(-np.pi, np.pi, (B, P))
        el = rng.uniform(-0.4, -0.05, (B, P))
        u = unit_direction(az, el)
        psi = rng.normal(0, 0.05, (B, 64))
        f = (np.arange(N) - (N - 1) / 2) * 120e3 * 4
        noise = (rng.normal(size=(B, N, 64)) + 1j * rng.normal(size=(B, N, 64))) / math.sqrt(2)
        return beta, tau, u, psi, f, noise

    def test_synthesis_torch_equals_numpy(self):
        import torch

        from sim.positioning.estimator import synth_numpy, synth_torch

        beta, tau, u, psi, f, _ = self._case()
        r = element_positions(WL)
        Yt = synth_torch(torch.tensor(beta), torch.tensor(tau), torch.tensor(u), f, r, WL, psi=torch.tensor(psi), add_noise=False).numpy()
        for b in range(beta.shape[0]):
            Yn = synth_numpy(beta[b], tau[b], u[b], f, r, WL, psi=psi[b])
            self.assertLess(np.abs(Yt[b] - Yn).max(), 1e-5 * np.abs(Yn).max())

    def test_measurement_torch_equals_numpy(self):
        import torch

        from sim.positioning.estimator import measure_numpy, measure_torch, synth_numpy

        beta, tau, u, psi, f, noise = self._case()
        r = element_positions(WL)
        Y = np.stack([synth_numpy(beta[b], tau[b], u[b], f, r, WL, psi=psi[b], noise=noise[b]) for b in range(beta.shape[0])]).astype(np.complex64)
        mt = measure_torch(torch.tensor(Y), f, WL)
        mn = measure_numpy(Y.astype(np.complex128), f, WL)
        for k in ("tau_ns", "uy", "uz"):
            self.assertLess(np.abs(mt[k] - mn[k]).max(), 1e-4, k)
        self.assertLess(np.abs(mt["ratio_db"] - mn["ratio_db"]).max(), 1e-3)

    def test_noiseless_single_path_measurement_is_accurate(self):
        import torch

        from sim.positioning.estimator import measure_torch, synth_torch

        f = (np.arange(792) - 395.5) * 120e3
        r = element_positions(WL)
        u = unit_direction(np.array([[2.5]]), np.array([[-0.2]]))
        Y = synth_torch(torch.tensor([[100.0 + 0j]]), torch.tensor([[123.4]]), torch.tensor(u), f, r, WL, add_noise=False)
        m = measure_torch(Y, f, WL)
        self.assertLess(abs(m["tau_ns"][0] - 123.4), 0.05)
        self.assertLess(abs(m["uy"][0] - u[0, 0, 1]), 2e-3)
        self.assertLess(abs(m["uz"][0] - u[0, 0, 2]), 2e-3)


class MirrorTest(unittest.TestCase):
    """Variant A: a UE behind an O-RU's array plane must be resolved (not placed at its front/back mirror image)."""

    WMAP = {"x_min": -40.0, "x_max": 40.0, "sidewalks_y": [(-8.613335, -4.25), (4.25, 9.571564)], "street_y": (-8.613335, 9.571564),
            "crossings_x": [], "crossing_half_width": 1.5}
    PARAMS = {"gate_db": 0.0, "floor_tau_ns": 0.3, "floor_u": 0.002, "res_gate": 30.0, "q": 0.1, "chi2": 1e9, "sel_gate": 1e9, "refl_penalty": 10.0}

    def _track(self, xs, oru, timing="tdoa", noise=0.0, variant=None, reflect_c=None):
        from sim.positioning.estimator import noise_model
        from sim.positioning.estimator_v2 import candidates, reflection_planes, run_tracker_v2

        E = len(xs)
        ue = np.stack([xs, np.full(E, -7.0), np.full(E, 1.5)], -1)
        o = np.broadcast_to(np.array(oru), (E, 2, 3))
        rng = np.random.default_rng(0)
        m = {k: np.zeros((E, 2)) for k in ("tau_ns", "uy", "uz")}
        for c in range(2):
            q = ue.copy()
            if c == reflect_c:  # dominant path = north-facade reflection (image of the UE across y = 9.571564)
                q[:, 1] = 2 * 9.571564 - q[:, 1]
            d = q - o[:, c]
            r = np.linalg.norm(d, axis=-1)
            m["tau_ns"][:, c] = r / C0 * 1e9
            m["uy"][:, c] = d[:, 1] / r + noise * rng.standard_normal(E)
            m["uz"][:, c] = d[:, 2] / r + noise * rng.standard_normal(E)
        m["snr"] = np.full((E, 2), 1e6)
        m["ratio_db"] = np.full((E, 2), 30.0)
        use, st, su = noise_model(m, self.PARAMS, 960e3, 3168)
        planes = reflection_planes({"known_planes": {"y": [-8.613335, 9.571564, 10.337294], "z": [-0.030794]}}) if variant == "AB" else None
        cand = candidates(m, o, use, timing, st, su, 1.5, self.WMAP, planes)
        return run_tracker_v2(cand, self.PARAMS)["xy_ekf"], ue

    def test_ue_behind_first_panel_resolved(self):
        xs = np.linspace(2.0, 5.0, 31)  # behind O-RU 0 (x = 12), in front of O-RU 1 (x = -28)
        for timing in ("tdoa", "aoa"):
            est, ue = self._track(xs, [[12.0, 6.5, 5.0], [-28.0, -7.0, 5.0]], timing, noise=1e-4)
            err = np.linalg.norm(est - ue[:, :2], axis=-1)
            self.assertLess(np.nanmax(err[5:]), 0.3, f"{timing}: {np.nanmax(err[5:]):.2f} m (mirror of x=3 across x=12 is x=21)")

    def test_ue_behind_second_panel_resolved(self):
        xs = np.linspace(-36.0, -33.0, 31)  # behind O-RU 1 (x = -28), behind O-RU 0 (x = 12)
        est, ue = self._track(xs, [[12.0, 6.5, 5.0], [-28.0, -7.0, 5.0]], "tdoa", noise=1e-4)
        err = np.linalg.norm(est - ue[:, :2], axis=-1)
        self.assertLess(np.nanmax(err[5:]), 0.3, f"{np.nanmax(err[5:]):.2f} m")

    def test_reflection_hypothesis_used_when_dominant_path_is_reflected(self):
        xs = np.linspace(14.0, 17.0, 31)
        est, ue = self._track(xs, [[12.0, 6.5, 5.0], [-28.0, -7.0, 5.0]], "tdoa", noise=1e-4, variant="AB", reflect_c=1)
        err = np.linalg.norm(est - ue[:, :2], axis=-1)
        self.assertLess(np.nanmax(err[5:]), 0.3, f"{np.nanmax(err[5:]):.2f} m")


@unittest.skipUnless(_cuda(), "needs the GPU")
class JobTest(unittest.TestCase):
    def test_model_b_per_path_matches_paper1(self):
        if not (JOB / "m3_timeline.npz").exists():
            self.skipTest("paper-1 caches absent")
        from sim.positioning.blockage import path_losses_db
        from sim.scenes.config import load_yaml
        from sim.scenes.traffic import prepare_scenario

        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        geom = {k: v for k, v in np.load(JOB / "comm_geometry.npz").items()}
        sc = prepare_scenario(raw, seed=1001, mount="lamppost", density="low", duration_s=60.0, dt_s=0.1)
        L = path_losses_db(geom, sc, WL)["loss_db"]
        los = np.where(geom["path_class"] == 0, L, -1.0).max(-1)
        ref = np.load(JOB / "m3_timeline.npz")["los_loss_db"][::10][: los.shape[0]]
        self.assertLess(np.abs(np.nan_to_num(los, posinf=1e9) - np.nan_to_num(ref, posinf=1e9)).max(), 1e-5)

    def test_peb_job_deterministic_and_matches_reference(self):
        if not (ROOT / "results" / "P2" / "cache" / "lamppost" / "low" / "seed_1001" / "p2_paths.npz").exists():
            self.skipTest("p2 cache absent")
        import p2_peb
        from sim.scenes.config import load_yaml

        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        cfg = p2_peb.load_cfg()
        a = p2_peb.build_job((1001, "lamppost", "low"), raw, cfg)
        b = p2_peb.build_job((1001, "lamppost", "low"), raw, cfg)
        self.assertTrue(np.array_equal(a["peb"], b["peb"]), "PEB not bitwise reproducible in one process")
        worst = p2_peb.reference_check(a, cfg, np.random.default_rng(3), n=12)
        self.assertLess(worst, 1e-3)


    def test_estimator_measurement_deterministic(self):
        if not (ROOT / "results" / "P2" / "cache" / "lamppost" / "low" / "seed_1001" / "p2_paths.npz").exists():
            self.skipTest("p2 cache absent")
        import p2_estimate as E
        from sim.scenes.config import load_yaml

        raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
        p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
        job = (1001, "lamppost", "low")
        jp = E.job_paths(job, raw, p2cfg)
        cfg = E.configs()["bw100_tdoa_s1_p2_b"]
        a = E.measure_job(job, jp, cfg, p2cfg)
        b = E.measure_job(job, jp, cfg, p2cfg)
        for k in a:
            self.assertTrue(np.array_equal(a[k], b[k]), k)


if __name__ == "__main__":
    unittest.main()
