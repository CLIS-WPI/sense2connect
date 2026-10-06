"""TVT T1 tests: uncertain-map bound and physical array model. Run: python scripts/test_tvt_t1.py"""

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


def _toy_scene():
    """Two O-RUs, UE between facades y = -8.613 / 9.572 and ground; single and double bounces on known planes."""
    from sim.positioning.geometry import path_geometry

    known = {"x": [], "y": [-8.613335, 9.571564], "z": [-0.030794]}
    orus = [np.array([12.0, 6.5, 5.0]), np.array([-28.0, -7.0, 5.0])]
    ue = np.array([-5.3, -6.2, 1.5])
    planes = [(np.array([0.0, 1.0, 0.0]), np.array([0.0, -8.613335, 0.0])), (np.array([0.0, -1.0, 0.0]), np.array([0.0, 9.571564, 0.0])),
              (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -0.030794]))]
    combos = [(), (0,), (1,), (2,), (0, 1), (1, 0)]
    N = np.zeros((2, len(combos), 2, 3))
    A = np.zeros((2, len(combos), 2, 3))
    V = np.zeros((2, len(combos), 2), dtype=bool)
    for c in range(2):
        for i, cb in enumerate(combos):
            for j, k in enumerate(cb):
                N[c, i, j], A[c, i, j] = planes[k]
                V[c, i, j] = True
    return known, orus, ue, N, A, V, path_geometry


class MapBoundTest(unittest.TestCase):
    def test_offset_derivatives_vs_finite_differences(self):
        from sim.positioning.geometry import path_geometry
        from sim.tvt.geometry_map import offset_derivatives

        known, orus, ue, N, A, V, _ = _toy_scene()
        h = 1e-5
        for c in range(2):
            oru = np.broadcast_to(orus[c], (N.shape[1], 3))
            uev = np.broadcast_to(ue, (N.shape[1], 3))
            d = offset_derivatives(oru, uev, N[c], A[c], V[c])
            for slot in range(2):
                Ap, Am = A[c].copy(), A[c].copy()
                Ap[:, slot] += h * N[c][:, slot]
                Am[:, slot] -= h * N[c][:, slot]
                gp, gm = path_geometry(oru, uev, N[c], Ap, V[c]), path_geometry(oru, uev, N[c], Am, V[c])
                for q, dq in (("tau", "dtau"), ("az", "daz"), ("el", "del")):
                    diff = gp[q] - gm[q]
                    if q == "az":
                        diff = np.angle(np.exp(1j * diff))
                    fd = diff / (2 * h)
                    an = d[dq][:, slot]
                    scale = np.abs(an).max() + (1e-12 if q == "tau" else 1e-6)
                    self.assertLess(np.max(np.abs(fd - an)) / scale, 1e-5, f"{q} slot {slot}")
                    self.assertTrue(np.all(an[~V[c][:, slot]] == 0))

    def test_surface_ids(self):
        from sim.tvt.geometry_map import surface_ids, surfaces

        known, orus, ue, N, A, V, _ = _toy_scene()
        surf = surfaces(known)
        self.assertEqual(len(surf), 2)  # facades only by default, ground known
        sid = surface_ids(N, A, V, surf)
        self.assertEqual(sid[0, 1, 0], 0)
        self.assertEqual(sid[0, 2, 0], 1)
        self.assertEqual(sid[0, 3, 0], -1)  # ground
        self.assertEqual(sid[0, 4, 1], 1)
        self.assertEqual(sid[0, 0, 0], -1)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_gram_general_equals_frozen_gram(self):
        import torch

        from sim.positioning.array import element_positions
        from sim.positioning.fim import gram_torch
        from sim.tvt.fim import gram_general

        rng = np.random.default_rng(3)
        P = 4
        f = (np.arange(64) - 31.5) * 120e3 * 20
        beta = torch.as_tensor(rng.standard_normal((2, P)) + 1j * rng.standard_normal((2, P)), device="cuda")
        tau = torch.as_tensor(rng.uniform(20, 200, (2, P)), device="cuda")
        az = torch.as_tensor(rng.uniform(-2, 2, (2, P)), device="cuda")
        el = torch.as_tensor(rng.uniform(-0.5, 0.3, (2, P)), device="cuda")
        r = element_positions(WL)
        a = gram_torch(beta, tau, az, el, f, r, WL)
        b = gram_general(beta, tau, az, el, f, r, WL, pattern="iso", kinds=("phase",))
        self.assertLess(float((a - b).abs().max() / a.abs().max()), 1e-12)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_gram_general_vs_explicit_synthesis(self):
        """Pattern + phase/gain/position errors: Gram vs J = 2 Re(D^H D) with D by complex-step-free central differences of mu."""
        import torch

        from sim.positioning.array import element_positions
        from sim.tvt.array_model import pattern_amplitude
        from sim.tvt.fim import gram_general

        rng = np.random.default_rng(4)
        P, Nf = 3, 24
        f = (np.arange(Nf) - (Nf - 1) / 2) * 120e3 * 40
        r = element_positions(WL)
        M = r.shape[0]
        beta = rng.standard_normal(P) + 1j * rng.standard_normal(P)
        tau = rng.uniform(20, 200, P)
        az = rng.uniform(-1.0, 1.0, P)
        el = rng.uniform(-0.4, 0.2, P)
        kinds = ("phase", "gain", "pos")

        def mu(x):
            t, a_, e_ = x[0:P], x[P:2 * P], x[2 * P:3 * P]
            b = x[3 * P:4 * P] + 1j * x[4 * P:5 * P]
            o = 5 * P
            psi, g, dr = x[o:o + M], x[o + M:o + 2 * M], x[o + 2 * M:o + 5 * M].reshape(M, 3)
            u = np.stack([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)], -1)
            s = pattern_amplitude(a_, e_, "tr38901")[:, None] * np.exp(1j * 2 * np.pi / WL * (u @ (r + dr).T)) * np.exp(g + 1j * psi)[None]
            gf = np.exp(-2j * np.pi * f[None, :] * t[:, None] * 1e-9)
            return np.einsum("p,pn,pm->nm", b, gf, s).reshape(-1)

        x0 = np.concatenate([tau, az, el, beta.real, beta.imag, np.zeros(5 * M)])
        D = np.zeros((Nf * M, x0.size), complex)
        for i in range(x0.size):
            h = 1e-6 * max(1.0, abs(x0[i])) if i >= 5 * P + 2 * M or i < 3 * P else 1e-6
            if i >= 5 * P + 2 * M:
                h = 1e-9
            e = np.zeros_like(x0)
            e[i] = h
            D[:, i] = (mu(x0 + e) - mu(x0 - e)) / (2 * h)
        Jn = 2 * np.real(D.conj().T @ D)
        # reorder to gram_general's per-path interleaved order
        order = [j * P + p for p in range(P) for j in range(5)] + list(range(5 * P, 5 * P + 5 * M))
        # pos columns in gram_general: kind-major (x for all m, y for all m, z for all m)
        pos = [5 * P + 2 * M + 3 * m + i for i in range(3) for m in range(M)]
        order = order[:5 * P + 2 * M] + pos
        Jn = Jn[np.ix_(order, order)]
        Jg = gram_general(torch.as_tensor(beta, device="cuda"), torch.as_tensor(tau, device="cuda"), torch.as_tensor(az, device="cuda"),
                          torch.as_tensor(el, device="cuda"), f, r, WL, pattern="tr38901", kinds=kinds).cpu().numpy()
        d = np.sqrt(np.outer(np.diag(Jn), np.diag(Jn)))
        self.assertLess(np.max(np.abs(Jg - Jn) / d), 1e-5)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_map_peb_vs_bruteforce(self):
        """Toy scene: composed pipeline (gram_general, efim_geometric, H with offset derivatives, priors) vs ONE numerical
        FIM over all unknowns (x, y, delta_c, b, d_s, complex gains, element phases) from central differences of mu."""
        import torch

        from sim.positioning.array import element_positions
        from sim.positioning.fim import peb_from_theta, theta_information
        from sim.positioning.geometry import path_geometry
        from sim.tvt.fim import efim_geometric, element_prior, gram_general
        from sim.tvt.geometry_map import offset_derivatives, surface_ids, surfaces

        known, orus, ue, N, A, V, _ = _toy_scene()
        surf = surfaces(known)
        S = len(surf)
        r = element_positions(WL)
        M = r.shape[0]
        Nf = 16
        f = (np.arange(Nf) - (Nf - 1) / 2) * 120e3 * 100
        rng = np.random.default_rng(5)
        Pn = N.shape[1]
        beta = [(rng.standard_normal(Pn) + 1j * rng.standard_normal(Pn)) * 3 for _ in range(2)]
        sig_sync, sig_phi, sig_map = 1.0, math.radians(2.0), 0.3

        def mu_all(th):
            x, y, d0, d1, b = th[:5]
            ds = th[5:5 + S]
            out = []
            off = 5 + S
            for c in range(2):
                bc = th[off:off + Pn] + 1j * th[off + Pn:off + 2 * Pn]
                psi = th[off + 2 * Pn:off + 2 * Pn + M]
                off += 2 * Pn + M
                Ac = A[c].copy()
                sid = surface_ids(N[c], A[c], V[c], surf)
                for slot in range(2):
                    for s_ in range(S):
                        Ac[:, slot] += np.where(sid[:, slot] == s_, ds[s_], 0.0)[:, None] * N[c][:, slot]
                p = np.array([x, y, ue[2]])
                g = path_geometry(np.broadcast_to(orus[c], (Pn, 3)), np.broadcast_to(p, (Pn, 3)), N[c], Ac, V[c])
                tau = g["tau"] * 1e9 + (d0 if c == 0 else d1) + b
                s = np.exp(1j * 2 * np.pi / WL * (g["u"] @ r.T)) * np.exp(1j * psi)[None]
                gf = np.exp(-2j * np.pi * f[None, :] * tau[:, None] * 1e-9)
                out.append(np.einsum("p,pn,pm->nm", bc, gf, s).reshape(-1))
            return np.concatenate(out)

        th0 = np.concatenate([[ue[0], ue[1], 0, 0, 0], np.zeros(S)] + [np.concatenate([beta[c].real, beta[c].imag, np.zeros(M)]) for c in range(2)])
        D = np.zeros((2 * Nf * M, th0.size), complex)
        for i in range(th0.size):
            h = 1e-6
            e = np.zeros_like(th0)
            e[i] = h
            D[:, i] = (mu_all(th0 + e) - mu_all(th0 - e)) / (2 * h)
        J = 2 * np.real(D.conj().T @ D)
        pr = np.zeros(th0.size)
        pr[2:4] = 1 / sig_sync ** 2
        pr[5:5 + S] = 1 / sig_map ** 2
        off = 5 + S
        for c in range(2):
            pr[off + 2 * Pn: off + 2 * Pn + M] = 1 / sig_phi ** 2
            off += 2 * Pn + M
        J = J + np.diag(pr)
        ref = math.sqrt(np.trace(np.linalg.inv(J)[:2, :2]))
        # composed pipeline
        Hs, Ks = [], []
        for c in range(2):
            oru = np.broadcast_to(orus[c], (Pn, 3))
            uev = np.broadcast_to(ue, (Pn, 3))
            g = path_geometry(oru, uev, N[c], A[c], V[c])
            od = offset_derivatives(oru, uev, N[c], A[c], V[c])
            sid = surface_ids(N[c], A[c], V[c], surf)
            H = np.zeros((3 * Pn, 5 + S))
            for p in range(Pn):
                H[3 * p, 0:2] = g["dtau"][p] * 1e9
                H[3 * p, 2 + c] = 1.0
                H[3 * p, 4] = 1.0
                H[3 * p + 1, 0:2] = g["daz"][p]
                H[3 * p + 2, 0:2] = g["del"][p]
                for slot in range(2):
                    if sid[p, slot] >= 0:
                        H[3 * p, 5 + sid[p, slot]] += od["dtau"][p, slot] * 1e9
                        H[3 * p + 1, 5 + sid[p, slot]] += od["daz"][p, slot]
                        H[3 * p + 2, 5 + sid[p, slot]] += od["del"][p, slot]
            Jg = gram_general(torch.as_tensor(beta[c], device="cuda"), torch.as_tensor(g["tau"] * 1e9, device="cuda"), torch.as_tensor(g["az"], device="cuda"),
                              torch.as_tensor(g["el"], device="cuda"), f, r, WL, pattern="iso", kinds=("phase",))
            free = torch.zeros(3 * Pn, dtype=torch.bool, device="cuda")
            Ks.append(efim_geometric(Jg, Pn, element_prior(("phase",), M, sig_phi, 0, 0), free))
            Hs.append(H)
        K = torch.stack(Ks)[None]
        Ht = torch.as_tensor(np.stack(Hs), device="cuda")[None]
        J0 = theta_information(K, Ht)
        prior = torch.zeros((1, 5 + S, 5 + S), dtype=torch.float64, device="cuda")
        prior[0, 2, 2] = prior[0, 3, 3] = 1 / sig_sync ** 2
        for s_ in range(S):
            prior[0, 5 + s_, 5 + s_] = 1 / sig_map ** 2
        present = torch.ones((1, 5 + S), dtype=torch.bool, device="cuda")
        got = float(peb_from_theta(J0, prior, present)[0])
        self.assertLess(abs(got - ref) / ref, 2e-3, f"{got} vs {ref}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
