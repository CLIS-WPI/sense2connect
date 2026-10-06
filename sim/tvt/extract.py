"""Sparse multipath extraction per O-RU and epoch (TVT T2), batched on the GPU.

Input: SRS snapshot Y [B, N, M] normalised by the noise standard deviation (sim/positioning/
estimator.synth_torch), subcarrier offsets f_n, the 8x8 UPA of sim/positioning/array.py.
Atom of a path with delay tau and spatial frequencies (u_y, u_z):
    x[n, m] = exp(-j 2 pi f_n tau) exp(j k (u_y y_m + u_z z_m)),  ||x||^2 = N M.
Algorithm (orthogonal-matching-pursuit style with continuous refinement):
1. Beam-delay grid: 2-D DFT over the array (Q x Q, oversampled) and IFFT over the
   subcarriers (L bins, zero-padded) of the residual; the strongest cell is the candidate.
2. Local refinement: coordinate-wise parabolic search on the matched-filter power
   |x(theta)^H r|^2 over (tau, u_y, u_z) with shrinking steps (continuous, off-grid).
3. Gain by least squares beta = x^H r / (N M); subtract; repeat.
4. Stop when the candidate's matched-filter power is below the detection threshold
   eta = N M T_noise (noise-only cell power ~ N M Exp(1); T_noise from the per-snapshot
   false-alarm probability over all grid cells) or below the strongest component by more
   than ``dyn_range_db`` (model-mismatch residue of strong paths, e.g. element phase errors),
   or after K_max components.
5. Two cyclic refinement sweeps (SAGE-like): each component is re-estimated on the residual
   plus its own contribution.
Covariance of each component (tau [ns], u_y, u_z) from its local Fisher information (single
path, other paths ignored; centred subcarriers and centred array -> diagonal after the gain
is eliminated): var(tau) = 1 / (2 |beta|^2 M sum_n (2 pi f_n)^2),
var(u_y) = 1 / (2 |beta|^2 N k^2 sum_m y_m^2), var(u_z) likewise with z_m.
The delay is returned modulo 1/df; (u_y, u_z) do not tell the front/back hemisphere apart.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


REFINE_OFFSET = 2.5  # noise-only Monte Carlo (N = 792 and 3168 subcarriers, 64 elements): the continuously refined
#                      maximum exceeds the grid-cell model ln(cells / pfa) by 1.4-2.6; 2.5 keeps the false-alarm rate <= pfa


def detection_factor(pfa: float, n_cells: int) -> float:
    """T_noise with P(max of the refined noise-only matched-filter power / (N M) > T) ~ pfa."""
    return math.log(n_cells / pfa) + REFINE_OFFSET


def _corr(R, f, yz, k, tau_ns, uy, uz):
    """x(theta)^H R for a batch: R [B, N, M]; tau_ns, uy, uz [B]. Returns complex [B]."""
    import torch

    g = torch.exp(1j * TWO_PI * f[None, :] * tau_ns[:, None] * 1e-9)  # conj of exp(-j 2 pi f tau)
    s = torch.exp(-1j * k * (uy[:, None] * yz[None, :, 0] + uz[:, None] * yz[None, :, 1]))  # conj steering
    return torch.einsum("bn,bnm,bm->b", g, R, s)


def _atom(f, yz, k, tau_ns, uy, uz):
    import torch

    g = torch.exp(-1j * TWO_PI * f[None, :] * tau_ns[:, None] * 1e-9)
    s = torch.exp(1j * k * (uy[:, None] * yz[None, :, 0] + uz[:, None] * yz[None, :, 1]))
    return g[:, :, None] * s[:, None, :]


def _refine(R, f, yz, k, tau, uy, uz, steps, passes: int = 4):
    """Coordinate-wise parabolic refinement of |x^H R|^2 around (tau, uy, uz)."""
    import torch

    th = [tau.clone(), uy.clone(), uz.clone()]
    st = list(steps)
    for _ in range(passes):
        for i in range(3):
            vals = []
            for d in (-1.0, 0.0, 1.0):
                t = [x.clone() for x in th]
                t[i] = t[i] + d * st[i]
                vals.append(_corr(R, f, yz, k, *t).abs() ** 2)
            vm, v0, vp = vals
            den = vm - 2 * v0 + vp
            off = torch.where(den.abs() > 1e-30, 0.5 * (vm - vp) / torch.where(den.abs() > 1e-30, den, torch.ones_like(den)), torch.zeros_like(den))
            off = off.clamp(-1.0, 1.0)
            off = torch.where(den < 0, off, torch.zeros_like(off))  # only at a maximum
            th[i] = th[i] + off * st[i]
            st[i] = st[i] * 0.5
    return th


def extract(Y, f_hz: np.ndarray, wavelength_m: float, *, k_max: int = 8, pfa: float = 1e-2, dyn_range_db: float = 25.0, q: int = 16,
            delay_pad: int = 2, sweeps: int = 2) -> dict[str, np.ndarray]:
    """Components per snapshot, padded to k_max: tau_ns, uy, uz, beta (complex), var_tau, var_uy, var_uz, valid [B, K]."""
    import torch

    from sim.positioning.array import element_positions

    dev = Y.device
    B, N, M = Y.shape
    r = element_positions(wavelength_m)
    yz = torch.as_tensor(r[:, 1:], device=dev, dtype=torch.float64)
    k = TWO_PI / wavelength_m
    f = torch.as_tensor(f_hz, device=dev, dtype=torch.float64)
    df = float(f_hz[1] - f_hz[0])
    L = delay_pad * (1 << int(math.ceil(math.log2(N))))
    R = Y.to(torch.complex128).clone()
    n_cells = L * q * q
    eta = N * M * detection_factor(pfa, n_cells)
    out = {n: torch.zeros((B, k_max), dtype=torch.float64, device=dev) for n in ("tau_ns", "uy", "uz", "var_tau", "var_uy", "var_uz")}
    beta = torch.zeros((B, k_max), dtype=torch.complex128, device=dev)
    valid = torch.zeros((B, k_max), dtype=torch.bool, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    p_first = torch.zeros(B, dtype=torch.float64, device=dev)
    # grid -> parameter maps
    # element order: y index outer (8), z index inner (8), z decreasing (sim.positioning.array)
    dy = float(r[8, 1] - r[0, 1])  # lambda / 2
    for it in range(k_max):
        Rg = R.reshape(B, N, 8, 8)
        # beamspace: sum_m conj(s_m(u)) R[., m]: with y_i = i dy + y0 and z_j = -(j dz) + z0 the 2-D DFT gives u_y = 2 a / Q, u_z = -2 b / Q (wrapped)
        F2 = torch.fft.fft2(Rg, s=(q, q))  # [B, N, q, q]
        # delay: sum_n exp(+j 2 pi f_n tau) -> IFFT over n
        cube = torch.fft.ifft(F2, n=L, dim=1) * L  # [B, L, q, q]
        pw = cube.abs() ** 2
        flat = pw.reshape(B, -1)
        pk, idx = flat.max(-1)
        li = idx // (q * q)
        rem = idx % (q * q)
        ai, bi = rem // q, rem % q
        tau0 = (li.double() / L) / df * 1e9  # ns, modulo 1/df (relative to f offsets centred: phase constant absorbed in beta)
        uy0 = ((ai.double() / q * 2.0 + 1.0) % 2.0) - 1.0
        uz0 = -(((bi.double() / q * 2.0 + 1.0) % 2.0) - 1.0)
        # power of the candidate with the exact atom (grid offset of the element origin is a phase only)
        th = _refine(R, f, yz, k, tau0, uy0, uz0, steps=(1e9 / (L * df), 2.0 / q, 2.0 / q))
        c = _corr(R, f, yz, k, *th)
        p = c.abs() ** 2
        if it == 0:
            p_first = p.clone()
        ok = active & (p > eta) & (p > p_first * 10 ** (-dyn_range_db / 10.0))
        b = c / (N * M)
        atom = _atom(f, yz, k, *th)
        R = R - torch.where(ok[:, None, None], b[:, None, None] * atom, torch.zeros_like(atom))
        out["tau_ns"][:, it] = th[0]
        out["uy"][:, it] = th[1]
        out["uz"][:, it] = th[2]
        beta[:, it] = b
        valid[:, it] = ok
        active = ok
        if not bool(active.any()):
            break
    # cyclic refinement
    for _ in range(sweeps):
        for it in range(k_max):
            v = valid[:, it]
            if not bool(v.any()):
                continue
            th_old = (out["tau_ns"][:, it], out["uy"][:, it], out["uz"][:, it])
            Rk = R + beta[:, it, None, None] * _atom(f, yz, k, *th_old)
            th = _refine(Rk, f, yz, k, *th_old, steps=(0.25e9 / (L * df), 0.5 / q, 0.5 / q), passes=3)
            c = _corr(Rk, f, yz, k, *th)
            b = c / (N * M)
            newR = Rk - b[:, None, None] * _atom(f, yz, k, *th)
            R = torch.where(v[:, None, None], newR, R)
            for name, val in zip(("tau_ns", "uy", "uz"), th):
                out[name][:, it] = torch.where(v, val, out[name][:, it])
            beta[:, it] = torch.where(v, b, beta[:, it])
    # local Fisher covariances
    fz = (TWO_PI * f * 1e-9) ** 2  # per ns^2
    yc = yz[:, 0] - yz[:, 0].mean()
    zc = yz[:, 1] - yz[:, 1].mean()
    a2 = beta.abs() ** 2
    out["var_tau"] = 1.0 / torch.clamp(2.0 * a2 * M * fz.sum(), min=1e-300)
    out["var_uy"] = 1.0 / torch.clamp(2.0 * a2 * N * k ** 2 * (yc ** 2).sum(), min=1e-300)
    out["var_uz"] = 1.0 / torch.clamp(2.0 * a2 * N * k ** 2 * (zc ** 2).sum(), min=1e-300)
    res = {kk: v.cpu().numpy() for kk, v in out.items()}
    res["beta"] = beta.cpu().numpy()
    res["valid"] = valid.cpu().numpy()
    res["residual_power"] = (R.abs() ** 2).sum((1, 2)).cpu().numpy() / (N * M)
    del dy
    return res
