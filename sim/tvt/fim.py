"""Fisher information with the physical array model and an uncertain map (TVT T1), torch on the GPU.

Generalises sim/positioning/fim.gram_torch (frozen, kept as the reference): the received
signal of one O-RU, normalised by the noise standard deviation,
    mu[n, m] = sum_p beta_p exp(-j 2 pi f_n tau_p) a_m(u_p),
with a_m from sim/tvt/array_model.py (pattern A, element gain/phase/position errors at their
nominal value 0). Every column of D = d mu / d eta has the form
    D_x[n, m] = sum_p c_{x,p} (f_n)^{k_x} exp(-j 2 pi f_n tau_p) V_{x,p,m},
so the Gram matrix factorises exactly into the frequency sums F_k[p, q] of the frozen code
and antenna inner products:
    (D^H D)[x, y] = sum_{p,q} conj(c_xp) c_yq F_{kx+ky}[p, q] (V_xp^H V_yq).
Columns (in this order): per path (tau [ns], az, el, Re beta, Im beta) as in the frozen code,
then the element nuisances of the requested kinds: "phase" psi_m, "gain" g_m (log-amplitude),
"pos" dr_m (x, y, z) [m]. J = 2 Re(D^H D).
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi
ELEMENT_KINDS = ("phase", "gain", "pos")


def gram_general(beta, tau_ns, az, el, f_hz: np.ndarray, r: np.ndarray, wavelength_m: float, *, pattern: str = "iso",
                 kinds: tuple[str, ...] = ("phase",)):
    """J [..., 5P + E, 5P + E]; E = M x (1 per phase/gain kind + 3 for pos)."""
    import torch

    from sim.positioning.fim import _freq_sums
    from sim.tvt.array_model import pattern_amplitude, pattern_log_derivatives

    dev = tau_ns.device
    k = TWO_PI / wavelength_m
    rr = torch.as_tensor(r, device=dev, dtype=torch.float64)
    M = rr.shape[0]
    ca, sa, ce, se = torch.cos(az), torch.sin(az), torch.cos(el), torch.sin(el)
    u = torch.stack([ce * ca, ce * sa, se], -1)
    du_az = torch.stack([-ce * sa, ce * ca, torch.zeros_like(az)], -1)
    du_el = torch.stack([-se * ca, -se * sa, ce], -1)
    A = pattern_amplitude(az, el, pattern, torch)
    dla, dle = pattern_log_derivatives(az, el, pattern, torch)
    s = A[..., None] * torch.exp(1j * k * (u @ rr.T))  # [..., P, M]
    s_az = (1j * k * (du_az @ rr.T) + dla[..., None]) * s
    s_el = (1j * k * (du_el @ rr.T) + dle[..., None]) * s
    P = beta.shape[-1]
    # path columns: (coef [..., P], frequency power, antenna vectors V [..., P, M])
    path_cols = [(beta * (-1j * TWO_PI), 1, s), (beta, 0, s_az), (beta, 0, s_el), (torch.ones_like(beta), 0, s), (1j * torch.ones_like(beta), 0, s)]
    F = _freq_sums(tau_ns, f_hz)
    n_e = sum(1 if kd in ("phase", "gain") else 3 for kd in kinds) * M
    n = 5 * P + n_e
    G = torch.zeros(beta.shape[:-1] + (n, n), dtype=torch.complex128, device=dev)
    # path x path blocks (as the frozen code, with the patterned steering)
    for x, (cx, kx, Vx) in enumerate(path_cols):
        for y, (cy, ky, Vy) in enumerate(path_cols):
            Ax = torch.einsum("...pm,...qm->...pq", torch.conj(Vx), Vy)
            G[..., x:5 * P:5, y:5 * P:5] = torch.conj(cx)[..., :, None] * cy[..., None, :] * F[kx + ky] * Ax
    # element columns: E_{e}[n, m'] = sum_p beta_p g_p(f) w_{e,p} s_{p,m} delta(m', m(e)),
    # with w = j (phase), 1 (gain), j k u_{p,i} (pos coordinate i).
    ws = []
    for kd in kinds:
        if kd == "phase":
            ws.append(1j * torch.ones_like(beta))
        elif kd == "gain":
            ws.append(torch.ones_like(beta))
        elif kd == "pos":
            ws += [1j * k * u[..., 0], 1j * k * u[..., 1], 1j * k * u[..., 2]]
        else:
            raise ValueError(kd)
    bs = beta[..., :, None] * s  # [..., p, m]
    off = 5 * P
    for a_, wa in enumerate(ws):
        # element-element block (same kind pair a_, b_): diagonal over m
        for b_, wb in enumerate(ws):
            val = torch.einsum("...pm,...p,...pq,...q,...qm->...m", torch.conj(bs), torch.conj(wa), F[0], wb, bs)
            G[..., off + a_ * M + torch.arange(M), off + b_ * M + torch.arange(M)] = val
        # path-element blocks
        for x, (cx, kx, Vx) in enumerate(path_cols):
            # <D_x(path p), E_{a,m}> = sum_q conj(c_xp) conj(V_x[p,m]) F_kx[p, q] beta_q w_q s_q[m]
            t = torch.einsum("...pq,...qm->...pm", F[kx], (beta * wa)[..., :, None] * s)
            val = torch.conj(cx)[..., :, None] * torch.conj(Vx) * t  # [..., p, m]
            G[..., x:5 * P:5, off + a_ * M: off + (a_ + 1) * M] = val
            G[..., off + a_ * M: off + (a_ + 1) * M, x:5 * P:5] = torch.conj(val).transpose(-1, -2)
    return 2.0 * G.real


def element_prior(kinds: tuple[str, ...], M: int, sigma_phi_rad: float, sigma_g_db: float, sigma_r_m: float, device: str = "cuda"):
    """Prior precision diagonal [E] for the element nuisances (inf precision for sigma 0 = known)."""
    import torch

    out = []
    for kd in kinds:
        if kd == "phase":
            out += [sigma_phi_rad] * M
        elif kd == "gain":
            out += [sigma_g_db * math.log(10.0) / 20.0] * M
        else:
            out += [sigma_r_m] * (3 * M)
    s = torch.as_tensor(out, device=device, dtype=torch.float64)
    return torch.where(s > 0, 1.0 / torch.clamp(s, min=1e-300) ** 2, torch.full_like(s, math.inf))


def efim_geometric(J, n_paths: int, prior_e, free):
    """EFIM of the informative geometric parameters [..., 3P, 3P] (free ones and gains eliminated, element nuisances
    with prior precision ``prior_e`` [E] (inf = known) eliminated).

    J [..., 5P + E, 5P + E] from gram_general; free [..., 3P] bool.
    """
    import torch

    from sim.positioning.fim import _schur

    n = J.shape[-1]
    P = n_paths
    E = n - 5 * P
    Jp = J.clone()
    known_e = torch.isinf(prior_e)
    if E:
        idx = torch.arange(5 * P, n, device=J.device)
        pr = torch.where(known_e, torch.zeros_like(prior_e), prior_e)
        Jp[..., idx, idx] = Jp[..., idx, idx] + pr
    keep = torch.zeros(J.shape[:-1], dtype=torch.bool, device=J.device)
    geo = torch.zeros(n, dtype=torch.bool, device=J.device)
    for p in range(P):
        geo[5 * p: 5 * p + 3] = True
    keep[..., geo] = ~free
    # known element nuisances: remove rows/cols by making them "kept" then dropping (equivalent to fixing them)
    if E and bool(known_e.any()):
        alive = torch.ones(n, dtype=torch.bool, device=J.device)
        alive[5 * P:][known_e] = False
        idx_alive = alive.nonzero().flatten()
        Jp = Jp[..., idx_alive, :][..., :, idx_alive]
        keep = keep[..., idx_alive]
        geo = geo[idx_alive]
    K = _schur(Jp, keep)
    gidx = geo.nonzero().flatten()
    return K[..., gidx, :][..., :, gidx]
