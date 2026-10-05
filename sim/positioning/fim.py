"""Fisher information and position error bound (PEB), batched on the GPU (torch) with NumPy references.

Signal at O-RU c (uplink SRS, one OFDM symbol, N active subcarriers, 64
elements), normalised by the noise standard deviation per subcarrier and
element:
    mu[n, m] = sum_p beta_p exp(-j 2 pi f_n tau_p) s_m(az_p, el_p),
    f_n = (n - (N - 1) / 2) df  (baseband offsets; the carrier phase is part of
    the unknown complex gain beta_p, i.e. no carrier-phase positioning).
Slepian-Bangs for a deterministic signal in white circular Gaussian noise:
J = 2 Re(D^H D), D = d mu / d eta, eta_p = (tau_p, az_p, el_p, Re beta_p,
Im beta_p). D^H D factorises into frequency sums F_k(dtau) =
sum_n f_n^k exp(-j 2 pi f_n dtau) and antenna sums over steering vectors and
their angle derivatives, so the per-path Gram matrix is exact without
synthesising the channel.

Position domain (UE height known): unknowns theta = (x, y | nuisance). The
complex gains are always nuisance (eliminated first by a Schur complement:
K = geometric-parameter EFIM per O-RU). Per variant, the geometric
parameters of some paths are free nuisances (Schur out of K): LoS-only
information -> all NLoS paths; blocked-LoS variant "biased" -> the LoS path of
a blocked O-RU; AoA-only timing -> every delay. The remaining ("informative")
parameters map to theta through
    tau_p = tau_p(x, y) + delta_c + b,  az_p = az_p(x, y) + beta_c,az,
    el_p = el_p(x, y) + beta_c,el
with delta_c the O-RU time offset (Gaussian prior, sigma_sync) and b the UE
clock bias (case "tdoa": free; "toa": known; "aoa": absent). Array
calibration: the 64 per-element phase errors psi_m of each O-RU enter the
signal exactly, mu[n, m] -> mu[n, m] exp(j psi_m), as nuisance parameters with
the Gaussian prior N(0, sigma_phi^2) (Bayesian EFIM; d mu / d psi_m =
j mu[n, m] at psi = 0); each path's apparent angle bias is thus path dependent. EFIM for
(x, y) by Schur complement; PEB = sqrt(trace(EFIM^-1)) [m] (inf if singular).
Units inside: delays in ns, angles in rad, positions in m.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TWO_PI = 2.0 * math.pi
NS = 1e-9


# ----------------------------------------------------------------------------- NumPy reference
def gram_numpy(beta: np.ndarray, tau_ns: np.ndarray, az: np.ndarray, el: np.ndarray, f_hz: np.ndarray, r: np.ndarray, wavelength_m: float) -> np.ndarray:
    """J [5P + M, 5P + M] for one O-RU by explicit synthesis of D (reference; small sizes).

    Order: per path (tau [ns], az, el, Re beta, Im beta), then the M element phases psi_m.
    """
    from sim.positioning.array import direction_derivatives, steering, unit_direction

    P = beta.shape[0]
    M = r.shape[0]
    k = TWO_PI / wavelength_m
    u = unit_direction(az, el)
    du_az, du_el = direction_derivatives(az, el)
    s = steering(u, r, wavelength_m)  # [P, M]
    g = np.exp(-1j * TWO_PI * f_hz[None, :] * tau_ns[:, None] * NS)  # [P, N]
    mu = sum(beta[p] * np.outer(g[p], s[p]) for p in range(P))
    cols = []
    for p in range(P):
        sp, gp = s[p], g[p]
        dsa = 1j * k * (du_az[p] @ r.T) * sp
        dse = 1j * k * (du_el[p] @ r.T) * sp
        cols += [beta[p] * np.outer(-1j * TWO_PI * f_hz * NS * gp, sp), beta[p] * np.outer(gp, dsa), beta[p] * np.outer(gp, dse),
                 np.outer(gp, sp), 1j * np.outer(gp, sp)]
    for m in range(M):
        e = np.zeros_like(mu)
        e[:, m] = 1j * mu[:, m]
        cols.append(e)
    D = np.stack([c.reshape(-1) for c in cols], axis=1)
    return 2.0 * np.real(D.conj().T @ D)


# ----------------------------------------------------------------------------- torch kernels
def _freq_sums(tau_ns, f_hz, chunk: int = 1024):
    """F_k[..., p, q] = sum_n f_n^k exp(-j 2 pi f_n (tau_q - tau_p)), k = 0, 1, 2 (f in GHz to keep scales O(1))."""
    import torch

    dt = (tau_ns[..., None, :] - tau_ns[..., :, None])  # [..., p, q] ns
    fg = torch.as_tensor(f_hz, device=tau_ns.device, dtype=torch.float64) * 1e-9  # GHz
    out = [torch.zeros(dt.shape, dtype=torch.complex128, device=dt.device) for _ in range(3)]
    for lo in range(0, fg.shape[0], chunk):
        f = fg[lo:lo + chunk]
        e = torch.exp(-1j * TWO_PI * dt[..., None] * f)  # [..., p, q, n]
        out[0] += e.sum(-1)
        out[1] += (e * f).sum(-1)
        out[2] += (e * f * f).sum(-1)
    return out  # in units with f in GHz and tau in ns


def gram_torch(beta, tau_ns, az, el, f_hz: np.ndarray, r: np.ndarray, wavelength_m: float):
    """J [..., 5P, 5P] per O-RU, batched. beta complex [..., P] (zero for absent paths); angles [rad]; tau [ns].

    Parameter order per path: (tau [ns], az, el, Re beta, Im beta).
    """
    import torch

    dev = tau_ns.device
    k = TWO_PI / wavelength_m
    rr = torch.as_tensor(r, device=dev, dtype=torch.float64)
    ca, sa, ce, se = torch.cos(az), torch.sin(az), torch.cos(el), torch.sin(el)
    u = torch.stack([ce * ca, ce * sa, se], -1)
    du_az = torch.stack([-ce * sa, ce * ca, torch.zeros_like(az)], -1)
    du_el = torch.stack([-se * ca, -se * sa, ce], -1)
    s = torch.exp(1j * k * (u @ rr.T))  # [..., P, M]
    wa = 1j * k * (du_az @ rr.T)
    we = 1j * k * (du_el @ rr.T)
    w = [torch.ones_like(wa), wa, we]
    A = {}
    for i in range(3):
        for j in range(3):
            A[i, j] = (torch.conj(w[i] * s)[..., :, None, :] * (w[j] * s)[..., None, :, :]).sum(-1)  # [..., p, q]
    F = _freq_sums(tau_ns, f_hz)
    P = beta.shape[-1]
    # params: 0 tau (coef beta * (-j 2 pi), freq power 1 [GHz -> per ns], ant 0), 1 az, 2 el, 3 Re (coef 1), 4 Im (coef j)
    coef = [beta * (-1j * TWO_PI), beta, beta, torch.ones_like(beta), 1j * torch.ones_like(beta)]
    fpow = [1, 0, 0, 0, 0]
    ant = [0, 1, 2, 0, 0]
    M = rr.shape[0]
    G = torch.zeros(beta.shape[:-1] + (5 * P + M, 5 * P + M), dtype=torch.complex128, device=dev)
    for x in range(5):
        for y in range(5):
            blk = torch.conj(coef[x])[..., :, None] * coef[y][..., None, :] * F[fpow[x] + fpow[y]] * A[ant[x], ant[y]]
            G[..., x:5 * P:5, y:5 * P:5] = blk
    # element phases: <Psi_m, X_p> = -j c_xp w_xpm s_pm sum_q conj(beta_q s_qm) F_kx[q, p];  <Psi_m, Psi_m> = sum_qq' conj(b_q s_qm) b_q' s_q'm F_0[q, q']
    bs = beta[..., :, None] * s  # [..., q, m]
    for x in range(5):
        Fx = F[fpow[x]]  # [..., q, p]
        t = torch.einsum("...qm,...qp->...pm", torch.conj(bs), Fx)  # [..., p, m]
        val = -1j * coef[x][..., :, None] * (w[ant[x]] * s) * t  # [..., p, m]
        G[..., 5 * P:, x:5 * P:5] = val.transpose(-1, -2)
        G[..., x:5 * P:5, 5 * P:] = torch.conj(val)
    pp = torch.einsum("...qm,...qr,...rm->...m", torch.conj(bs), F[0], bs)
    G[..., 5 * P:, 5 * P:] = torch.diag_embed(pp)
    return 2.0 * G.real


def _schur(J, keep, eps: float = 1e-12):
    """EFIM of the parameters with keep == True (bool [..., n]); others eliminated. Returns J_kk - J_kd J_dd^-1 J_dk
    with the dropped block regularised (relative eps) and the kept parameters' rows/cols in their original positions
    (dropped rows/cols zeroed)."""
    import torch

    n = J.shape[-1]
    kf = keep.to(J.dtype)
    df = 1.0 - kf
    d = torch.diagonal(J, dim1=-2, dim2=-1).abs()
    scale = torch.where(d > 0, d, torch.ones_like(d)).sqrt()
    Js = J / (scale[..., :, None] * scale[..., None, :])
    eye = torch.eye(n, dtype=J.dtype, device=J.device)
    Jdd = Js * (df[..., :, None] * df[..., None, :]) + eye * (kf[..., None, :] + eps)
    Jdk = Js * (df[..., :, None] * kf[..., None, :])
    sol = torch.linalg.solve(Jdd, Jdk)
    out = Js * (kf[..., :, None] * kf[..., None, :]) - Jdk.transpose(-1, -2) @ sol
    return out * (scale[..., :, None] * scale[..., None, :])


def geometric_efim(Jfull, n_paths: int):
    """Eliminate the complex gains: K [..., 3P + M, 3P + M] over (tau, az, el) per path, then the M element phases."""
    import torch

    n = Jfull.shape[-1]
    keep = torch.ones(n, dtype=torch.bool, device=Jfull.device)
    for i in range(n_paths):
        keep[5 * i + 3: 5 * i + 5] = False
    K = _schur(Jfull, keep.expand(Jfull.shape[:-1]))
    idx = keep.nonzero().flatten()
    return K[..., idx, :][..., :, idx]


def calibrated_efim(K, free, sigma_phi_rad: float, n_geo: int):
    """Per O-RU EFIM of the informative geometric parameters [..., 3P, 3P] (free ones zeroed) for a phase-error prior.

    K [..., 3P + M, 3P + M]; free [..., 3P] bool. sigma_phi = 0: phases known.
    """
    import torch

    n = K.shape[-1]
    M = n - n_geo
    if sigma_phi_rad > 0:
        Kp = K.clone()
        idx = torch.arange(n_geo, n, device=K.device)
        Kp[..., idx, idx] = Kp[..., idx, idx] + 1.0 / sigma_phi_rad ** 2
        keep = torch.cat([~free, torch.zeros(free.shape[:-1] + (M,), dtype=torch.bool, device=K.device)], -1)
        return _schur(Kp, keep)[..., :n_geo, :n_geo]
    return _schur(K[..., :n_geo, :n_geo], ~free)


def theta_information(Keff: Any, H: Any) -> Any:
    """J0 [..., T, T] = sum_c H_c^T Keff_c H_c (Keff from calibrated_efim, free comps already eliminated)."""
    import torch

    return torch.einsum("...cia,...cij,...cjb->...ab", H, Keff, H)


def peb_from_theta(J0: Any, prior: Any, present: Any) -> Any:
    """PEB [...] from J0 + prior precision; present [..., T] bool (absent nuisances are known/fixed)."""
    import torch

    J = J0 + prior
    T = J.shape[-1]
    pres = present.clone()
    pres[..., :2] = True
    keep_pos = torch.zeros(T, dtype=torch.bool, device=J.device)
    keep_pos[:2] = True
    pf = pres.to(J.dtype)
    J = J * (pf[..., :, None] * pf[..., None, :]) + torch.diag_embed(1.0 - pf)
    Jp = _schur(J, keep_pos.expand(J.shape[:-1]) & pres)[..., :2, :2]
    det = Jp[..., 0, 0] * Jp[..., 1, 1] - Jp[..., 0, 1] * Jp[..., 1, 0]
    tr = (Jp[..., 0, 0] + Jp[..., 1, 1])
    ok = (det > 1e-12 * tr.clamp_min(1e-300) ** 2) & (tr > 0)
    return torch.where(ok, torch.sqrt(tr / torch.where(ok, det, torch.ones_like(det))), torch.full_like(det, math.inf))


def peb_numpy(Jfull: np.ndarray, n_paths: int, H: np.ndarray, free: np.ndarray, sigma_phi_rad: float, prior: np.ndarray, present: np.ndarray) -> float:
    """Independent reference for one (UE, epoch): ONE information matrix over ALL unknowns and a single
    eigenvalue-based pseudo-inverse (no sequential Schur complements).

    Unknowns: theta (x, y, delta_0, delta_1, b; absent entries are known) plus, per O-RU, the free geometric
    components, the complex gains (Re, Im per path) and the M element phases (prior N(0, sigma_phi^2); known if
    sigma_phi = 0). Jfull [C, 5P + M, 5P + M] in gram_numpy order; H [C, 3P, T]; free [C, 3P].
    PEB^2 = trace of the (x, y) block of the pseudo-inverse of the equilibrated matrix.
    """
    C, n, _ = Jfull.shape
    M = n - 5 * n_paths
    th = np.flatnonzero(present)
    maps = []
    width = len(th)
    for c in range(C):
        nfree = int(free[c].sum())
        nphi = M if sigma_phi_rad > 0 else 0
        Tc = np.zeros((n, 2 * n_paths + nfree + nphi))
        j = 0
        Hc = np.zeros((n, len(th)))
        for p in range(n_paths):
            for i in range(3):
                if free[c][3 * p + i]:
                    Tc[5 * p + i, j] = 1.0
                    j += 1
                else:
                    Hc[5 * p + i] = H[c][3 * p + i][th]
        for p in range(n_paths):
            Tc[5 * p + 3, j] = 1.0
            Tc[5 * p + 4, j + 1] = 1.0
            j += 2
        for m in range(nphi):
            Tc[5 * n_paths + m, j] = 1.0
            j += 1
        maps.append((Hc, Tc, width, nphi))
        width += Tc.shape[1]
    J = np.zeros((width, width))
    J[: len(th), : len(th)] += prior[np.ix_(th, th)]
    for c, (Hc, Tc, off, nphi) in enumerate(maps):
        T = np.zeros((n, width))
        T[:, : len(th)] = Hc
        T[:, off: off + Tc.shape[1]] = Tc
        J += T.T @ Jfull[c] @ T
        if nphi:
            e = off + Tc.shape[1]
            J[e - nphi: e, e - nphi: e] += np.eye(nphi) / sigma_phi_rad ** 2
    d = np.sqrt(np.where(np.diag(J) > 0, np.diag(J), 1.0))
    w, V = np.linalg.eigh(J / np.outer(d, d))
    keep = w > w.max() * 1e-14
    inv = (V[:, keep] / w[keep]) @ V[:, keep].T / np.outer(d, d)
    tr = float(inv[0, 0] + inv[1, 1])
    # unidentifiable position (zero eigenvalue in the x, y directions) -> inf
    null = V[:, ~keep]
    if null.size and np.abs(null[:2]).max() > 1e-6:
        return math.inf
    return math.sqrt(tr) if tr > 0 else math.inf
