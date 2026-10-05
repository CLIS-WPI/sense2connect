"""Practical UE positioning from uplink SRS at the two O-RUs (P2-M2).

1. Channel synthesis (GPU, NumPy reference): per O-RU, epoch and UE,
   Y[n, m] = sum_p beta_p exp(-j 2 pi f_n (tau_p + delta_c + b)) s_m(u_p) exp(j psi_m) + w,
   beta_p normalised by the noise standard deviation (w ~ CN(0, 1)).
2. Per-O-RU measurement of the dominant path (GPU, NumPy reference):
   beamspace = 2D DFT over the 8 x 8 UPA (zero-padded to Q x Q), power summed
   over subcarriers -> strongest beam; beamforming + zero-padded IFFT over
   subcarriers -> delay peak with parabolic refinement; angle refinement on the
   array snapshot at that delay (2D DFT zero-padded to Q2 x Q2, parabolic);
   one more delay pass with the refined beam (delay profile with a Hann window
   over the subcarriers, sidelobes about -31 dB). Outputs: delay [ns] (modulo
   1/df), spatial frequencies (u_y, u_z), coherent energy SNR, and the
   peak-to-second-peak ratio of the delay profile (outside +/-2 resolution
   cells) as LoS/NLoS gate statistic.
3. Fusion (NumPy): weighted least squares over the O-RUs that pass the gate,
   unknowns (x, y) [+ UE clock bias b for TDoA], UE height known; residuals
   (u_y, u_z) and, unless AoA only, the delay; weights from CRLB proxies plus
   tuned floors; Gauss-Newton from every single-O-RU AoA ray fix (both
   front/back hypotheses, since the UPA in the y-z plane cannot tell +x from
   -x); lowest cost wins, ties within 1 broken by the distance to the EKF
   prediction. Candidate subsets: all gated O-RUs, and each gated O-RU alone;
   if the joint fit's cost exceeds res_gate (tuned) and the track exists, the
   single-O-RU fix closest to the EKF prediction is used (before the track
   starts the joint fix is always used; a single O-RU is front/back ambiguous).
4. EKF (NumPy): constant-velocity pedestrian model (dt 0.1 s, white
   acceleration q [m^2/s^3]); measurement = WLS fix with its covariance;
   chi-square innovation gate; prediction only when no fix; restart from the
   current fix after 10 consecutive rejected fixes (1 s; fixed, not tuned).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

C0 = 299_792_458.0
TWO_PI = 2.0 * math.pi
NPOS = 8
REINIT_AFTER = 10  # consecutive gated-out fixes (1 s) before the EKF restarts from the current fix (fixed, not tuned)


# ----------------------------------------------------------------------------- synthesis
def synth_numpy(beta, tau_ns, u, f_hz, r, wl, psi=None, noise=None):
    """Y [N, M] for one O-RU/epoch (reference)."""
    from sim.positioning.array import steering

    s = steering(u, r, wl)  # [P, M]
    g = np.exp(-1j * TWO_PI * f_hz[None, :] * tau_ns[:, None] * 1e-9)  # [P, N]
    Y = (beta[:, None, None] * g[:, :, None] * s[:, None, :]).sum(0)
    if psi is not None:
        Y = Y * np.exp(1j * psi)[None, :]
    if noise is not None:
        Y = Y + noise
    return Y


def synth_torch(beta, tau_ns, u, f_hz, r, wl, psi=None, gen=None, add_noise=True):
    """Y [B, N, M] complex64. beta [B, P] complex, tau_ns [B, P], u [B, P, 3]; psi [B, M] or None."""
    import torch

    dev = beta.device
    rr = torch.as_tensor(r, device=dev, dtype=torch.float64)
    s = torch.exp(1j * (TWO_PI / wl) * (u @ rr.T))  # [B, P, M]
    f = torch.as_tensor(f_hz, device=dev, dtype=torch.float64)
    g = torch.exp(-1j * TWO_PI * f[None, None, :] * tau_ns[..., None] * 1e-9)  # [B, P, N]
    Y = torch.einsum("bp,bpn,bpm->bnm", beta.to(torch.complex128), g, s)
    if psi is not None:
        Y = Y * torch.exp(1j * psi)[:, None, :]
    Y = Y.to(torch.complex64)
    if add_noise:
        w = (torch.randn(Y.shape, generator=gen, device=dev) + 1j * torch.randn(Y.shape, generator=gen, device=dev)) / math.sqrt(2.0)
        Y = Y + w.to(torch.complex64)
    return Y


# ----------------------------------------------------------------------------- measurement
def _parabolic(vm, v0, vp):
    """Sub-bin offset of a peak from three samples (power)."""
    den = vm - 2.0 * v0 + vp
    return np.where(np.abs(den) > 1e-30, 0.5 * (vm - vp) / np.where(np.abs(den) > 1e-30, den, 1.0), 0.0)


def measure_torch(Y, f_hz, wl, *, Q: int = 32, Q2: int = 64, pad: int = 8, guard: int = 2) -> dict[str, np.ndarray]:
    """Dominant-path measurement for a batch Y [B, N, 64] (element order of sim.positioning.array)."""
    import torch

    B, N, M = Y.shape
    df = float(f_hz[1] - f_hz[0])
    Yg = Y.reshape(B, N, NPOS, NPOS)  # [B, N, i(y), j(z)]
    # 1. strongest beam (power summed over subcarriers)
    P = torch.zeros((B, Q, Q), device=Y.device)
    for lo in range(0, N, 512):
        F2 = torch.fft.fft2(Yg[:, lo:lo + 512], s=(Q, Q))
        P += (F2.abs() ** 2).sum(1)
    k = P.reshape(B, -1).argmax(-1)
    qi, qj = (k // Q).cpu().numpy(), (k % Q).cpu().numpy()
    Pn = P.cpu().numpy()
    bi = np.arange(B)
    di = _parabolic(Pn[bi, (qi - 1) % Q, qj], Pn[bi, qi, qj], Pn[bi, (qi + 1) % Q, qj])
    dj = _parabolic(Pn[bi, qi, (qj - 1) % Q], Pn[bi, qi, qj], Pn[bi, qi, (qj + 1) % Q])
    uy = _wrap2((qi + di) / Q * 2.0)
    uz = -_wrap2((qj + dj) / Q * 2.0)
    fr = torch.as_tensor(f_hz, device=Y.device, dtype=torch.float64)
    L = pad * (1 << int(math.ceil(math.log2(N))))
    out = {}
    for _it in range(2):
        w = _bf_weights(uy, uz, Y.device)  # [B, 64]
        yb = torch.einsum("bnm,bm->bn", Y.to(torch.complex128), torch.conj(w))
        win = torch.as_tensor(np.hanning(N + 2)[1:-1], device=Y.device)  # Hann over subcarriers: sidelobes ~ -31 dB
        prof = torch.fft.ifft(yb * win, n=L) * L  # peak at l/L = frac(df * tau)
        pw = (prof.abs() ** 2).cpu().numpy()
        l0 = pw.argmax(-1)
        dl = _parabolic(pw[bi, (l0 - 1) % L], pw[bi, l0], pw[bi, (l0 + 1) % L])
        tau = ((l0 + dl) / L) / df  # [s] modulo 1/df
        tau_ns = tau * 1e9
        # gate: second peak outside +/- guard resolution cells
        cell = int(round(L / N))
        mask = np.ones_like(pw, dtype=bool)
        for s in range(-guard * cell, guard * cell + 1):
            mask[bi, (l0 + s) % L] = False
        second = np.where(mask, pw, 0.0).max(-1)
        ratio_db = 10 * np.log10(np.maximum(pw[bi, l0], 1e-30) / np.maximum(second, 1e-30))
        # angle refinement on the snapshot at tau
        ph = torch.exp(1j * TWO_PI * fr[None, :] * torch.as_tensor(tau, device=Y.device)[:, None])
        z = torch.einsum("bnm,bn->bm", Y.to(torch.complex128), ph).reshape(B, NPOS, NPOS)
        A = (torch.fft.fft2(z, s=(Q2, Q2)).abs() ** 2).cpu().numpy()
        kk = A.reshape(B, -1).argmax(-1)
        ai, aj = kk // Q2, kk % Q2
        di = _parabolic(A[bi, (ai - 1) % Q2, aj], A[bi, ai, aj], A[bi, (ai + 1) % Q2, aj])
        dj = _parabolic(A[bi, ai, (aj - 1) % Q2], A[bi, ai, aj], A[bi, ai, (aj + 1) % Q2])
        uy = _wrap2((ai + di) / Q2 * 2.0)
        uz = -_wrap2((aj + dj) / Q2 * 2.0)
        snr = A[bi, ai, aj] / (N * M)  # coherent energy / noise variance of the coherent sum
        out = {"tau_ns": tau_ns, "uy": uy, "uz": uz, "snr": snr, "ratio_db": ratio_db}
    return out


def measure_numpy(Y, f_hz, wl, *, Q: int = 32, Q2: int = 64, pad: int = 8, guard: int = 2) -> dict[str, np.ndarray]:
    """Reference of measure_torch for a batch Y [B, N, 64] (same algorithm, NumPy FFTs)."""
    B, N, M = Y.shape
    df = float(f_hz[1] - f_hz[0])
    Yg = Y.reshape(B, N, NPOS, NPOS)
    P = (np.abs(np.fft.fft2(Yg, s=(Q, Q))) ** 2).sum(1)
    k = P.reshape(B, -1).argmax(-1)
    qi, qj = k // Q, k % Q
    bi = np.arange(B)
    di = _parabolic(P[bi, (qi - 1) % Q, qj], P[bi, qi, qj], P[bi, (qi + 1) % Q, qj])
    dj = _parabolic(P[bi, qi, (qj - 1) % Q], P[bi, qi, qj], P[bi, qi, (qj + 1) % Q])
    uy, uz = _wrap2((qi + di) / Q * 2.0), -_wrap2((qj + dj) / Q * 2.0)
    L = pad * (1 << int(math.ceil(math.log2(N))))
    out = {}
    for _it in range(2):
        w = _bf_weights_np(uy, uz)
        yb = np.einsum("bnm,bm->bn", Y, np.conj(w))
        pw = np.abs(np.fft.ifft(yb * np.hanning(N + 2)[1:-1], n=L) * L) ** 2
        l0 = pw.argmax(-1)
        dl = _parabolic(pw[bi, (l0 - 1) % L], pw[bi, l0], pw[bi, (l0 + 1) % L])
        tau = ((l0 + dl) / L) / df
        cell = int(round(L / N))
        mask = np.ones_like(pw, dtype=bool)
        for s in range(-guard * cell, guard * cell + 1):
            mask[bi, (l0 + s) % L] = False
        second = np.where(mask, pw, 0.0).max(-1)
        ratio_db = 10 * np.log10(np.maximum(pw[bi, l0], 1e-30) / np.maximum(second, 1e-30))
        z = np.einsum("bnm,bn->bm", Y, np.exp(1j * TWO_PI * f_hz[None, :] * tau[:, None])).reshape(B, NPOS, NPOS)
        A = np.abs(np.fft.fft2(z, s=(Q2, Q2))) ** 2
        kk = A.reshape(B, -1).argmax(-1)
        ai, aj = kk // Q2, kk % Q2
        di = _parabolic(A[bi, (ai - 1) % Q2, aj], A[bi, ai, aj], A[bi, (ai + 1) % Q2, aj])
        dj = _parabolic(A[bi, ai, (aj - 1) % Q2], A[bi, ai, aj], A[bi, ai, (aj + 1) % Q2])
        uy, uz = _wrap2((ai + di) / Q2 * 2.0), -_wrap2((aj + dj) / Q2 * 2.0)
        out = {"tau_ns": tau * 1e9, "uy": uy, "uz": uz, "snr": A[bi, ai, aj] / (N * M), "ratio_db": ratio_db}
    return out


def _wrap2(x):
    """Spatial frequency in [-1, 1)."""
    return (np.asarray(x) + 1.0) % 2.0 - 1.0


def _bf_weights_np(uy, uz):
    i = np.arange(NPOS) - (NPOS - 1) / 2.0
    ph = math.pi * (uy[:, None, None] * i[None, :, None] - uz[:, None, None] * i[None, None, :])
    return np.exp(1j * ph).reshape(len(uy), -1)


def _bf_weights(uy, uz, device):
    import torch

    return torch.as_tensor(_bf_weights_np(np.asarray(uy), np.asarray(uz)), device=device)


# ----------------------------------------------------------------------------- fusion (WLS) and EKF
def predict_meas(p, o):
    """Predicted (range [m], u_y, u_z) from UE position p [..., 3] and O-RU o [..., 3]."""
    d = p - o
    rng = np.linalg.norm(d, axis=-1)
    return rng, d[..., 1] / rng, d[..., 2] / rng


def ray_fix(o, uy, uz, front: float, z_ue: float):
    """UE position from one O-RU's AoA (u_x sign = front) intersected with z = z_ue; NaN if not downward."""
    ux = front * np.sqrt(np.clip(1.0 - uy ** 2 - uz ** 2, 0.0, 1.0))
    t = (z_ue - o[..., 2]) / np.where(uz < -1e-6, uz, np.nan)
    return np.stack([o[..., 0] + t * ux, o[..., 1] + t * uy], -1)


def wls_candidates(meas: dict, oru: np.ndarray, use: np.ndarray, timing: str, sig_tau_ns: np.ndarray, sig_u: np.ndarray, z_ue: float,
                   n_iter: int = 8) -> dict[str, np.ndarray]:
    """Batched WLS over epochs from every single-O-RU AoA ray fix (both front/back hypotheses).

    meas: tau_ns/uy/uz [E, C]; oru [E, C, 3]; use [E, C] bool; sig_* [E, C].
    Returns per init k: xy [E, K, 2], cov [E, K, 2, 2], cost [E, K] (inf if the init is invalid or no O-RU is used).
    """
    E, C = use.shape
    inits = []
    for c in range(C):
        for front in (1.0, -1.0):
            inits.append(ray_fix(oru[:, c], meas["uy"][:, c], meas["uz"][:, c], front, z_ue))
    n_par = 3 if timing == "tdoa" else 2
    w_tau = timing != "aoa"
    xs, covs, costs = [], [], []
    for init in inits:
        th = np.zeros((E, n_par))
        th[:, :2] = np.nan_to_num(init, nan=0.0)
        valid0 = np.isfinite(init).all(-1) & use.any(-1)
        if timing == "tdoa":  # start the clock bias at the mean delay residual
            p0 = np.concatenate([th[:, :2], np.full((E, 1), z_ue)], -1)
            res = [np.where(use[:, c], meas["tau_ns"][:, c] - predict_meas(p0, oru[:, c])[0] / C0 * 1e9, np.nan) for c in range(C)]
            th[:, 2] = np.nan_to_num(np.nanmean(np.stack(res, -1), -1))
        for _ in range(n_iter):
            p = np.concatenate([th[:, :2], np.full((E, 1), z_ue)], -1)
            rows_r, rows_J = [], []
            for c in range(C):
                rng, py, pz = predict_meas(p, oru[:, c])
                d = p - oru[:, c]
                dy = np.stack([-d[:, 1] * d[:, 0] / rng ** 3, 1.0 / rng - d[:, 1] ** 2 / rng ** 3], -1)
                dz = np.stack([-d[:, 2] * d[:, 0] / rng ** 3, -d[:, 2] * d[:, 1] / rng ** 3], -1)
                wgt = use[:, c].astype(float)
                for res, jac, sig in ((meas["uy"][:, c] - py, dy, sig_u[:, c]), (meas["uz"][:, c] - pz, dz, sig_u[:, c])):
                    Jr = np.zeros((E, n_par))
                    Jr[:, :2] = jac
                    rows_r.append(wgt * res / sig)
                    rows_J.append(wgt[:, None] * Jr / sig[:, None])
                if w_tau:
                    pred = rng / C0 * 1e9 + (th[:, 2] if timing == "tdoa" else 0.0)
                    Jr = np.zeros((E, n_par))
                    Jr[:, :2] = d[:, :2] / rng[:, None] / C0 * 1e9
                    if timing == "tdoa":
                        Jr[:, 2] = 1.0
                    rows_r.append(wgt * (meas["tau_ns"][:, c] - pred) / sig_tau_ns[:, c])
                    rows_J.append(wgt[:, None] * Jr / sig_tau_ns[:, c][:, None])
            r = np.stack(rows_r, -1)
            J = np.stack(rows_J, -2)
            JtJ = J.transpose(0, 2, 1) @ J + 1e-9 * np.eye(n_par)
            step = np.linalg.solve(JtJ, (J.transpose(0, 2, 1) @ r[..., None]))[..., 0]
            step[:, :2] = np.clip(step[:, :2], -20.0, 20.0)
            th = th + step
        p = np.concatenate([th[:, :2], np.full((E, 1), z_ue)], -1)
        r_final = []
        for c in range(C):
            rng, py, pz = predict_meas(p, oru[:, c])
            wgt = use[:, c].astype(float)
            r_final += [wgt * (meas["uy"][:, c] - py) / sig_u[:, c], wgt * (meas["uz"][:, c] - pz) / sig_u[:, c]]
            if w_tau:
                r_final.append(wgt * (meas["tau_ns"][:, c] - rng / C0 * 1e9 - (th[:, 2] if timing == "tdoa" else 0.0)) / sig_tau_ns[:, c])
        cost = (np.stack(r_final, -1) ** 2).sum(-1)
        cov = np.linalg.inv(JtJ)[:, :2, :2]
        xs.append(th[:, :2])
        covs.append(cov)
        costs.append(np.where(valid0 & np.isfinite(cost), cost, np.inf))
    return {"xy": np.stack(xs, 1), "cov": np.stack(covs, 1), "cost": np.stack(costs, 1)}


def _pick(cand: dict, e: int, prior) -> tuple[np.ndarray, np.ndarray, float] | None:
    c = cand["cost"][e]
    if not np.isfinite(c).any():
        return None
    cmin = np.min(c)
    ties = np.flatnonzero(c <= cmin + 1.0)
    if prior is not None and ties.size > 1:
        k = ties[np.argmin(np.linalg.norm(cand["xy"][e, ties] - prior, axis=-1))]
    else:
        k = int(np.argmin(c))
    return cand["xy"][e, k], cand["cov"][e, k], float(cmin)


def run_tracker(cands: dict, snr: np.ndarray, params: dict, dt: float = 0.1) -> dict[str, np.ndarray]:
    """EKF over the epochs of one UE. cands: {"all": wls_candidates(all gated O-RUs), c: (O-RU c alone)}; snr [E, C]."""
    E = cands["all"]["cost"].shape[0]
    singles = [k for k in cands if k != "all"]
    q = float(params["q"])
    chi2 = float(params["chi2"])
    res_gate = float(params["res_gate"])
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Qm = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2], [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
    xy_ekf = np.full((E, 2), np.nan)
    xy_fix = np.full((E, 2), np.nan)
    x = None
    Pm = None
    rejected = 0
    for e in range(E):
        prior = None
        if x is not None:
            x = F @ x
            Pm = F @ Pm @ F.T + Qm
            prior = x[:2]
        sel = _pick(cands["all"], e, prior)
        # before the track exists the joint fix is always preferred (a single O-RU is front/back ambiguous)
        if sel is None or (sel[2] > res_gate and prior is not None):
            opts = [(k, _pick(cands[k], e, prior)) for k in singles]
            opts = [(k, o) for k, o in opts if o is not None and np.all(np.isfinite(o[0]))]
            if opts:
                if prior is not None:
                    sel = min(opts, key=lambda ko: float(np.linalg.norm(ko[1][0] - prior)))[1]
                else:
                    sel = max(opts, key=lambda ko: float(snr[e, ko[0]]))[1]
            elif sel is not None and sel[2] > res_gate:
                sel = None
        if sel is not None and np.all(np.isfinite(sel[0])) and np.all(np.isfinite(sel[1])):
            z, R = sel[0], sel[1] + 1e-6 * np.eye(2)
            xy_fix[e] = z
            if x is None:
                x = np.array([z[0], z[1], 0.0, 0.0])
                Pm = np.diag([R[0, 0], R[1, 1], 4.0, 4.0])
            else:
                S = Pm[:2, :2] + R
                v = z - x[:2]
                if float(v @ np.linalg.solve(S, v)) <= chi2:
                    Kg = Pm[:, :2] @ np.linalg.inv(S)
                    x = x + Kg @ v
                    Pm = Pm - Kg @ Pm[:2, :]
                    rejected = 0
                else:
                    rejected += 1
                    if rejected >= REINIT_AFTER:  # track lost: restart from the current fix
                        x = np.array([z[0], z[1], 0.0, 0.0])
                        Pm = np.diag([R[0, 0], R[1, 1], 4.0, 4.0])
                        rejected = 0
        if x is not None:
            xy_ekf[e] = x[:2]
    return {"xy_ekf": xy_ekf, "xy_fix": xy_fix}


def noise_model(meas: dict, params: dict, df_hz: float, n_sc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gate and measurement standard deviations (CRLB proxies + tuned floors): use [E, C], sig_tau_ns, sig_u."""
    snr = np.maximum(meas["snr"], 1e-3)
    brms = df_hz * math.sqrt((n_sc ** 2 - 1) / 12.0)
    sig_tau = np.sqrt((1e9 / (TWO_PI * brms * np.sqrt(2.0 * snr))) ** 2 + float(params["floor_tau_ns"]) ** 2)
    arr_rms = math.sqrt((NPOS ** 2 - 1) / 12.0) * math.pi
    sig_u = np.sqrt((1.0 / (arr_rms * np.sqrt(2.0 * snr * NPOS))) ** 2 + float(params["floor_u"]) ** 2)
    use = meas["ratio_db"] >= float(params["gate_db"])
    return use, sig_tau, sig_u
