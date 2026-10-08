"""Service models for the cell SNR (TVT T0 harmonization).

The paper-1 timeline (xapp/timeline.py, frozen) gives, per 10 ms step, UE and cell,
the POWER SUM over the traced paths of each path's power over the 8x8 array
(sum_m |a_{m,p}|^2 = 64 |a_p|^2, i.e. full array gain for every path, powers added)
after its model-B loss. This module adds two coherent models on the same paths,
blockage losses and timeline:

- ``best_beam``: the gain of the best beam of a fixed codebook (2-D DFT over the
  (u_y, u_z) spatial frequencies of the y-z array, oversampling ``os``, unit-norm
  weights), averaged over the occupied band:
  G(w) = (1/N) sum_n |w^H h(f_n)|^2 = sum_{p,q} c_p c_q^* K(tau_p - tau_q),
  c_p = sqrt(g_p) a_p (w^H s(u_p)), K(d) = (1/N) sum_n exp(-j 2 pi f_n d),
  with g_p the model-B power gain of path p and f_n the subcarrier offsets.
- ``mrt``: matched filter over the band, G = sum_{p,q} sqrt(g_p g_q) a_p a_q^* s_p^H s_q K_pq
  (upper bound of every single-beam model; equals the power sum for one path).
- ``power_sum``: the paper-1 model, sum_p g_p 64 |a_p|^2 (bitwise the frozen timeline).

Path coefficients a_p: complex coefficient at the array centre per 0.1 s snapshot
(results/P2/cache/.../p2_paths.npz, a_center; |a|^2 64 = cached power to 3e-7).
Between snapshots the magnitude is held (as in the frozen timeline) and the phase
follows the path length L(t) of the held polyline with the exact UE end point:
phase(t) = arg a(snapshot) - 2 pi (L(t) - L(t_snapshot)) / lambda (specular point
stationary to first order, Fermat). Arrival direction u_p(t): first segment of the
held polyline (the LoS direction follows the UE).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MODELS = ("power_sum", "best_beam", "mrt")
CLASS_LOS = 0


def per_path_loss_timeline(seg: dict[str, np.ndarray], blocker_pos: np.ndarray, blocker_size: np.ndarray, wavelength_m: float,
                           *, device: str = "cuda", max_elements: int = 60_000_000) -> np.ndarray:
    """Model-B loss [dB] per path [T, U, C, P] (inf = fully blocked); same arithmetic as xapp.timeline.model_b_timeline."""
    import torch

    from sim.comm.blockage_torch import screen_loss_db

    n_t = seg["starts"].shape[0]
    per_t = int(np.prod(seg["starts"].shape[1:-1])) * blocker_pos.shape[1]
    chunk = max(1, int(max_elements // max(per_t, 1)))
    size = torch.as_tensor(blocker_size, device=device, dtype=torch.float64)
    out = []
    for lo in range(0, n_t, chunk):
        hi = min(n_t, lo + chunk)
        starts = torch.as_tensor(seg["starts"][lo:hi], device=device)[..., None, :]
        ends = torch.as_tensor(seg["ends"][lo:hi], device=device)[..., None, :]
        centers = torch.as_tensor(blocker_pos[lo:hi], device=device)[:, None, None, None, None, :, :]
        loss = screen_loss_db(starts, ends, centers, size[:, 0], size[:, 1], size[:, 2], wavelength_m)
        valid = torch.as_tensor(seg["seg_valid"][lo:hi], device=device)[..., None]
        loss = torch.where(valid, loss, torch.zeros_like(loss))  # [t,U,C,P,S,B]
        inf_any = (~torch.isfinite(loss)).any(dim=-2)
        per_pb = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)).sum(dim=-2)
        per_pb = torch.where(inf_any, torch.full_like(per_pb, math.inf), per_pb)  # [t,U,C,P,B]
        inf_p = (~torch.isfinite(per_pb)).any(dim=-1)
        path = torch.where(torch.isfinite(per_pb), per_pb, torch.zeros_like(per_pb)).sum(dim=-1)
        out.append(torch.where(inf_p, torch.full_like(path, math.inf), path).cpu().numpy())
    return np.concatenate(out, axis=0)


def codebook(wavelength_m: float, oversampling: int = 2) -> np.ndarray:
    """Unit-norm 2-D DFT beams [W, 64] on the (u_y, u_z) grid (-1 + (2k + 1) / (8 os)), Sionna element order."""
    from sim.positioning.array import element_positions

    r = element_positions(wavelength_m)
    n = 8 * int(oversampling)
    g = -1.0 + (2.0 * np.arange(n) + 1.0) / n
    uy, uz = np.meshgrid(g, g, indexing="ij")
    phase = 2.0 * np.pi / wavelength_m * (uy.reshape(-1, 1) * r[None, :, 1] + uz.reshape(-1, 1) * r[None, :, 2])
    return np.exp(1j * phase) / 8.0


def path_states(seg: dict[str, np.ndarray], a_center: np.ndarray, wavelength_m: float, steps_per_snapshot: int) -> dict[str, np.ndarray]:
    """Complex coefficient a_p(t) [T,U,C,P], arrival direction u_p(t) [T,U,C,P,3] and delay tau_p(t) [T,U,C,P] (s)."""
    starts, ends, valid = seg["starts"], seg["ends"], seg["seg_valid"]
    length = np.where(valid, np.linalg.norm(ends - starts, axis=-1), 0.0).sum(-1)  # [T,U,C,P]
    snap = seg["snap"]
    k_snap = np.minimum(snap * int(steps_per_snapshot), length.shape[0] - 1)
    l_ref = length[k_snap]
    a0 = a_center[snap]  # [T,U,C,P]
    a = np.abs(a0) * np.exp(1j * (np.angle(a0) - 2.0 * np.pi * (length - l_ref) / wavelength_m))
    d = ends[..., 0, :] - starts[..., 0, :]
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    return {"a": np.where(seg["path_class"] >= 0, a, 0.0), "u": u, "tau": length / 299_792_458.0}


def band_kernel(tau: Any, f_offsets: Any, xp: Any = np) -> Any:
    """K_pq = mean_n exp(-j 2 pi f_n (tau_p - tau_q)) [..., P, P] (Dirichlet kernel; f_n symmetric)."""
    d = tau[..., :, None] - tau[..., None, :]
    n = f_offsets.shape[0]
    df = float(f_offsets[1] - f_offsets[0])
    x = math.pi * df * d
    num = xp.sin(n * x)
    den = n * xp.sin(x)
    small = xp.abs(den) < 1e-12
    return xp.where(small, xp.ones_like(d), num / xp.where(small, xp.ones_like(den), den)) + 0j


def gains(model: str, a: np.ndarray, u: np.ndarray, tau: np.ndarray, gain: np.ndarray, wavelength_m: float, f_offsets: np.ndarray,
          *, oversampling: int = 2, device: str = "cuda", chunk: int = 512) -> dict[str, np.ndarray]:
    """Per step [T,U,C]: total gain (all paths with their model-B gains), unblocked gain, LoS-only and best single NLoS gain.

    ``gain`` [T,U,C,P] linear model-B power gain (0 = blocked). ``a`` 0 for empty slots.
    """
    import torch

    from sim.positioning.array import element_positions

    if model not in MODELS:
        raise ValueError(f"service model {model!r} not in {MODELS}")
    cls_los = None
    r = torch.as_tensor(element_positions(wavelength_m), device=device)
    k0 = 2.0 * math.pi / wavelength_m
    W = torch.as_tensor(codebook(wavelength_m, oversampling), device=device) if model == "best_beam" else None
    f = torch.as_tensor(f_offsets, device=device)
    out = {k: [] for k in ("total", "unblocked", "single")}
    for lo in range(0, a.shape[0], chunk):
        hi = min(a.shape[0], lo + chunk)
        at = torch.as_tensor(a[lo:hi], device=device)
        ut = torch.as_tensor(u[lo:hi], device=device)
        tt = torch.as_tensor(tau[lo:hi], device=device)
        gt = torch.as_tensor(gain[lo:hi], device=device)
        s = torch.exp(1j * k0 * (ut @ r.T))  # [t,U,C,P,64]
        K = band_kernel(tt, f, torch)  # [t,U,C,P,P]
        if model == "best_beam":
            b = torch.einsum("wm,...pm->...pw", W.conj(), s)  # w^H s_p
        res = {}
        for name, amp in (("total", torch.sqrt(gt) * at), ("unblocked", at)):
            if model == "power_sum":
                res[name] = 64.0 * (amp.abs() ** 2).sum(-1)
            elif model == "mrt":
                ss = torch.einsum("...pm,...qm->...pq", s.conj(), s)
                res[name] = torch.einsum("...p,...q,...pq,...pq->...", amp, amp.conj(), ss, K).real
            else:
                c = amp[..., None] * b  # [t,U,C,P,W]
                res[name] = torch.einsum("...pw,...qw,...pq->...w", c, c.conj(), K).real.amax(-1)
        # single-path gains (blocked): power_sum / mrt -> 64 g |a|^2; best_beam -> g |a|^2 max_w |w^H s|^2
        amp = torch.sqrt(gt) * at
        if model == "best_beam":
            res["single"] = (amp.abs() ** 2) * (b.abs() ** 2).amax(-1)
        else:
            res["single"] = 64.0 * amp.abs() ** 2
        for k in out:
            out[k].append(res[k].cpu().numpy())
    del cls_los
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def timeline_fields(model: str, seg: dict[str, np.ndarray], path_loss_db: np.ndarray, a_center: np.ndarray, wavelength_m: float,
                    f_offsets: np.ndarray, steps_per_snapshot: int, *, oversampling: int = 2, device: str = "cuda",
                    panels: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """The four power fields of the frozen timeline under ``model``: unblocked_power, blocked_power, los_blocked_power, best_alt_power [T,U,C].

    ``panels`` (configs/tvt.yaml panels: back_to_back): {"ue": [T, U, 3] UE positions per step, "oru": [C, 3]}; every
    path coefficient is weighted by the TR 38.901 element amplitude of the panel facing the UE at its departure
    direction (sim/tvt/panels.link_amplitude). The back panel's element grid is the mirror of the front one, so the
    best-beam codebook gains are unchanged (symmetric 8x8 grid and DFT grid). None = the papers' isotropic UPA.
    """
    st = path_states(seg, a_center, wavelength_m, steps_per_snapshot)
    if panels is not None:
        from sim.tvt.panels import link_amplitude, link_amplitude_yaws

        if panels.get("yaws") is None:
            amp, _sx = link_amplitude(st["u"], panels["ue"], panels["oru"])
        else:  # any panel set (intersection: four panels): codebook of the selected panel (sim/tvt/panels.link_amplitude_yaws)
            amp, _yaw, st["u"] = link_amplitude_yaws(st["u"], panels["ue"], panels["oru"], panels["yaws"])
        st["a"] = st["a"] * amp
    gain = np.where(np.isfinite(path_loss_db), 10.0 ** (-np.nan_to_num(path_loss_db, posinf=0.0) / 10.0), 0.0)
    gain = np.where(seg["path_class"] >= 0, gain, 0.0)
    g = gains(model, st["a"], st["u"], st["tau"], gain, wavelength_m, f_offsets, oversampling=oversampling, device=device)
    cls = seg["path_class"]
    los = cls == CLASS_LOS
    single = g["single"]
    los_p = np.where(los, single, -1.0).max(-1)
    alt = np.where((cls >= 0) & ~los, single, -1.0).max(-1)
    return {"unblocked_power": g["unblocked"], "blocked_power": g["total"], "los_blocked_power": np.maximum(los_p, 0.0), "best_alt_power": np.maximum(alt, 0.0)}
