"""Position error bound with the physical array model and an uncertain map (TVT T1).

Per job and 0.1 s epoch (paper-2 inputs: paper-1 comm geometry, complex path coefficients,
model-B loss per path, image-method geometry on the known planes; scripts/p2_peb.job_inputs,
frozen). Unknowns theta = (x, y, delta_0, delta_1, b, d_1..d_S): UE horizontal position, O-RU
time offsets (prior sigma_sync), UE clock bias (TDoA, free) and the offsets of the S uncertain
surfaces (facades) with prior N(0, sigma_map^2) (sigma_map = 0: map known; inf: offsets free).
Element nuisances (phase / gain / position errors) per O-RU with Gaussian priors are
eliminated per O-RU (sim/tvt/fim.efim_geometric). "los": every NLoS path is a free nuisance
(LoS-only information); "map": NLoS paths informative through the image method. Blocked LoS
(model-B LoS loss >= 10 dB): "biased" (geometry free) as the paper-2 main configuration.
Sanity model "va" (``va_sigmas``): instead of the surface offsets, the virtual anchor of EVERY
NLoS path gets its own free 3-D displacement with prior N(0, sigma_va^2 I)
(sim/tvt/geometry_map.va_derivatives); sigma_va -> 0 must give map_0, sigma_va -> inf approaches "los".
"""

from __future__ import annotations

import math

import numpy as np

N_BASE = 5  # x, y, delta_0, delta_1, b


def job_setup(job: tuple, raw: dict, p2cfg: dict, axes: str = "xy") -> dict:
    """Frozen paper-2 inputs plus surface ids and offset derivatives (uncertain surfaces: planes normal to ``axes``)."""
    import p2_peb
    from sim.positioning.geometry import bounce_planes
    from sim.tvt.geometry_map import offset_derivatives, surface_ids, surfaces, va_derivatives

    inp = p2_peb.job_inputs(job, raw, p2cfg)
    seed, mount, density = job
    from pathlib import Path

    root = Path(p2_peb.__file__).resolve().parents[1]
    with np.load(root / "results" / "cache" / mount / density / f"seed_{seed}" / "comm_geometry.npz") as g:
        P = g["points_m"]
        n = g["n_points"].astype(np.int64)
        ue = np.broadcast_to(g["ue_position_m"][:, :, None, None, :], P.shape[:-2] + (3,))
        oru = np.broadcast_to(g["oru_position_m"][:, None, :, None, :], P.shape[:-2] + (3,))
    N, A, V = bounce_planes(P, n, p2cfg["known_planes"])
    surf = surfaces(p2cfg["known_planes"], axes)
    sid = surface_ids(N, A, V, surf)
    od = offset_derivatives(oru, ue, N, A, V)
    inp.update({"surf": surf, "sid": sid, "doff": od, "dva": va_derivatives(oru, ue, N, A, V)})
    return inp


def h_matrix(inp: dict) -> np.ndarray:
    """H [T, U, C, 3P, 5 + S]: d(tau [ns], az, el) / d theta."""
    g = inp["geo"]
    T, U, C, P = inp["a"].shape
    S = len(inp["surf"])
    H = np.zeros((T, U, C, 3 * P, N_BASE + S))
    for c in range(C):
        for p in range(P):
            H[:, :, c, 3 * p, 0:2] = g["dtau"][:, :, c, p] * 1e9
            H[:, :, c, 3 * p, 2 + c] = 1.0
            H[:, :, c, 3 * p, 4] = 1.0
            H[:, :, c, 3 * p + 1, 0:2] = g["daz"][:, :, c, p]
            H[:, :, c, 3 * p + 2, 0:2] = g["del"][:, :, c, p]
    od = inp["doff"]
    sid = inp["sid"]  # [T, U, C, P, 2]
    for slot in range(2):
        for s in range(S):
            m = sid[..., slot] == s  # [T, U, C, P]
            for i, key, scale in ((0, "dtau", 1e9), (1, "daz", 1.0), (2, "del", 1.0)):
                val = np.where(m, od[key][..., slot] * scale, 0.0)  # [T, U, C, P]
                H[..., i::3, N_BASE + s] += val
    return H


def h_matrix_va(inp: dict) -> np.ndarray:
    """H [T, U, C, 3P, 5 + 3 C P]: base columns and a free 3-D VA displacement per (O-RU, path) (zero for LoS / absent)."""
    H0 = h_matrix(inp)[..., :N_BASE]
    T, U, C, P = inp["a"].shape
    dva = inp["dva"]  # [T, U, C, P, 3]
    H = np.concatenate([H0, np.zeros(H0.shape[:-1] + (3 * C * P,))], -1)
    for c in range(C):
        for p in range(P):
            col = N_BASE + 3 * (c * P + p)
            for i, key, scale in ((0, "dtau", 1e9), (1, "daz", 1.0), (2, "del", 1.0)):
                H[:, :, c, 3 * p + i, col:col + 3] = dva[key][:, :, c, p] * scale
    return H


def pebs(inp: dict, f_hz: np.ndarray, beta_scale: float, *, sync_ns: float, sigma_phi_deg: float, sigma_g_db: float, sigma_r_m: float,
         pattern: str, sigma_maps: list[float], timing: str = "tdoa", blocked_db: float = 10.0, device: str = "cuda",
         chunk: int = 25, va_sigmas: list[float] | None = None, panels: bool | list[float] = False) -> dict[str, np.ndarray]:
    """PEB [T, U] for "los", for "map" at every sigma_map (m; 0 = known, inf = free offsets) and "va_<s>" for every sigma_va.

    ``panels`` (configs/tvt.yaml panels: back_to_back): every O-RU has two back-to-back panels (+x, -x) with ``pattern``.
    Each panel's Fisher information is the frozen gram in its local frame (az - pi for the -x panel: the local direction
    R^T u with the unrotated element grid is the world direction with the rotated grid); path amplitudes and element
    errors are eliminated per panel (independent nuisances, unknown inter-panel calibration), and the geometric EFIMs of
    the two panels are summed (shared delays, angles, clocks). A list = the panel yaws [deg] (intersection: 0, 180, 90,
    270; az - yaw per panel, the EFIMs of all panels summed)."""
    import torch

    from sim.positioning.array import element_positions
    from sim.positioning.fim import peb_from_theta, theta_information
    from sim.tvt.fim import efim_geometric, element_prior, gram_general

    wl = inp["wl"]
    r = element_positions(wl)
    g = inp["geo"]
    T, U, C, P = inp["a"].shape
    S = len(inp["surf"])
    kinds = tuple(k for k, s in (("phase", sigma_phi_deg), ("gain", sigma_g_db), ("pos", sigma_r_m)) if s > 0) or ("phase",)
    prior_e = element_prior(kinds, r.shape[0], math.radians(sigma_phi_deg), sigma_g_db, sigma_r_m, device)
    H = torch.as_tensor(h_matrix(inp), device=device)
    beta = torch.as_tensor(inp["a"] * beta_scale, device=device)
    tau = torch.as_tensor(g["tau"] * 1e9, device=device)
    az = torch.as_tensor(g["az"], device=device)
    el = torch.as_tensor(g["el"], device=device)
    cls = torch.as_tensor(inp["cls"], device=device)
    absent = torch.as_tensor(~inp["valid"], device=device)
    blk = torch.as_tensor(inp["los_loss"] >= blocked_db, device=device)
    is_los, is_nlos = cls == 0, cls > 0
    if isinstance(panels, (list, tuple)):
        shifts = tuple(0.0 if y == 0.0 else (math.pi if y == 180.0 else math.radians(y)) for y in panels)
    else:
        shifts = (0.0, math.pi) if panels else (0.0,)
    Jpan = []
    for sh in shifts:
        Ks = []
        for lo in range(0, T, chunk):
            azl = torch.remainder(az[lo:lo + chunk] - sh + math.pi, 2 * math.pi) - math.pi
            Ks.append(gram_general(beta[lo:lo + chunk], tau[lo:lo + chunk], azl, el[lo:lo + chunk], f_hz, r, wl, pattern=pattern, kinds=kinds))
        Jpan.append(torch.cat(Ks, 0))
    out = {}
    for info in ["los", "map"]:
        free_path = absent | (is_los & blk[..., None])
        if info == "los":
            free_path = free_path | is_nlos
        free = free_path[..., None].expand(*free_path.shape, 3).clone()
        if timing == "aoa":
            free[..., 0] = True
        free = free.reshape(T, U, C, 3 * P)
        Keff = sum(torch.cat([efim_geometric(Jall[lo:lo + chunk], P, prior_e, free[lo:lo + chunk]) for lo in range(0, T, chunk)], 0) for Jall in Jpan)
        Keff = torch.where(free[..., :, None] | free[..., None, :], torch.zeros_like(Keff), Keff)
        J0 = theta_information(Keff, H)
        for sm in ([math.nan] if info == "los" else sigma_maps):
            n_th = N_BASE + S
            prior = torch.zeros((T, U, n_th, n_th), dtype=torch.float64, device=device)
            present = torch.zeros((T, U, n_th), dtype=torch.bool, device=device)
            if timing != "aoa" and sync_ns > 0:
                for c in range(C):
                    present[..., 2 + c] = True
                    prior[..., 2 + c, 2 + c] = 1.0 / sync_ns ** 2
            if timing == "tdoa":
                present[..., 4] = True
            if info == "map" and sm > 0:
                present[..., N_BASE:] = True
                if math.isfinite(sm):
                    idx = torch.arange(N_BASE, n_th, device=device)
                    prior[..., idx, idx] = 1.0 / sm ** 2
            key = "los" if info == "los" else f"map_{sm:g}"
            out[key] = peb_from_theta(J0, prior, present).cpu().numpy()
        if info == "map" and va_sigmas:
            Hv = torch.as_tensor(h_matrix_va(inp), device=device)
            Jv = theta_information(Keff, Hv)
            n_th = Hv.shape[-1]
            nlos_used = (is_nlos & ~free_path).reshape(T, U, C * P)  # [T, U, C P]
            for sv in va_sigmas:
                prior = torch.zeros((T, U, n_th, n_th), dtype=torch.float64, device=device)
                present = torch.zeros((T, U, n_th), dtype=torch.bool, device=device)
                if timing != "aoa" and sync_ns > 0:
                    for c in range(C):
                        present[..., 2 + c] = True
                        prior[..., 2 + c, 2 + c] = 1.0 / sync_ns ** 2
                if timing == "tdoa":
                    present[..., 4] = True
                if sv > 0:
                    present[..., N_BASE:] = nlos_used.repeat_interleave(3, dim=-1)
                    if math.isfinite(sv):
                        idx = torch.arange(N_BASE, n_th, device=device)
                        prior[..., idx, idx] = 1.0 / sv ** 2
                out[f"va_{sv:g}"] = peb_from_theta(Jv, prior, present).cpu().numpy()
    return out
