"""Physical array model (TVT T1) on top of the paper-1/2 8x8 UPA (sim/positioning/array.py, frozen).

Element field response for a plane wave from world direction u (unit, array -> source):
    a_m(u) = A(u) exp(j k u . (r_m + dr_m)) exp(g_m + j psi_m)
- A(u): element field-amplitude pattern, identical for all elements, in the array frame
  whose boresight is +x of the world (the paper orientation, elements in the y-z plane):
  * "iso": A = 1 (papers 1 and 2);
  * "tr38901": 3GPP TR 38.901 Table 7.3-1 single element, G_max 8 dBi, 65 deg HPBW in
    both planes, 30 dB front-to-back / side-lobe floor:
    A_dB = G_max - min(12 ((theta - 90)/65)^2 + 12 (phi/65)^2, 30) with theta = 90 deg - el and
    phi = az (degrees, phi = 0 boresight). The 30 dB floor is the "back plane": paths arriving
    from x < 0 (behind the O-RU) are 30 dB below boresight.
- impairments per element m (fixed per O-RU and run): gain error g_m (log-amplitude, Np;
  sigma_g given in dB), phase error psi_m (sigma_phi), position error dr_m (sigma_r per
  coordinate, metres; all three coordinates).
The pattern multiplies every path of an element equally, so it cannot by itself tell the two
mirror directions apart; it suppresses the paths that arrive from behind (back plane), which
is what the tracker uses as a hemisphere constraint (T4).
"""

from __future__ import annotations

import math

import numpy as np

PATTERNS = ("iso", "tr38901")


def pattern_amplitude(az, el, pattern: str, xp=np):
    """Field amplitude A(az, el) (linear) of one element; az, el world angles [rad] (boresight az = 0, el = 0)."""
    if pattern == "iso":
        return xp.ones_like(az)
    if pattern == "tr38901":
        theta = 90.0 - el * (180.0 / math.pi)
        phi = xp.remainder(az * (180.0 / math.pi) + 180.0, 360.0) - 180.0
        q = 12.0 * ((theta - 90.0) / 65.0) ** 2 + 12.0 * (phi / 65.0) ** 2
        att = xp.where(q < 30.0, q, xp.full_like(q, 30.0))
        return 10.0 ** ((8.0 - att) / 20.0)
    raise ValueError(f"unknown pattern {pattern!r}")


def pattern_log_derivatives(az, el, pattern: str, xp=np):
    """d ln A / d az, d ln A / d el [1/rad] (0 on the 30 dB floor and for iso)."""
    if pattern == "iso":
        z = xp.zeros_like(az)
        return z, z
    theta = 90.0 - el * (180.0 / math.pi)
    phi = xp.remainder(az * (180.0 / math.pi) + 180.0, 360.0) - 180.0
    q = 12.0 * ((theta - 90.0) / 65.0) ** 2 + 12.0 * (phi / 65.0) ** 2
    live = q < 30.0
    c = -math.log(10.0) / 20.0 * (180.0 / math.pi)  # d(dB)/d(deg) -> d ln A / d rad
    d_phi = 24.0 * phi / 65.0 ** 2
    d_theta = 24.0 * (theta - 90.0) / 65.0 ** 2
    daz = xp.where(live, c * d_phi, 0.0 * phi)
    del_ = xp.where(live, c * d_theta * (-1.0), 0.0 * phi)
    return daz, del_


def draw_impairments(rng: np.random.Generator, n_elem: int, sigma_g_db: float, sigma_phi_deg: float, sigma_r_m: float) -> dict[str, np.ndarray]:
    """One O-RU's element errors: g [M] (Np), psi [M] (rad), dr [M, 3] (m)."""
    return {"g": sigma_g_db * math.log(10.0) / 20.0 * rng.standard_normal(n_elem),
            "psi": math.radians(sigma_phi_deg) * rng.standard_normal(n_elem),
            "dr": sigma_r_m * rng.standard_normal((n_elem, 3))}


def response(u: np.ndarray, r: np.ndarray, wavelength_m: float, pattern: str = "iso", imp: dict | None = None) -> np.ndarray:
    """a [..., M] for unit directions u [..., 3]; element positions r [M, 3]; optional impairments (draw_impairments)."""
    az = np.arctan2(u[..., 1], u[..., 0])
    el = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    rr = r if imp is None else r + imp["dr"]
    a = np.exp(1j * 2.0 * np.pi / wavelength_m * (u @ rr.T)) * pattern_amplitude(az, el, pattern)[..., None]
    if imp is not None:
        a = a * np.exp(imp["g"] + 1j * imp["psi"])
    return a
