"""O-RU array model of paper 1: 8 x 8 UPA, lambda/2, isotropic elements, default orientation.

Element positions are those of sionna.rt.PlanarArray(8, 8, 0.5, 0.5) with zero
orientation: the y-z plane, centred, column-major (y outer, z inner). The
steering vector for a plane wave arriving from unit direction u (pointing from
the array towards the source) is s_m = exp(+j 2 pi / lambda u . r_m); the sign
is verified against the Sionna per-element coefficients by scripts/p2_trace.py.
Because the elements lie in the y-z plane, s depends only on (u_y, u_z):
directions mirrored across that plane (+x / -x) are indistinguishable.
"""

from __future__ import annotations

import numpy as np

N_ROWS = 8
N_COLS = 8
SPACING = 0.5  # wavelengths


def element_positions(wavelength_m: float) -> np.ndarray:
    """Element positions [64, 3] in metres (x = 0), Sionna order."""
    k = (np.arange(N_ROWS) - (N_ROWS - 1) / 2.0) * SPACING * wavelength_m
    y = np.repeat(k, N_COLS)  # column index outer
    z = np.tile(-k, N_ROWS)   # row index inner, top row first
    return np.stack([np.zeros_like(y), y, z], axis=-1)


def unit_direction(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """u(az, el) = (cos el cos az, cos el sin az, sin el)."""
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def direction_derivatives(az: np.ndarray, el: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """du/daz, du/del [..., 3]."""
    du_az = np.stack([-np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.zeros_like(az)], axis=-1)
    du_el = np.stack([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)], axis=-1)
    return du_az, du_el


def steering(u: np.ndarray, r: np.ndarray, wavelength_m: float) -> np.ndarray:
    """s [..., M] for unit directions u [..., 3] and element positions r [M, 3]."""
    return np.exp(1j * 2.0 * np.pi / wavelength_m * (u @ r.T))


def phase_jacobian(az: np.ndarray, el: np.ndarray, r: np.ndarray, wavelength_m: float) -> np.ndarray:
    """D [..., M, 2] = d(element phase)/d(az, el) [rad/rad]."""
    du_az, du_el = direction_derivatives(az, el)
    k = 2.0 * np.pi / wavelength_m
    return np.stack([k * (du_az @ r.T), k * (du_el @ r.T)], axis=-1)
