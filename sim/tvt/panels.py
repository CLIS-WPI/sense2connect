"""Back-to-back O-RU panels (configs/tvt.yaml ``panels:``; human decision, third round before the freeze).

Every O-RU array (both cells and the monostatic radar at oru-0) is two 8x8 panels at the O-RU position with
boresight +x (yaw 0, panel sign sx = +1) and -x (yaw 180 deg, sx = -1), no tilt, each element with the 3GPP
TR 38.901 Table 7.3-1 pattern (sim/tvt/array_model.pattern_amplitude, "tr38901": 8 dBi, 65 deg, 30 dB).
Panel-local frame of the -x panel: Rz(pi), i.e. local (x, y, z) = (-x, -y, z) of the world.

The pattern is applied to the traced path angles (no geometric re-trace): for a synthetic array the solver
weights each path by one scalar per end (the element field pattern at the departure / arrival direction,
mono-polarized variant), and the element phases are exp(+j 2 pi / lambda u . r_m) with r_m the element
positions in the world frame and u the unit direction from the array towards the far end of the path
(sim/positioning/array.py). Hence a_m(panel) = a_c A_tx(u_dep) A_rx(u_arr) exp(j k u_arr . R_P r_m) with a_c
the centre coefficient of the isotropic trace. scripts/test_tvt_panels.py checks this against Sionna.

Selection ("facing"): the panel whose boresight half-space contains the target, sign(x_target - x_O-RU),
+x on a tie.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.tvt.array_model import pattern_amplitude

ROOT = Path(__file__).resolve().parents[2]
SIGNS = (1, -1)
YAW = {1: 0.0, -1: math.pi}


def config() -> dict[str, Any]:
    from sim.scenes.config import load_yaml

    return load_yaml(ROOT / "configs" / "tvt.yaml")["panels"]


def enabled(cfg: dict[str, Any] | None = None) -> bool:
    """True for the journal array model (two back-to-back panels), False for the papers' single isotropic UPA."""
    m = (cfg or config())["model"]
    if m not in ("back_to_back", "single_iso"):
        raise ValueError(f"panels.model {m!r}")
    return m == "back_to_back"


def to_local(u: Any, sx: Any) -> Any:
    """World directions [..., 3] in the frame of panel sx (broadcast): (sx ux, sx uy, uz)."""
    s = np.asarray(sx, dtype=np.float64)[..., None] if np.ndim(sx) else float(sx)
    return np.concatenate([u[..., :2] * s, u[..., 2:3]], axis=-1)


def amplitude(u: np.ndarray, sx: Any, pattern: str = "tr38901") -> np.ndarray:
    """Element field amplitude (linear) of panel sx for world unit directions u [..., 3]."""
    ul = to_local(np.asarray(u, dtype=np.float64), sx)
    az = np.arctan2(ul[..., 1], ul[..., 0])
    el = np.arcsin(np.clip(ul[..., 2], -1.0, 1.0))
    return pattern_amplitude(az, el, pattern)


def facing(target_x: Any, oru_x: Any) -> np.ndarray:
    """Panel sign (+1 / -1) facing a target at world x ``target_x`` from an O-RU at ``oru_x`` (+1 on a tie)."""
    return np.where(np.asarray(target_x) - np.asarray(oru_x) >= 0.0, 1, -1).astype(np.int64)


def rotation(sx: int) -> np.ndarray:
    """Panel-to-world rotation Rz(yaw)."""
    c, s = math.cos(YAW[int(sx)]), math.sin(YAW[int(sx)])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orientation(sx: int) -> np.ndarray:
    """Sionna / TR 38.901 orientation (alpha = yaw, beta, gamma) [rad] of panel sx."""
    return np.array([YAW[int(sx)], 0.0, 0.0])


# --- communication link -----------------------------------------------------------------------------------

def link_amplitude(u: np.ndarray, ue: np.ndarray, oru: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Panel amplitude per path for the link to the UE.

    u [T, U, C, P, 3] departure directions at the O-RU (towards the first interaction / the UE), ue [T, U, 3]
    UE positions, oru [C, 3] O-RU positions. Returns amp [T, U, C, P] and the panel sign [T, U, C].
    """
    sx = facing(ue[:, :, None, 0], oru[None, None, :, 0])  # [T, U, C]
    return amplitude(u, sx[..., None]), sx


# --- monostatic radar ---------------------------------------------------------------------------------------

def _unit(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def radar_panel_paths(packed: dict[str, np.ndarray], sx: int, positions_local: np.ndarray, rot_old: np.ndarray, wavelength_m: float,
                      *, pattern: str = "tr38901", check: bool = True) -> tuple[dict[str, np.ndarray], float]:
    """Path fields of one cached radar CPI as seen by panel sx.

    ``packed``: one split_frame / load_static dict, ``a`` [1, M, 1, 1, P, T] per RX element of the cached
    (isotropic, paper-1 orientation ``rot_old``) RX array, TX one element at the centre; angles in the world
    frame. Returns the same fields with ``a`` replaced by the panel's per-element coefficients (same local
    element positions ``positions_local`` [M, 3], world positions R_P r_m) and the max relative residual of the
    centre-coefficient fit (0 for a pure plane-wave model up to float32 rounding).
    """
    a = np.asarray(packed["a"])
    if a.shape[-2] == 0:
        return dict(packed), 0.0
    k = 2.0 * math.pi / wavelength_m
    u_r = _unit(np.asarray(packed["theta_r"], dtype=np.float64), np.asarray(packed["phi_r"], dtype=np.float64))  # [1, 1, P, 3]
    u_t = _unit(np.asarray(packed["theta_t"], dtype=np.float64), np.asarray(packed["phi_t"], dtype=np.float64))
    r_old = positions_local @ rot_old.T  # [M, 3] world
    r_new = positions_local @ rotation(sx).T
    ph_old = np.exp(1j * k * np.einsum("...pc,mc->...mp", u_r, r_old))[0, 0]  # [M, P]
    ph_new = np.exp(1j * k * np.einsum("...pc,mc->...mp", u_r, r_new))[0, 0]
    am = a[0, :, 0, 0].astype(np.complex128)  # [M, P, T]
    centre = (am * ph_old.conj()[..., None]).mean(axis=0)  # [P, T]
    resid = 0.0
    if check:
        fit = centre[None] * ph_old[..., None]
        scale = np.maximum(np.abs(am).max(axis=0, keepdims=True), 1e-30)
        live = np.abs(am).max(axis=0) > 0
        resid = float((np.abs(am - fit) / scale)[:, live].max()) if live.any() else 0.0
    g = (amplitude(u_t, sx) * amplitude(u_r, sx))[0, 0]  # [P]
    new = centre[None] * (g[:, None] * ph_new[..., None])  # [M, P, T]
    out = dict(packed)
    out["a"] = new.astype(a.dtype)[None, :, None, None]
    return out, resid


def keep_facing(detections: list[dict], sx: int, oru_x: float) -> list[dict]:
    """Detections of panel sx inside its half-space (sign(x - x_O-RU) = sx, +1 on a tie)."""
    return [d for d in detections if int(facing(float(d["x_m"]), oru_x)) == int(sx)]


# --- positioning (UE SRS on both panels) -----------------------------------------------------------------

def snapshot_amplitude(u: np.ndarray) -> np.ndarray:
    """Pattern amplitude of both panels for arrival directions u [..., 3] at the O-RU: [2, ...] (+x, -x)."""
    return np.stack([amplitude(u, s) for s in SIGNS], axis=0)
