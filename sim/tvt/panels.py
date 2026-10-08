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

More than two panels (configs/tvt.yaml panels.yaw_deg_by_mount; the intersection: yaw 0, 180, 90, 270 deg,
human decision of the fourth round): a panel is identified by its yaw psi = a + (0 or 180) with the pair axis
a = psi mod 180 and the sign sx = +1 / -1. Components are stored in the frame R_a (a rotation by a about z):
u_y = (R_a^T u)_y, u_z, and sx = sign((R_a^T u)_x) of the panel's front half-space, which for a = 0 is the
third-round world convention. Link and radar select the panel with the nearest boresight azimuth
(``select``; the first listed on a tie), which for yaws (0, 180) is ``facing``. Every function keeps the
third-round arithmetic for yaw 0 / 180 (``rotation(sx)``, ``to_local``), so canyon results are unchanged.
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


def yaws_deg(mount: str | None = None, cfg: dict[str, Any] | None = None) -> list[float]:
    """Panel yaws [deg] of every O-RU (and the radar) of ``mount``: panels.yaw_deg_by_mount, else panels.yaw_deg."""
    c = cfg or config()
    by = c.get("yaw_deg_by_mount") or {}
    ys = [float(y) % 360.0 for y in (by.get(mount) if mount in by else c["yaw_deg"])]
    if any(y % 90.0 for y in ys) or len(set(ys)) != len(ys):
        raise ValueError(f"panel yaws {ys}: distinct multiples of 90 deg only (exact rotations)")
    return ys


def is_pair(yaws: list[float]) -> bool:
    """True for the two back-to-back panels +x / -x of the third round."""
    return [float(y) for y in yaws] == [0.0, 180.0]


def split(yaw: float) -> tuple[float, int]:
    """Pair axis a = yaw mod 180 [deg] and sign sx (+1 for yaw < 180, else -1)."""
    y = float(yaw) % 360.0
    return y % 180.0, (1 if y < 180.0 else -1)


def rot_z(deg: float) -> np.ndarray:
    """Exact rotation about z by a multiple of 90 deg (yaw 0 / 180: the third-round ``rotation``)."""
    y = float(deg) % 360.0
    if y in (0.0, 180.0):
        return rotation(1 if y == 0.0 else -1)
    c, s = round(math.cos(math.radians(y))), round(math.sin(math.radians(y)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def to_frame(u: np.ndarray, deg: float) -> np.ndarray:
    """World vectors [..., 3] in the frame rotated by ``deg`` about z: R^T u."""
    y = float(deg) % 360.0
    if y in (0.0, 180.0):
        return to_local(np.asarray(u, dtype=np.float64), 1 if y == 0.0 else -1)
    return np.asarray(u, dtype=np.float64) @ rot_z(y)


def amplitude_yaw(u: np.ndarray, yaw: float, pattern: str = "tr38901") -> np.ndarray:
    """Element field amplitude (linear) of the panel with yaw ``yaw`` [deg] for world unit directions u [..., 3]."""
    y = float(yaw) % 360.0
    if y in (0.0, 180.0):
        return amplitude(u, 1 if y == 0.0 else -1, pattern)
    ul = to_frame(u, y)
    az = np.arctan2(ul[..., 1], ul[..., 0])
    el = np.arcsin(np.clip(ul[..., 2], -1.0, 1.0))
    return pattern_amplitude(az, el, pattern)


def select(target_xy: Any, oru_xy: Any, yaws: list[float]) -> np.ndarray:
    """Index (into ``yaws``) of the panel with the nearest boresight azimuth to the target; first listed on a tie.

    target_xy [..., 2], oru_xy broadcastable [..., 2]. For yaws (0, 180) equal to ``facing`` (0 <-> +1, 1 <-> -1).
    """
    d = np.asarray(target_xy, dtype=np.float64)[..., :2] - np.asarray(oru_xy, dtype=np.float64)[..., :2]
    if is_pair(yaws):
        return np.where(d[..., 0] >= 0.0, 0, 1).astype(np.int64)
    b = np.array([[round(math.cos(math.radians(y))), round(math.sin(math.radians(y)))] for y in yaws], dtype=np.float64)  # [P, 2]
    score = d @ b.T  # |d| cos(az - yaw)
    return np.argmax(score >= score.max(-1, keepdims=True), axis=-1).astype(np.int64)


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


def link_amplitude_yaws(u: np.ndarray, ue: np.ndarray, oru: np.ndarray, yaws: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``link_amplitude`` for any panel set: amp [T, U, C, P], selected panel yaw [T, U, C] [deg], and the departure
    directions in the frame of the selected panel's pair axis (R_a^T u, a = yaw mod 180) for the codebook gains: the
    yaw-180 grid is the mirror of the yaw-0 grid (same symmetric codebook gains), so the third-round world directions
    are kept for a = 0. For yaws (0, 180) amp and the directions equal the third-round values exactly."""
    if is_pair(yaws):
        amp, sx = link_amplitude(u, ue, oru)
        return amp, np.where(sx > 0, 0.0, 180.0), u
    idx = select(ue[:, :, None, :2], oru[None, None, :, :2], yaws)  # [T, U, C]
    amp = np.zeros(u.shape[:-1])
    u_eff = np.array(u, dtype=np.float64, copy=True)
    for i, y in enumerate(yaws):
        m = idx == i
        if not m.any():
            continue
        amp[m] = amplitude_yaw(u[m], y)
        a, _ = split(y)
        if a:
            u_eff[m] = to_frame(u[m], a)
    return amp, np.asarray(yaws, dtype=np.float64)[idx], u_eff


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


def keep_sector(detections: list[dict], i: int, yaws: list[float], oru_xy: np.ndarray) -> list[dict]:
    """Detections of panel ``yaws[i]`` inside its sector (nearest boresight azimuth; = keep_facing for (0, 180))."""
    if not detections:
        return []
    xy = np.array([[float(d["x_m"]), float(d["y_m"])] for d in detections])
    k = select(xy, np.asarray(oru_xy, dtype=np.float64)[:2], yaws)
    return [d for d, j in zip(detections, k) if int(j) == int(i)]


def radar_panel_paths_yaw(packed: dict[str, np.ndarray], yaw: float, positions_local: np.ndarray, rot_old: np.ndarray, wavelength_m: float,
                          *, pattern: str = "tr38901", check: bool = True) -> tuple[dict[str, np.ndarray], float]:
    """``radar_panel_paths`` for the panel with yaw ``yaw`` [deg] (yaw 0 / 180: exactly radar_panel_paths(+1 / -1))."""
    y = float(yaw) % 360.0
    if y in (0.0, 180.0):
        return radar_panel_paths(packed, 1 if y == 0.0 else -1, positions_local, rot_old, wavelength_m, pattern=pattern, check=check)
    a = np.asarray(packed["a"])
    if a.shape[-2] == 0:
        return dict(packed), 0.0
    k = 2.0 * math.pi / wavelength_m
    u_r = _unit(np.asarray(packed["theta_r"], dtype=np.float64), np.asarray(packed["phi_r"], dtype=np.float64))
    u_t = _unit(np.asarray(packed["theta_t"], dtype=np.float64), np.asarray(packed["phi_t"], dtype=np.float64))
    r_old = positions_local @ rot_old.T
    r_new = positions_local @ rot_z(y).T
    ph_old = np.exp(1j * k * np.einsum("...pc,mc->...mp", u_r, r_old))[0, 0]
    ph_new = np.exp(1j * k * np.einsum("...pc,mc->...mp", u_r, r_new))[0, 0]
    am = a[0, :, 0, 0].astype(np.complex128)
    centre = (am * ph_old.conj()[..., None]).mean(axis=0)
    resid = 0.0
    if check:
        fit = centre[None] * ph_old[..., None]
        scale = np.maximum(np.abs(am).max(axis=0, keepdims=True), 1e-30)
        live = np.abs(am).max(axis=0) > 0
        resid = float((np.abs(am - fit) / scale)[:, live].max()) if live.any() else 0.0
    g = (amplitude_yaw(u_t, y, pattern) * amplitude_yaw(u_r, y, pattern))[0, 0]
    new = centre[None] * (g[:, None] * ph_new[..., None])
    out = dict(packed)
    out["a"] = new.astype(a.dtype)[None, :, None, None]
    return out, resid


# --- positioning (UE SRS on both panels) -----------------------------------------------------------------

def snapshot_amplitude(u: np.ndarray) -> np.ndarray:
    """Pattern amplitude of both panels for arrival directions u [..., 3] at the O-RU: [2, ...] (+x, -x)."""
    return np.stack([amplitude(u, s) for s in SIGNS], axis=0)


def snapshot_amplitude_yaws(u: np.ndarray, yaws: list[float]) -> np.ndarray:
    """Pattern amplitude of every panel for arrival directions u [..., 3]: [len(yaws), ...]."""
    return np.stack([amplitude_yaw(u, y) for y in yaws], axis=0)


def transverse(u: np.ndarray, a: Any) -> tuple[np.ndarray, np.ndarray]:
    """Stored-frame direction cosine (R_a^T u)_y and front sign sign((R_a^T u)_x) for pair axis a [deg] (0 or 90),
    u [..., 3], a broadcastable to u[..., 0]. a = 0: (u_y, sign u_x)."""
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    q = a == 90.0
    ty = np.where(q, -u[..., 0], u[..., 1])
    tx = np.where(q, u[..., 1], u[..., 0])
    return ty, np.where(tx >= 0.0, 1, -1)


def stored_candidates(uy: np.ndarray, uz: np.ndarray, a: Any) -> np.ndarray:
    """World unit directions [..., 2, 3] consistent with a stored component (uy, uz) of pair axis a: front (+) and
    back (-) of the axis (the panel cannot tell them apart from the element phases alone)."""
    uy = np.asarray(uy, dtype=np.float64)
    uz = np.asarray(uz, dtype=np.float64)
    ux = np.sqrt(np.clip(1.0 - uy ** 2 - uz ** 2, 0.0, None))
    q = np.broadcast_to(np.asarray(a, dtype=np.float64) == 90.0, uy.shape)
    out = []
    for s in (1.0, -1.0):
        lx, ly = s * ux, uy  # frame R_a: world = R_a (lx, ly, uz); a = 90: (x, y) = (-ly, lx)
        out.append(np.stack([np.where(q, -ly, lx), np.where(q, lx, ly), uz], -1))
    return np.stack(out, -2)


# --- result locations (isotropic: the paper-1 / paper-2 / first-round paths; panels: results/TVT/...) ------

UE_SPEED_SUFFIXES = ("_ueslow", "_uefast")  # UE-speed variants share the blocker traffic (and radar) of their base mount


def detections_dir(mount: str, density: str, seed: int, inr: float | None = None) -> Path:
    """Directory with detections_1024.json of a job (residual-SI INR ``inr`` dB, T6)."""
    base = mount
    for s in UE_SPEED_SUFFIXES:
        if mount.endswith(s):
            base = mount[: -len(s)]
    if enabled():
        tag = "det" if not inr else f"det_inr{float(inr):g}"
        return ROOT / "results" / "TVT" / "radar" / tag / base / density / f"seed_{seed}"
    if inr is not None:
        return ROOT / "results" / "TVT" / "T6" / "si" / f"inr{float(inr):g}" / mount / density / f"seed_{seed}"
    return ROOT / "results" / "cache" / mount / density / f"seed_{seed}"


def est_track_path(set_name: str, cfg: str, job: tuple) -> Path:
    """Paper-2 estimator-A track of a job (set_name: tuning | development | variant)."""
    name = f"{job[1]}_{job[2]}_{job[0]}_track.npz"
    if enabled():
        s = "development" if set_name == "variant" else set_name
        return ROOT / "results" / "TVT" / "est_A" / s / cfg / name
    if set_name == "variant":
        return ROOT / "results" / "TVT" / "T6" / "est_A" / cfg / name
    if set_name == "tuning":
        return ROOT / "results" / "TVT" / "T5" / "est_A" / "tuning" / cfg / name
    return ROOT / "results" / "P2" / "est_A" / "dev" / cfg / name


def est_dir(set_name: str, cfg: str) -> Path:
    return est_track_path(set_name, cfg, (0, "x", "x")).parent


def est_params(bw: str, timing: str) -> dict:
    """Estimator-A parameters: re-tuned on the panel measurements (scripts/tvt_est_tune.py) or paper 2's."""
    import json

    f = ROOT / "results" / "TVT" / "est_A" / "est_tuned_A.json" if enabled() else ROOT / "results" / "P2" / "est_tuned_A.json"
    return json.loads(f.read_text())["tuned"][f"{bw}|{timing}"]["params"]
