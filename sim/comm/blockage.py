"""3GPP TR 38.901 clause 7.6.4.2 blockage model B (knife-edge screen).

Each blocker is a vertical rectangular screen. The screen faces the
incoming segment: LoS uses the O-RU to UE segment, and a reflected path
uses the segment from the last interaction point to the UE. Top and base
edges stay parallel to the ground.

The four edge terms and the loss follow clause 7.6.4.2. Distances that
enter one edge term are measured in that term's projection plane (top
view for the vertical edges, side view for the horizontal edges), so the
excess length inside the square root is non-negative. Several blockers
add in dB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_LOSS_FLOOR = 1e-15


def knife_edge_f(excess_m: float, wavelength_m: float, sign: float) -> float:
    """One edge factor from clause 7.6.4.2.

    ``excess_m`` is ``D1 + D2 - r`` [m]. ``wavelength_m`` is the carrier
    wavelength [m]. ``sign`` is +1 or -1 as specified for that edge.
    """
    excess = excess_m if excess_m > 0.0 else 0.0
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    argument = sign * (math.pi / 2.0) * math.sqrt(math.pi / wavelength_m * excess)
    return math.atan(argument) / math.pi


def loss_db_from_factors(f_h1: float, f_h2: float, f_w1: float, f_w2: float) -> float:
    """Screen loss [dB] from the four edge factors, clause 7.6.4.2.

    Returns ``+inf`` when the product of the summed factors reaches 1,
    which is the limit of the published expression.
    """
    product = (f_h1 + f_h2) * (f_w1 + f_w2)
    base = 1.0 - product
    if base <= _LOSS_FLOOR:
        return math.inf
    return -20.0 * math.log10(base)


@dataclass(frozen=True)
class ScreenLoss:
    """Model-B result for one blocker and one segment."""

    loss_db: float
    intersects: bool
    f_h1: float
    f_h2: float
    f_w1: float
    f_w2: float


def _unit_horizontal(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normal, width_axis)`` in the horizontal plane."""
    horiz = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(horiz))
    if norm < 1e-8:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        normal = horiz / norm
    width_axis = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
    return normal, width_axis


def _projected_half_width(
    length_m: float, width_m: float, width_axis: np.ndarray
) -> float:
    """Half-width [m] of an axis-aligned footprint projected onto ``width_axis``."""
    hx = 0.5 * length_m
    hy = 0.5 * width_m
    corners = ((hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy))
    lats = [dx * width_axis[0] + dy * width_axis[1] for dx, dy in corners]
    return max(abs(v) for v in lats)


def _view_factors(
    origin: np.ndarray,
    destination: np.ndarray,
    edge_offsets_m: tuple[float, float],
    wavelength_m: float,
) -> tuple[float, float, bool]:
    """Edge factors in one 2D view.

    ``origin`` is the transmitter side and ``destination`` the receiver
    side, both in view coordinates [m]. The screen lies on the line
    ``s = 0`` between the two edge offsets.
    """
    delta = destination - origin
    distance = float(np.hypot(delta[0], delta[1]))
    intersects = False
    if abs(delta[0]) >= 1e-12 and distance > 0.0:
        alpha = (0.0 - origin[0]) / delta[0]
        if 0.0 < alpha < 1.0:
            cross = float(origin[1] + alpha * delta[1])
            low = min(edge_offsets_m)
            high = max(edge_offsets_m)
            intersects = low - 1e-9 <= cross <= high + 1e-9

    excess: list[float] = []
    for offset in edge_offsets_m:
        edge = np.array([0.0, offset], dtype=np.float64)
        d_rx = float(np.hypot(*(destination - edge)))
        d_tx = float(np.hypot(*(origin - edge)))
        excess.append(d_rx + d_tx - distance)

    if intersects:
        signs = (1.0, 1.0)
    elif excess[0] <= excess[1]:
        signs = (-1.0, 1.0)
    else:
        signs = (1.0, -1.0)

    factors = tuple(
        knife_edge_f(ex, wavelength_m, sign) for ex, sign in zip(excess, signs)
    )
    return factors[0], factors[1], intersects


def screen_blockage(
    tx_m: np.ndarray,
    rx_m: np.ndarray,
    center_m: np.ndarray,
    length_m: float,
    width_m: float,
    height_m: float,
    wavelength_m: float,
) -> ScreenLoss:
    """Model-B loss of one vertical screen for the segment ``tx_m`` to ``rx_m``.

    ``length_m`` is the cuboid size along world x [m], ``width_m`` along
    world y [m], and ``height_m`` along z [m], all before the screen is
    turned to face the segment. ``center_m`` is the screen center [m].
    """
    tx = np.asarray(tx_m, dtype=np.float64)
    rx = np.asarray(rx_m, dtype=np.float64)
    center = np.asarray(center_m, dtype=np.float64)
    normal, width_axis = _unit_horizontal(rx - tx)
    half_w = _projected_half_width(length_m, width_m, width_axis)
    half_h = 0.5 * height_m

    def coords(point: np.ndarray) -> tuple[float, float, float]:
        delta = point - center
        along = float(np.dot(delta, normal))
        lateral = float(np.dot(delta, width_axis))
        return along, lateral, float(delta[2])

    tx_s, tx_l, tx_z = coords(tx)
    rx_s, rx_l, rx_z = coords(rx)
    f_w1, f_w2, hit_w = _view_factors(
        np.array([tx_s, tx_l]),
        np.array([rx_s, rx_l]),
        (-half_w, half_w),
        wavelength_m,
    )
    f_h1, f_h2, hit_h = _view_factors(
        np.array([tx_s, tx_z]),
        np.array([rx_s, rx_z]),
        (-half_h, half_h),
        wavelength_m,
    )
    return ScreenLoss(
        loss_db=loss_db_from_factors(f_h1, f_h2, f_w1, f_w2),
        intersects=hit_w and hit_h,
        f_h1=f_h1,
        f_h2=f_h2,
        f_w1=f_w1,
        f_w2=f_w2,
    )


def sum_blockage_db(losses_db: list[float]) -> float:
    """Sum per-blocker losses [dB], as clause 7.6.4.2 specifies for several screens."""
    total = 0.0
    for loss in losses_db:
        if not math.isfinite(loss):
            return math.inf
        total += loss
    return total


def linear_amplitude_gain(loss_db: float) -> float:
    """Field gain corresponding to a loss [dB]. Zero when the loss diverges."""
    if not math.isfinite(loss_db):
        return 0.0
    return 10.0 ** (-loss_db / 20.0)


def lateral_offset_sweep(
    offsets_m: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """Loss [dB] as a vehicle screen slides sideways across a 20 m link.

    The link runs from ``(0, 0, 1.5)`` m to ``(20, 0, 1.5)`` m. The screen
    center is ``(10, offset, 0.8)`` m with a 5 m by 2 m by 1.6 m cuboid.
    """
    tx = np.array([0.0, 0.0, 1.5])
    rx = np.array([20.0, 0.0, 1.5])
    losses = np.empty(len(offsets_m), dtype=np.float64)
    for index, offset in enumerate(offsets_m):
        center = np.array([10.0, float(offset), 0.8])
        losses[index] = screen_blockage(
            tx, rx, center, 5.0, 2.0, 1.6, wavelength_m
        ).loss_db
    return losses
