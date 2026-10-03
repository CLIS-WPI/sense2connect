"""GPU model B with the same screen geometry as the NumPy reference.

Losses are computed in float64. A batch is blockers × paths × segments.
The NumPy functions in ``sim.comm.blockage`` stay the reference.
"""

from __future__ import annotations

import math

import torch

_LOSS_FLOOR = 1e-15


def screen_loss_db(
    tx_m: torch.Tensor,
    rx_m: torch.Tensor,
    center_m: torch.Tensor,
    length_m: torch.Tensor,
    width_m: torch.Tensor,
    height_m: torch.Tensor,
    wavelength_m: float,
) -> torch.Tensor:
    """Model-B loss [dB] for every screen in a batch.

    ``tx_m``, ``rx_m`` and ``center_m`` have shape ``[..., 3]`` [m].
    The sizes are in metres and broadcast with the points. A non-finite
    entry is a fully blocked segment. Segments shorter than 1 µm return 0.
    """
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    tx = tx_m.to(dtype=torch.float64)
    rx = rx_m.to(dtype=torch.float64)
    center = center_m.to(dtype=torch.float64)
    length = length_m.to(dtype=torch.float64)
    width = width_m.to(dtype=torch.float64)
    height = height_m.to(dtype=torch.float64)
    delta = rx - tx
    distance = torch.linalg.norm(delta, dim=-1)
    normal, width_axis = _unit_horizontal(delta)
    half_w = 0.5 * (length.abs() * width_axis[..., 0].abs() + width.abs() * width_axis[..., 1].abs())
    half_h = 0.5 * height
    tx_s, tx_l, tx_z = _coords(tx, center, normal, width_axis)
    rx_s, rx_l, rx_z = _coords(rx, center, normal, width_axis)
    f_w1, f_w2 = _view_factors(tx_s, tx_l, rx_s, rx_l, -half_w, half_w, wavelength_m)
    f_h1, f_h2 = _view_factors(tx_s, tx_z, rx_s, rx_z, -half_h, half_h, wavelength_m)
    product = (f_h1 + f_h2) * (f_w1 + f_w2)
    base = 1.0 - product
    loss = torch.full_like(base, math.inf)
    ok = base > _LOSS_FLOOR
    loss = torch.where(ok, -20.0 * torch.log10(base.clamp_min(_LOSS_FLOOR)), loss)
    return torch.where(distance < 1e-6, torch.zeros_like(loss), loss)


def blocker_path_loss_db(
    starts_m: torch.Tensor,
    ends_m: torch.Tensor,
    centers_m: torch.Tensor,
    length_m: torch.Tensor,
    width_m: torch.Tensor,
    height_m: torch.Tensor,
    wavelength_m: float,
    segment_valid: torch.Tensor,
) -> torch.Tensor:
    """Sum of model-B losses [dB], shape ``[blockers, paths]``.

    ``starts_m`` and ``ends_m`` are ``[paths, segments, 3]``. Centers and
    sizes are ``[blockers, ...]``. ``segment_valid`` is ``[paths, segments]``.
    """
    # [blockers, paths, segments, 3]
    tx = starts_m[None, :, :, :]
    rx = ends_m[None, :, :, :]
    center = centers_m[:, None, None, :]
    loss = screen_loss_db(
        tx,
        rx,
        center,
        length_m[:, None, None],
        width_m[:, None, None],
        height_m[:, None, None],
        wavelength_m,
    )
    valid = segment_valid[None, :, :]
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    infinite = valid & ~torch.isfinite(loss)
    total = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)).sum(dim=-1)
    return torch.where(infinite.any(dim=-1), torch.full_like(total, math.inf), total)


def _unit_horizontal(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horiz = torch.stack((direction[..., 0], direction[..., 1], torch.zeros_like(direction[..., 0])), dim=-1)
    norm = torch.linalg.norm(horiz, dim=-1, keepdim=True)
    fallback = torch.zeros_like(horiz)
    fallback[..., 0] = 1.0
    normal = torch.where(norm < 1e-8, fallback, horiz / norm.clamp_min(1e-30))
    width_axis = torch.stack((-normal[..., 1], normal[..., 0], torch.zeros_like(normal[..., 0])), dim=-1)
    return normal, width_axis


def _coords(
    point: torch.Tensor,
    center: torch.Tensor,
    normal: torch.Tensor,
    width_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = point - center
    along = (delta * normal).sum(dim=-1)
    lateral = (delta * width_axis).sum(dim=-1)
    return along, lateral, delta[..., 2]


def _view_factors(
    origin_s: torch.Tensor,
    origin_l: torch.Tensor,
    dest_s: torch.Tensor,
    dest_l: torch.Tensor,
    edge_low: torch.Tensor,
    edge_high: torch.Tensor,
    wavelength_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_s = dest_s - origin_s
    delta_l = dest_l - origin_l
    distance = torch.hypot(delta_s, delta_l)
    safe = torch.where(delta_s.abs() >= 1e-12, delta_s, torch.ones_like(delta_s))
    alpha = (0.0 - origin_s) / safe
    crosses = (delta_s.abs() >= 1e-12) & (distance > 0.0) & (alpha > 0.0) & (alpha < 1.0)
    cross = origin_l + alpha * delta_l
    low = torch.minimum(edge_low, edge_high)
    high = torch.maximum(edge_low, edge_high)
    intersects = crosses & (cross >= low - 1e-9) & (cross <= high + 1e-9)

    def excess(offset: torch.Tensor) -> torch.Tensor:
        d_rx = torch.hypot(dest_s - 0.0, dest_l - offset)
        d_tx = torch.hypot(origin_s - 0.0, origin_l - offset)
        return d_rx + d_tx - distance

    excess_low = excess(edge_low)
    excess_high = excess(edge_high)
    sign_low = torch.where(intersects, torch.ones_like(excess_low), torch.where(excess_low <= excess_high, -torch.ones_like(excess_low), torch.ones_like(excess_low)))
    sign_high = torch.where(intersects, torch.ones_like(excess_high), torch.where(excess_low <= excess_high, torch.ones_like(excess_high), -torch.ones_like(excess_high)))
    return _knife(excess_low, sign_low, wavelength_m), _knife(excess_high, sign_high, wavelength_m)


def _knife(excess_m: torch.Tensor, sign: torch.Tensor, wavelength_m: float) -> torch.Tensor:
    excess = torch.clamp(excess_m, min=0.0)
    argument = sign * (math.pi / 2.0) * torch.sqrt(math.pi / wavelength_m * excess)
    return torch.atan(argument) / math.pi
