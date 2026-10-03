"""Range-Doppler processing on the GPU, batched over CPIs.

The NumPy CA-CFAR in ``sim.sensing.cfar`` is the reference. Coefficients
come from the channel cache: one coefficient and one Doppler per path,
expanded to the CPI by the slow-time phase.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sim.sensing.cfar import alpha_from_pfa, training_cells
from sim.sensing.noise import subcarrier_noise_power_w, tx_power_w
from sim.sensing.waveform import Waveform

_SI_DELAY_S = 1e-9


def cpi_from_paths(
    packed: dict[str, np.ndarray],
    waveform: Waveform,
    tx_power_dbm: float,
    *,
    max_paths: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild one CPI on the GPU.

    Returns coefficients ``[1, rx, antennas, tx, antennas, paths, time]``
    and delays ``[1, rx, tx, paths]``. Paths are padded with zeros to
    ``max_paths`` so every CPI in a run has the same shape.
    """
    coefficients, delays = _expand(packed, waveform, tx_power_dbm, max_paths)
    return coefficients.to(device="cuda"), delays.to(device="cuda")


def frequency_responses(
    coefficients: torch.Tensor,
    delays: torch.Tensor,
    waveform: Waveform,
) -> torch.Tensor:
    """OFDM frequency response for a batch of CPIs.

    ``coefficients`` is ``[batch, rx, rx antennas, tx, tx antennas, paths, time]``.
    The return is ``[batch, rx antennas, time, subcarriers]``.
    """
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

    if coefficients.shape[-2] == 0:
        batch = int(coefficients.shape[0])
        n_ant = int(coefficients.shape[2])
        return torch.zeros(
            batch,
            n_ant,
            waveform.cpi_slots,
            waveform.n_subcarriers,
            dtype=torch.complex64,
            device=coefficients.device,
        )
    frequencies = subcarrier_frequencies(waveform.n_subcarriers, waveform.subcarrier_spacing_hz)
    response = cir_to_ofdm_channel(frequencies, coefficients, delays)
    return response[:, 0, :, 0, 0]


def delay_doppler_batch(response: torch.Tensor, n_lag: int, window: bool) -> torch.Tensor:
    """Windowed delay-Doppler cubes, shape ``[batch, antennas, Doppler, delay]``."""
    from sionna.phy.channel import ofdm_to_delay_doppler_channel

    if window:
        time = torch.hann_window(response.shape[-2], periodic=False, device=response.device, dtype=torch.float32)
        frequency = torch.hann_window(response.shape[-1], periodic=False, device=response.device, dtype=torch.float32)
        response = response * time[None, None, :, None] * frequency[None, None, None, :]
    return ofdm_to_delay_doppler_channel(response, l_min=0, l_max=n_lag)


def power_maps(cubes: torch.Tensor) -> torch.Tensor:
    """Noncoherent power, shape ``[batch, Doppler, delay]``."""
    return cubes.abs().square().sum(dim=1).real


def ca_cfar_torch(
    power: torch.Tensor,
    *,
    guard: int,
    train: int,
    pfa: float,
    noise_applied: bool,
) -> tuple[torch.Tensor, float]:
    """CA-CFAR for a batch of maps. ``power`` is ``[batch, Doppler, delay]``."""
    if not noise_applied:
        raise RuntimeError("CFAR requires thermal noise; refusing a noiseless detection")
    if guard < 0 or train < 1:
        raise ValueError("guard must be >= 0 and train must be >= 1")
    if power.ndim != 3:
        raise ValueError("batched CA-CFAR expects [batch, Doppler, delay]")
    image = power.to(dtype=torch.float64)
    n_cells = training_cells(guard, train)
    alpha = alpha_from_pfa(n_cells, pfa)
    radius_outer = guard + train
    total = _box_sum(image, radius_outer)
    guard_sum = _box_sum(image, guard)
    noise = (total - guard_sum) / float(n_cells)
    threshold = alpha * noise
    doppler = torch.arange(image.shape[1], device=image.device)[None, :, None]
    lag = torch.arange(image.shape[2], device=image.device)[None, None, :]
    full = (
        (doppler - radius_outer >= 0)
        & (doppler + radius_outer < image.shape[1])
        & (lag - radius_outer >= 0)
        & (lag + radius_outer < image.shape[2])
    )
    mask = full & torch.isfinite(image) & (image > threshold)
    return mask, alpha


def add_noise_batch(
    response: torch.Tensor,
    waveform: Waveform,
    noise_figure_db: float,
    temperature_k: float,
    seeds: list[int],
) -> torch.Tensor:
    """Add thermal noise. Each batch row has its own seed."""
    power = subcarrier_noise_power_w(waveform.subcarrier_spacing_hz, noise_figure_db, temperature_k)
    sigma = math.sqrt(power / 2.0)
    noisy = response.clone()
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=response.device)
        generator.manual_seed(int(seed))
        real = torch.randn(response.shape[1:], generator=generator, device=response.device, dtype=torch.float32)
        imag = torch.randn(response.shape[1:], generator=generator, device=response.device, dtype=torch.float32)
        noisy[index] = response[index] + torch.complex(real, imag) * sigma
    return noisy


def _expand(
    packed: dict[str, np.ndarray],
    waveform: Waveform,
    tx_power_dbm: float,
    max_paths: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficient = np.asarray(packed["a"])
    delay = np.asarray(packed["tau"])
    doppler = np.asarray(packed["doppler"], dtype=np.float64)
    if coefficient.shape[-1] == 1:
        coefficient = coefficient[..., 0]
    if delay.shape[-1] == 1 and delay.ndim == coefficient.ndim + 0:
        delay = np.squeeze(delay, axis=-1)
    n_paths = int(coefficient.shape[-2] if coefficient.ndim == 6 else coefficient.shape[-1])
    if n_paths > max_paths:
        raise RuntimeError(f"{n_paths} paths exceed the fixed pad of {max_paths}")
    # cir() with one time step returns [rx, ant, tx, ant, paths, time].
    if coefficient.ndim == 6:
        coefficient = coefficient[..., 0]
    if delay.ndim == 4:
        delay = delay[..., 0]
    scale = math.sqrt(tx_power_w(tx_power_dbm))
    pad = max_paths - n_paths
    if pad:
        coefficient = np.pad(coefficient, [(0, 0)] * (coefficient.ndim - 1) + [(0, pad)])
        delay = np.pad(delay, [(0, 0)] * (delay.ndim - 1) + [(0, pad)])
        doppler = np.pad(doppler, [(0, 0)] * (doppler.ndim - 1) + [(0, pad)])
    coeff_t = torch.as_tensor(np.ascontiguousarray(coefficient), dtype=torch.complex64, device="cuda")
    delay_t = torch.as_tensor(np.ascontiguousarray(delay), dtype=torch.float32, device="cuda")
    doppler_t = torch.as_tensor(np.ascontiguousarray(doppler), dtype=torch.float32, device="cuda")
    short = delay_t < _SI_DELAY_S
    while short.ndim < coeff_t.ndim:
        short = short.unsqueeze(1)
    coeff_t = coeff_t.masked_fill(short, 0)
    coeff_t = coeff_t * scale
    time = torch.arange(waveform.cpi_slots, device="cuda", dtype=torch.float32) / waveform.prf_hz
    phase = torch.exp(1j * 2.0 * math.pi * doppler_t[..., None] * time)
    while phase.ndim < coeff_t.ndim + 1:
        phase = phase.unsqueeze(1)
    coeff_t = (coeff_t[..., None] * phase)[None]
    # [rx, tx, paths] -> [batch, rx, tx, paths]
    if delay_t.ndim == 3:
        delay_t = delay_t[None, :, :, :]
    elif delay_t.ndim == 2:
        delay_t = delay_t[None, :, None, :]
    else:
        raise RuntimeError(f"unexpected delay rank {delay_t.ndim}")
    return coeff_t, delay_t


def _box_sum(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Sum of each square of half-width ``radius`` on a batch of images."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    cumulative = torch.nn.functional.pad(image, (1, 0, 1, 0))
    cumulative = torch.cumsum(torch.cumsum(cumulative, dim=1), dim=2)
    height = image.shape[1]
    width = image.shape[2]
    y = torch.arange(height, device=image.device)[:, None]
    x = torch.arange(width, device=image.device)[None, :]
    y0 = torch.clamp(y - radius, 0, height)
    y1 = torch.clamp(y + radius + 1, 0, height)
    x0 = torch.clamp(x - radius, 0, width)
    x1 = torch.clamp(x + radius + 1, 0, width)
    return cumulative[:, y1, x1] - cumulative[:, y0, x1] - cumulative[:, y1, x0] + cumulative[:, y0, x0]
