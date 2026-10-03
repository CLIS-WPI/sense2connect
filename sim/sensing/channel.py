"""OFDM frequency response, thermal noise, and the range-Doppler map."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.constants import speed_of_light

from sim.sensing.noise import subcarrier_noise_power_w, tx_power_w
from sim.sensing.waveform import Waveform

_SI_DELAY_S = 1e-9


def n_lags(waveform: Waveform, max_range_m: float) -> int:
    """Delay bins that cover ``max_range_m`` [m], clipped to the FFT length."""
    lag = int(math.ceil(2.0 * float(max_range_m) / speed_of_light * waveform.bandwidth_hz))
    return max(1, min(lag, waveform.n_subcarriers - 1))


def range_m(lag: int, waveform: Waveform) -> float:
    """Monostatic range [m] of a delay bin."""
    return int(lag) * speed_of_light / (2.0 * waveform.bandwidth_hz)


def doppler_hz(bin_index: int, waveform: Waveform) -> float:
    """Doppler [Hz] of a centered delay-Doppler bin. Positive approaches."""
    n = int(waveform.cpi_slots)
    centered = int(bin_index) - n // 2
    return centered * waveform.prf_hz / n


def radial_velocity_mps(bin_index: int, waveform: Waveform) -> float:
    """Approaching radial speed [m/s] from the Doppler bin.

    The ray tracer uses ``f_D = -2 v_receding / lambda``, so a positive
    Doppler bin is motion toward the radar.
    """
    return doppler_hz(bin_index, waveform) * waveform.wavelength_m / 2.0


def frequency_response(paths: object, waveform: Waveform, tx_power_dbm: float) -> torch.Tensor:
    """Channel frequency response [rx antennas, symbols, subcarriers].

    Coefficients are scaled to the configured transmit power. Paths shorter
    than 0.15 m are removed: they are the colocated self-link, which the
    ideal full-duplex assumption leaves out.
    """
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

    coefficients, delays = paths.cir(  # type: ignore[attr-defined]
        sampling_frequency=waveform.prf_hz,
        num_time_steps=waveform.cpi_slots,
        normalize_delays=False,
        out_type="torch",
    )
    coefficients = coefficients * math.sqrt(tx_power_w(tx_power_dbm))
    coefficients = _zero_self_interference(coefficients, delays)
    if coefficients.shape[-2] == 0:
        n_ant = int(coefficients.shape[1])
        return torch.zeros(
            n_ant,
            waveform.cpi_slots,
            waveform.n_subcarriers,
            dtype=torch.complex64,
            device=coefficients.device,
        )
    delay_arg = delays if delays.ndim == 5 else delays[:, None, :, None, :]
    frequencies = subcarrier_frequencies(waveform.n_subcarriers, waveform.subcarrier_spacing_hz)
    response = cir_to_ofdm_channel(frequencies, coefficients[None], delay_arg[None])
    # [batch, rx, rx antennas, tx, tx antennas, symbols, subcarriers]
    return response[0, 0, :, 0, 0]


def _zero_self_interference(coefficients: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
    """Zero path coefficients whose delay is below ``_SI_DELAY_S``."""
    if delays.ndim == 3:
        short = delays[:, None, :, None, :] < _SI_DELAY_S
    elif delays.ndim == 5:
        short = delays < _SI_DELAY_S
    else:
        raise RuntimeError(f"unexpected delay shape {tuple(delays.shape)}")
    return coefficients.masked_fill(short.unsqueeze(-1), 0)


def add_thermal_noise(
    response: torch.Tensor,
    waveform: Waveform,
    noise_figure_db: float,
    temperature_k: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Add complex thermal noise with power ``k T Δf F`` on each subcarrier."""
    power = subcarrier_noise_power_w(waveform.subcarrier_spacing_hz, noise_figure_db, temperature_k)
    sigma = math.sqrt(power / 2.0)
    real = torch.randn(response.shape, generator=generator, device=response.device, dtype=torch.float32)
    imag = torch.randn(response.shape, generator=generator, device=response.device, dtype=torch.float32)
    return response + torch.complex(real, imag) * sigma


def apply_window(response: torch.Tensor) -> torch.Tensor:
    """Hann window over slow time and over subcarriers."""
    time = torch.hann_window(response.shape[-2], periodic=False, device=response.device, dtype=torch.float32)
    frequency = torch.hann_window(response.shape[-1], periodic=False, device=response.device, dtype=torch.float32)
    return response * time[None, :, None] * frequency[None, None, :]


def delay_doppler(response: torch.Tensor, n_lag: int) -> torch.Tensor:
    """Delay-Doppler cube [rx antennas, Doppler, delay] via ``sionna.phy``."""
    from sionna.phy.channel import ofdm_to_delay_doppler_channel

    return ofdm_to_delay_doppler_channel(response, l_min=0, l_max=n_lag)


def power_map(cube: torch.Tensor) -> np.ndarray:
    """Noncoherent sum of antenna powers, shape ``[Doppler, delay]``."""
    return cube.abs().square().sum(dim=0).real.detach().cpu().numpy()


def antenna_positions_m(array: object, wavelength_m: float) -> np.ndarray:
    """Local antenna coordinates [m], shape ``[antennas, 3]``."""
    raw = array.positions(wavelength_m) if callable(array.positions) else array.positions  # type: ignore[attr-defined]
    positions = np.array(raw.numpy() if hasattr(raw, "numpy") else raw, dtype=np.float64)
    if positions.ndim != 2:
        raise RuntimeError(f"unexpected antenna position shape {positions.shape}")
    if positions.shape[0] == 3 and positions.shape[1] != 3:
        positions = positions.T
    if float(np.max(np.abs(positions))) > 0.2:
        positions = positions * float(wavelength_m)
    return positions


def rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """TR 38.901 rotation ``Rz(alpha) Ry(beta) Rx(gamma)``."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])
    return rz @ ry @ rx


def localize_cells(
    cube: torch.Tensor,
    doppler_index: torch.Tensor,
    lag_index: torch.Tensor,
    waveform: Waveform,
    positions_m: np.ndarray,
    orientation_rad: np.ndarray,
    radar_position_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal positions [m] of many range-Doppler cells.

    One steering matrix is applied to every cell. The coarse grid is the
    same 17×29 search the per-hit estimator uses, then a 9×9 refinement.
    Returns ``x``, ``y``, ``z`` [m], zenith ``theta`` [rad], and azimuth ``phi`` [rad].
    """
    if doppler_index.numel() == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty, empty
    spatial = cube[:, doppler_index, lag_index].transpose(0, 1).contiguous()
    positions = torch.as_tensor(positions_m, device=cube.device, dtype=torch.float32)
    coarse_t = torch.linspace(np.deg2rad(50.0), np.deg2rad(130.0), 17, device=cube.device)
    coarse_p = torch.linspace(np.deg2rad(-70.0), np.deg2rad(70.0), 29, device=cube.device)
    theta, phi = _refine_peaks(spatial, positions, waveform.wavelength_m, coarse_t, coarse_p)
    offset_t = torch.linspace(-np.deg2rad(4.0), np.deg2rad(4.0), 9, device=cube.device)
    offset_p = torch.linspace(-np.deg2rad(4.0), np.deg2rad(4.0), 9, device=cube.device)
    fine_t = (theta[:, None, None] + offset_t[None, :, None]).expand(-1, 9, 9).reshape(theta.shape[0], 81)
    fine_p = (phi[:, None, None] + offset_p[None, None, :]).expand(-1, 9, 9).reshape(phi.shape[0], 81)
    theta, phi = _peaks_on_grid(spatial, positions, waveform.wavelength_m, fine_t, fine_p)
    direction = torch.stack(
        (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)),
        dim=1,
    )
    rotation = torch.as_tensor(rotation_matrix(*orientation_rad.tolist()), device=cube.device, dtype=torch.float32)
    world = direction @ rotation.T
    ranges = torch.as_tensor(
        [range_m(int(lag), waveform) for lag in lag_index.detach().cpu().tolist()],
        device=cube.device,
        dtype=torch.float32,
    )
    radar = torch.as_tensor(radar_position_m, device=cube.device, dtype=torch.float32)
    point = radar[None, :] + ranges[:, None] * world
    return (
        point[:, 0].detach().cpu().numpy(),
        point[:, 1].detach().cpu().numpy(),
        point[:, 2].detach().cpu().numpy(),
        theta.detach().cpu().numpy(),
        phi.detach().cpu().numpy(),
    )


def spectrum_peaks(
    cube: torch.Tensor,
    doppler_index: torch.Tensor,
    lag_index: torch.Tensor,
    waveform: Waveform,
    positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zenith and azimuth [rad] from ``angular_delay_doppler_spectrum``.

    The transmit array is one element, so its steering vector is 1. The
    receive steering grid matches the coarse search.
    """
    from sionna.phy.isac import angular_delay_doppler_spectrum

    thetas = torch.linspace(np.deg2rad(50.0), np.deg2rad(130.0), 17, device=cube.device)
    phis = torch.linspace(np.deg2rad(-70.0), np.deg2rad(70.0), 29, device=cube.device)
    theta, phi = torch.meshgrid(thetas, phis, indexing="ij")
    positions = torch.as_tensor(positions_m, device=cube.device, dtype=torch.float32)
    steering = _steering(positions, waveform.wavelength_m, theta.reshape(-1), phi.reshape(-1))
    # [1, 1, antennas, 1, 1, doppler, delay]
    channel = cube[None, None, :, None, None, :, :]
    power = angular_delay_doppler_spectrum(
        channel,
        steering,
        torch.ones(1, 1, device=cube.device, dtype=cube.dtype),
    )
    while power.ndim > 3 and power.shape[0] == 1:
        power = power[0]
    if power.ndim != 3:
        raise RuntimeError(f"unexpected angular spectrum shape {tuple(power.shape)}")
    flat = power
    peaks = []
    for doppler, lag in zip(doppler_index.tolist(), lag_index.tolist(), strict=True):
        best = int(torch.argmax(flat[:, int(doppler), int(lag)]).item())
        peaks.append((float(theta.reshape(-1)[best]), float(phi.reshape(-1)[best])))
    if not peaks:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    stacked = np.asarray(peaks, dtype=np.float64)
    return stacked[:, 0], stacked[:, 1]


def localize(
    cube: torch.Tensor,
    doppler_index: int,
    lag_index: int,
    waveform: Waveform,
    positions_m: np.ndarray,
    orientation_rad: np.ndarray,
    radar_position_m: np.ndarray,
) -> tuple[float, float]:
    """Horizontal position [m] of one range-Doppler cell from the array snapshot."""
    spatial = cube[:, doppler_index, lag_index].detach().cpu().numpy()
    theta, phi = _peak_direction(spatial, positions_m, waveform.wavelength_m)
    local = np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=np.float64,
    )
    world = rotation_matrix(*orientation_rad.tolist()) @ local
    point = np.asarray(radar_position_m, dtype=np.float64) + range_m(lag_index, waveform) * world
    return float(point[0]), float(point[1])


def _steering(
    positions_m: torch.Tensor,
    wavelength_m: float,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Unit-norm steering vectors, shape ``[directions, antennas]``."""
    direction = torch.stack(
        (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)),
        dim=-1,
    )
    phase = (2.0 * math.pi / wavelength_m) * torch.matmul(direction, positions_m.T)
    return torch.exp(1j * phase) / math.sqrt(positions_m.shape[0])


def _refine_peaks(
    spatial: torch.Tensor,
    positions_m: torch.Tensor,
    wavelength_m: float,
    thetas: torch.Tensor,
    phis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best coarse angle of each snapshot. ``thetas`` and ``phis`` are 1-D."""
    theta, phi = torch.meshgrid(thetas, phis, indexing="ij")
    return _peaks_on_grid(spatial, positions_m, wavelength_m, theta.reshape(-1), phi.reshape(-1), shared=True)


def _peaks_on_grid(
    spatial: torch.Tensor,
    positions_m: torch.Tensor,
    wavelength_m: float,
    theta: torch.Tensor,
    phi: torch.Tensor,
    shared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax of ``|a^H y|^2``.

    ``shared=True`` uses one direction grid for every snapshot. Otherwise
    ``theta`` and ``phi`` have shape ``[snapshots, directions]``.
    """
    if shared:
        steering = _steering(positions_m, wavelength_m, theta, phi)
        scores = torch.abs(torch.matmul(steering.conj(), spatial.T)) ** 2
        best = torch.argmax(scores, dim=0)
        return theta[best], phi[best]
    direction = torch.stack(
        (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)),
        dim=-1,
    )
    phase = (2.0 * math.pi / wavelength_m) * torch.matmul(direction, positions_m.T)
    steering = torch.exp(1j * phase) / math.sqrt(positions_m.shape[0])
    scores = torch.abs(torch.einsum("nda,na->nd", steering.conj(), spatial)) ** 2
    best = torch.argmax(scores, dim=1)
    index = torch.arange(theta.shape[0], device=theta.device)
    return theta[index, best], phi[index, best]


def _peak_direction(spatial: np.ndarray, positions_m: np.ndarray, wavelength_m: float) -> tuple[float, float]:
    thetas = np.linspace(np.deg2rad(50.0), np.deg2rad(130.0), 17)
    phis = np.linspace(np.deg2rad(-70.0), np.deg2rad(70.0), 29)
    best = -1.0
    best_angles = (0.5 * math.pi, 0.0)
    for theta in thetas:
        for phi in phis:
            score = _beam(spatial, positions_m, wavelength_m, float(theta), float(phi))
            if score > best:
                best = score
                best_angles = (float(theta), float(phi))
    theta0, phi0 = best_angles
    fine_t = np.linspace(theta0 - np.deg2rad(4.0), theta0 + np.deg2rad(4.0), 9)
    fine_p = np.linspace(phi0 - np.deg2rad(4.0), phi0 + np.deg2rad(4.0), 9)
    for theta in fine_t:
        for phi in fine_p:
            score = _beam(spatial, positions_m, wavelength_m, float(theta), float(phi))
            if score > best:
                best = score
                best_angles = (float(theta), float(phi))
    return best_angles


def _beam(
    spatial: np.ndarray,
    positions_m: np.ndarray,
    wavelength_m: float,
    theta: float,
    phi: float,
) -> float:
    direction = np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )
    phase = 2.0 * math.pi / wavelength_m * positions_m @ direction
    steering = np.exp(1j * phase) / math.sqrt(positions_m.shape[0])
    return float(np.abs(np.vdot(steering, spatial)) ** 2)
