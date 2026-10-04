"""DFT beam, SNR, and Shannon throughput for the M3 comm model.

Sionna 2.2 ships ``sionna.sys.PHYAbstraction`` (EESM + NR BLER tables).
That API needs a per-user MCS and transport-block size. M3 reports a
deterministic rate from SNR only: Shannon spectral efficiency capped at
TS 38.214 Table 5.1.3.1-2 MCS 27 (7.4063 bit/s/Hz).
"""

from __future__ import annotations

import math

import numpy as np

from sim.sensing.noise import noise_power_w, tx_power_w

K_BOLTZMANN = 1.380649e-23
MAX_NR_SE = 7.4063


def steering_upa(n_rows: int, n_cols: int, zenith_rad: float, azimuth_rad: float) -> np.ndarray:
    """Unit-norm DFT steering vector of an 8x8 UPA, half-wavelength spacing."""
    weights = []
    for row in range(n_rows):
        for col in range(n_cols):
            phase = math.pi * (
                col * math.sin(zenith_rad) * math.cos(azimuth_rad)
                + row * math.cos(zenith_rad)
            )
            weights.append(math.cos(phase) + 1j * math.sin(phase))
    vector = np.asarray(weights, dtype=np.complex128)
    return vector / math.sqrt(float(n_rows * n_cols))


def direction_zenith_azimuth(from_m: np.ndarray, to_m: np.ndarray) -> tuple[float, float]:
    """Zenith and azimuth [rad] of ``to - from`` in the Sionna convention."""
    delta = np.asarray(to_m, dtype=np.float64) - np.asarray(from_m, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return 0.5 * math.pi, 0.0
    zenith = math.acos(max(-1.0, min(1.0, delta[2] / distance)))
    azimuth = math.atan2(delta[1], delta[0])
    return zenith, azimuth


def dft_path_power(coefficients: np.ndarray, rx_index: int, tx_index: int, weights: np.ndarray) -> float:
    """Coherent DFT power of one link [linear], summed over paths.

    ``coefficients`` is ``[rx, rx_ant, tx, tx_ant, paths, time]``.
    """
    block = np.asarray(coefficients[rx_index, 0, tx_index, :, :, 0])
    combined = weights.conj() @ block
    return float(np.sum(np.abs(combined) ** 2))


def snr_linear(
    received_power: float,
    tx_power_dbm: float,
    bandwidth_hz: float,
    noise_figure_db: float,
    temperature_k: float,
) -> float:
    """Linear SNR. ``received_power`` is the channel power gain (unit-Tx)."""
    noise = noise_power_w(bandwidth_hz, noise_figure_db, temperature_k)
    if noise <= 0.0:
        return 0.0
    return tx_power_w(tx_power_dbm) * max(float(received_power), 0.0) / noise


def snr_db(snr: float) -> float:
    """``10 log10(snr)``. Non-positive SNR is ``-inf``."""
    if snr <= 0.0:
        return -math.inf
    return 10.0 * math.log10(snr)


def shannon_bps(snr: float, bandwidth_hz: float, max_se: float = MAX_NR_SE) -> float:
    """Throughput [bit/s] = B * min(log2(1+SNR), max_se)."""
    if snr <= 0.0:
        return 0.0
    se = min(math.log2(1.0 + snr), float(max_se))
    return float(bandwidth_hz) * se


def apply_sensing_overhead(rate_bps: float, overhead: float) -> float:
    """Reduce the rate by the sensing-symbol fraction."""
    return float(rate_bps) * (1.0 - float(overhead))
