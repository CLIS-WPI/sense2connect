"""Thermal noise on one OFDM subcarrier."""

from __future__ import annotations

import math

from scipy.constants import Boltzmann


def noise_power_w(bandwidth_hz: float, noise_figure_db: float, temperature_k: float) -> float:
    """Noise power [W] in ``bandwidth_hz`` after the noise figure."""
    figure = 10.0 ** (float(noise_figure_db) / 10.0)
    return float(Boltzmann) * float(temperature_k) * float(bandwidth_hz) * figure


def tx_power_w(tx_power_dbm: float) -> float:
    """Convert a transmit power [dBm] to watts."""
    return 10.0 ** ((float(tx_power_dbm) - 30.0) / 10.0)


def subcarrier_noise_power_w(
    subcarrier_spacing_hz: float,
    noise_figure_db: float,
    temperature_k: float,
) -> float:
    """Complex-baseband noise power [W] on one subcarrier, ``k T Δf F``."""
    if subcarrier_spacing_hz <= 0.0 or not math.isfinite(subcarrier_spacing_hz):
        raise ValueError("subcarrier spacing must be positive")
    return noise_power_w(subcarrier_spacing_hz, noise_figure_db, temperature_k)
