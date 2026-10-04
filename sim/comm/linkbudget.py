"""Link budget, margin sweep, SNR and rate for every margin in one pass.

The ray-traced power ``P`` of a link is the channel power gain summed over
the 8x8 O-RU array with unit transmit power (MRT over all paths), with
isotropic elements and a single isotropic UE element. The default budget is
3GPP TR 38.802 V14.2.0 Table A.2.1-1 (dense urban, micro layer, above
6 GHz): BS transmit power 33 dBm, UE noise figure 13 dB (baseline). Cable,
body, implementation and shadow-fading losses are "reported by companies"
in TR 38.830 V17.0.0 Table A.3, so the default carries 0 dB of them; the
margin sweep stands in for them.

A margin point ``M`` [dB] is reached with a common extra path loss on both
cells: ``L(M) = M_ref - M``, where ``M_ref`` is the median (tuning jobs,
UEs, 10 ms steps) of the best-cell unblocked SNR above ``SNR_req`` under
the default budget. Margin is additive in dB, so SNR for every margin is
``SNR_ref - L``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim.comm.phy import MAX_NR_SE

THERMAL_DBM_PER_HZ = -174.0


def noise_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    """Thermal noise [dBm] at 290 K plus the noise figure."""
    return THERMAL_DBM_PER_HZ + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def snr_req_db(rate_bps: float, bandwidth_hz: float) -> float:
    """Shannon SNR [dB] needed for ``rate_bps`` on ``bandwidth_hz`` (no overhead)."""
    se = rate_bps / bandwidth_hz
    if se > MAX_NR_SE:
        raise ValueError("service rate exceeds the MCS 27 cap")
    return 10.0 * math.log10(2.0**se - 1.0)


def snr_ref_db(power: np.ndarray, tx_power_dbm: float, bandwidth_hz: float, noise_figure_db: float) -> np.ndarray:
    """SNR [dB] at the reference budget. Zero power is ``-inf``."""
    with np.errstate(divide="ignore"):
        gain_db = 10.0 * np.log10(np.asarray(power, dtype=np.float64))
    return tx_power_dbm + gain_db - noise_dbm(bandwidth_hz, noise_figure_db)


def extra_loss_db(margins_db: list[float], margin_ref_db: float) -> np.ndarray:
    """Extra path loss [dB] that moves the reference margin to each margin point."""
    return margin_ref_db - np.asarray(margins_db, dtype=np.float64)


def rate_bps(snr_db: Any, bandwidth_hz: float, overhead: Any = 0.0, max_se: float = MAX_NR_SE, xp: Any = np) -> Any:
    """``B (1 - overhead) min(log2(1 + SNR), max_se)`` [bit/s]; ``-inf`` dB gives 0."""
    snr = 10.0 ** (snr_db / 10.0)
    se = xp.minimum(xp.log2(1.0 + snr), max_se)
    return bandwidth_hz * (1.0 - overhead) * se


def sensing_overhead(symbols_fraction: float, duty_cycle: float) -> float:
    """Comm resources spent on sensing: (sensing symbols per slot / 14) x CPI duty cycle."""
    return float(symbols_fraction) * float(duty_cycle)


def snr_all_margins(snr_ref: np.ndarray, losses_db: np.ndarray, device: str | None = None) -> Any:
    """SNR [dB] for every margin: ``[M, ...]`` = ``snr_ref[None] - L[:, None]``.

    With ``device`` the result is a float64 torch tensor on that device.
    """
    if device is None:
        return snr_ref[None, ...] - losses_db.reshape((-1,) + (1,) * snr_ref.ndim)
    import torch

    ref = torch.as_tensor(snr_ref, device=device, dtype=torch.float64)
    loss = torch.as_tensor(losses_db, device=device, dtype=torch.float64)
    return ref[None, ...] - loss.reshape((-1,) + (1,) * ref.ndim)
