"""Handover baselines with the same information where they do not use sensing."""

from __future__ import annotations

import math
from typing import Any


def a3_trigger(
    serving_rsrp_db: float,
    other_rsrp_db: float,
    *,
    offset_db: float,
    hysteresis_db: float,
    above_since_s: float,
    ttt_s: float,
) -> bool:
    """3GPP A3: other exceeds serving by offset + hysteresis for TTT."""
    if other_rsrp_db - serving_rsrp_db <= float(offset_db) + float(hysteresis_db):
        return False
    return above_since_s >= float(ttt_s)


def l3_filter(previous: float | None, sample_db: float, alpha: float) -> float:
    """One-pole L3 RSRP filter."""
    if previous is None or not math.isfinite(previous):
        return sample_db
    return (1.0 - float(alpha)) * previous + float(alpha) * sample_db


def trend_trigger(
    history_db: list[float],
    dt_s: float,
    drop_db: float,
    horizon_s: float,
) -> bool:
    """True when a linear fit of serving RSRP drops by ``drop_db`` within ``horizon_s``."""
    if len(history_db) < 2:
        return False
    times = np_times(len(history_db), dt_s)
    values = history_db
    slope = _slope(times, values)
    predicted = values[-1] + slope * float(horizon_s)
    return values[-1] - predicted >= float(drop_db)


def np_times(n: int, dt_s: float) -> list[float]:
    return [index * dt_s for index in range(n)]


def _slope(times: list[float], values: list[float]) -> float:
    mean_t = sum(times) / len(times)
    mean_v = sum(values) / len(values)
    denom = sum((t - mean_t) ** 2 for t in times)
    if denom <= 0.0:
        return 0.0
    return sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values, strict=True)) / denom


def pick_oracle(links: list[dict[str, Any]]) -> str:
    """O-RU with the smallest LoS loss. Ties break on higher blocked power."""
    from sim.comm.diversity import oracle_link

    return str(oracle_link(links)["oru"])


def pick_serving_unblocked(links: list[dict[str, Any]]) -> str:
    """O-RU with the largest unblocked power."""
    from sim.comm.diversity import serving_link

    return str(serving_link(links)["oru"])
