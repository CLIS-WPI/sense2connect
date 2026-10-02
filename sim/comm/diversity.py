"""Link selection and post-blockage power ratios for the diversity study.

Powers are linear. Losses are in dB. A missing LoS is treated as an infinite loss.
"""

from __future__ import annotations

import math
from typing import Any

from sim.comm.blockage import headroom_db


def los_loss_db(row: dict[str, Any]) -> float:
    """LoS model-B loss [dB]. Missing or non-finite LoS is infinite."""
    value = row.get("los_loss_db")
    if value is None or not math.isfinite(float(value)):
        return math.inf
    return float(value)


def serving_link(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """O-RU with the largest unblocked power. Ties break on the smaller name."""
    if not rows:
        raise ValueError("serving_link needs at least one link")
    return min(rows, key=lambda row: (-float(row["unblocked_power"]), str(row["oru"])))


def other_link(rows: list[dict[str, Any]], serving_oru: str) -> dict[str, Any] | None:
    """The other O-RU. With more than one, the strongest unblocked remainder."""
    rest = [row for row in rows if str(row["oru"]) != serving_oru]
    if not rest:
        return None
    return min(rest, key=lambda row: (-float(row["unblocked_power"]), str(row["oru"])))


def oracle_link(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """O-RU with the smallest LoS loss. Ties break on higher post-model-B power."""
    if not rows:
        raise ValueError("oracle_link needs at least one link")
    return min(
        rows,
        key=lambda row: (los_loss_db(row), -float(row["blocked_power"]), str(row["oru"])),
    )


def power_ratio_db(numerator: float | None, denominator: float | None) -> float:
    """``10 log10(numerator / denominator)`` [dB]. Non-positive powers give ``-inf``."""
    if numerator is None or denominator is None:
        return -math.inf
    return headroom_db(float(denominator), float(numerator))
