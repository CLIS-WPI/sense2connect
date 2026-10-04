"""Proactive inter-cell handover and the return rule."""

from __future__ import annotations

from typing import Any


def should_handover(
    serving: str,
    other: str,
    serving_pred: dict[str, float | None],
    other_pred: dict[str, float | None],
    *,
    tau_e2_s: float,
    tau_ho_s: float,
    horizon_s: float,
    hold_remaining_s: float,
) -> bool:
    """Hand over if serving blockage is inside the actionable window and the other cell is clear.

    Actionable window: ``[tau_E2 + tau_HO, H]``. Hold must be expired.
    The other cell must stay clear for the predicted blockage duration.
    """
    if hold_remaining_s > 0.0:
        return False
    start = serving_pred.get("start_s")
    end = serving_pred.get("end_s")
    if start is None:
        return False
    earliest = float(tau_e2_s) + float(tau_ho_s)
    if start < earliest or start > float(horizon_s):
        return False
    if other_pred.get("clear"):
        return True
    other_start = other_pred.get("start_s")
    if other_start is None:
        return True
    duration = 0.0 if end is None else max(0.0, float(end) - float(start))
    return float(other_start) > duration


def should_return(
    serving_pred: dict[str, float | None],
    previous_pred: dict[str, float | None],
    hold_remaining_s: float,
) -> bool:
    """Return after the hold if the previous cell is clear and the current one is blocked."""
    if hold_remaining_s > 0.0:
        return False
    return bool(previous_pred.get("clear")) and serving_pred.get("start_s") is not None
