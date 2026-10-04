"""10 dB events at 10 ms, onset duration, and per-scheme metrics.

Events follow the M1 rules (LoS model-B loss >= 10 dB on the reference
link, short gaps < 0.5 s merged, nested in the 3 dB event) on the 10 ms
grid. The reference link is the M1/M1.5 "fixed" cell: the O-RU with the
largest unblocked power at each step. An event is actionable when the other
cell's LoS loss stays below 3 dB for the whole event.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim.comm.events import apply_hysteresis
from sim.scenes.loop import _ue_events

CLASS_OF = {"bus": "bus/truck", "truck": "bus/truck", "pedestrian": "pedestrian", "car": "car"}
CLASSES = ("bus/truck", "pedestrian", "car")
ONSET_CAP_DB = 200.0


def fixed_cell(unblocked: np.ndarray) -> np.ndarray:
    """Cell with the largest unblocked power [T, U]; ties go to the lower index."""
    return np.argmax(unblocked, axis=-1)


def link_events(
    los_loss: np.ndarray,
    los_blocker: np.ndarray,
    cell: np.ndarray,
    blocker_name: list[str],
    blocker_kind: list[str],
    blocker_lap: np.ndarray,
    dt_s: float,
    min_gap_s: float = 0.5,
    clear_db: float = 3.0,
) -> list[dict[str, Any]]:
    """10 dB LoS events on the link ``cell`` [T, U] with class, actionability and onset."""
    n_t, n_u, _ = los_loss.shape
    times = list(np.arange(n_t) * dt_s)
    rows = []
    t_idx = np.arange(n_t)
    for u in range(n_u):
        loss_u = los_loss[t_idx, u, cell[:, u]]
        blk_u = los_blocker[t_idx, u, cell[:, u]]
        for k in range(n_t):
            b = int(blk_u[k])
            rows.append(
                {
                    "ue": u,
                    "oru": "fixed",
                    "snapshot": k,
                    "los_loss_db": float(loss_u[k]),
                    "los_blocker_id": None if b < 0 else f"{blocker_name[b]}#{int(blocker_lap[k, b])}",
                    "los_blocker_kind": None if b < 0 else blocker_kind[b],
                    "strongest_loss_db": None,
                    "strongest_blocker_id": None,
                    "strongest_blocker_kind": None,
                    "power_loss_db": None,
                    "power_blocker_id": None,
                    "power_blocker_kind": None,
                }
            )
    raw = _ue_events(rows, times, [3.0, 10.0, 20.0])
    events = [ev for ev in apply_hysteresis(raw, min_gap_s, dt_s) if ev["metric"] == "los" and float(ev["threshold_db"]) == 10.0]
    out = []
    for ev in events:
        u = int(ev["ue"])
        k0, k1 = int(ev["start_snapshot"]), int(ev["end_snapshot"])
        span = np.arange(k0, k1 + 1)
        cells = cell[span, u]
        other_loss = los_loss[span, u, 1 - cells]
        series = np.clip(np.nan_to_num(los_loss[np.arange(n_t), u, cell[:, u]], posinf=ONSET_CAP_DB), 0.0, ONSET_CAP_DB)
        out.append(
            {
                "ue": u,
                "start_k": k0,
                "end_k": k1,
                "start_s": k0 * dt_s,
                "end_s": k1 * dt_s,
                "cell": int(cell[k0, u]),
                "blocker_id": ev["blocker_id"],
                "kind": ev["blocker_kind"],
                "class": CLASS_OF.get(str(ev["blocker_kind"]), "other"),
                "max_loss_db": ev["max_loss_db"],
                "actionable": bool(np.all(other_loss < clear_db)),
                "onset_s": onset_10_90(series, k0, k1, dt_s),
            }
        )
    return out


def onset_10_90(series_db: np.ndarray, k0: int, k1: int, dt_s: float) -> float:
    """Rise time [s] of the LoS loss [dB] from 10 % to 90 % of the event peak.

    The peak is the maximum in [k0, k1] (infinite loss capped at 200 dB).
    The 10 % point is the last step before the first peak crossing of 90 %
    where the loss is below 10 % of the peak.
    """
    peak = float(series_db[k0 : k1 + 1].max())
    if peak <= 0.0:
        return float("nan")
    above90 = np.flatnonzero(series_db[: k1 + 1] >= 0.9 * peak)
    above90 = above90[above90 >= _rise_start(series_db, k0, 0.1 * peak)]
    if above90.size == 0:
        return float("nan")
    k90 = int(above90[0])
    below10 = np.flatnonzero(series_db[:k90] < 0.1 * peak)
    if below10.size == 0:
        return float("nan")
    k10 = int(below10[-1])
    return (k90 - k10) * dt_s


def _rise_start(series_db: np.ndarray, k0: int, level: float) -> int:
    below = np.flatnonzero(series_db[: k0 + 1] < level)
    return int(below[-1]) if below.size else 0


def lane_summary(
    sim: dict[str, Any],
    lane: int,
    events: list[dict[str, Any]],
    oracle_rate: np.ndarray,
    dt_s: float,
    *,
    window_before_s: float = 2.0,
    interruption_before_s: float = 0.2,
    ping_pong_s: float = 1.0,
    proactive_reasons: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Metrics of one lane (one UE in one job under one scheme and parameter set)."""
    n_t = sim["outage_req"].shape[1]
    minutes = n_t * dt_s / 60.0
    hos = sim["handovers"]
    sel = hos["lane"] == lane
    steps, frm, to, reason = hos["step"][sel], hos["from"][sel], hos["to"][sel], hos["reason"][sel]
    rate = sim["rate"][lane].astype(np.float64)
    out_r = sim["outage_req"][lane]
    out_0 = sim["outage0"][lane]
    ping = sum(
        1
        for i in range(1, len(steps))
        if (steps[i] - steps[i - 1]) * dt_s <= ping_pong_s and to[i] == frm[i - 1]
    )
    considered = np.ones(len(steps), dtype=bool) if not proactive_reasons else np.isin(reason, proactive_reasons)
    matched = np.zeros(len(steps), dtype=bool)
    per_event = []
    for ev in events:
        lo = ev["start_k"] - int(round(window_before_s / dt_s))
        hit = (steps >= lo) & (steps <= ev["end_k"]) & (frm == ev["cell"]) & considered
        matched |= hit
        lead = [(ev["start_k"] - s) * dt_s for s in steps[hit] if s <= ev["start_k"]]
        i0 = max(0, ev["start_k"] - int(round(interruption_before_s / dt_s)))
        per_event.append(
            {
                "class": ev["class"],
                "actionable": ev["actionable"],
                "hit": bool(hit.any()),
                "proactive_hit": bool(lead),
                "lead_s": max(lead) if lead else None,
                "interruption_req_s": float(out_r[i0 : ev["end_k"] + 1].sum() * dt_s),
                "interruption_0_s": float(out_0[i0 : ev["end_k"] + 1].sum() * dt_s),
            }
        )
    n_cons = int(considered.sum())
    return {
        "outage_req_s_per_min": float(out_r.sum() * dt_s / minutes),
        "outage0_s_per_min": float(out_0.sum() * dt_s / minutes),
        "mean_rate_bps": float(rate.mean()),
        "p5_rate_bps": float(np.quantile(rate, 0.05)),
        "tp_loss_vs_oracle": float(1.0 - rate.mean() / max(float(oracle_rate.mean()), 1e-30)),
        "ho_per_min": float(len(steps) / minutes),
        "ping_pong": float(ping / len(steps)) if len(steps) else 0.0,
        "n_ho": int(len(steps)),
        "n_ho_considered": n_cons,
        "n_ho_matched": int((matched & considered).sum()),
        "false_ho_rate": float(1.0 - (matched & considered).sum() / n_cons) if n_cons else 0.0,
        "events": per_event,
        "interrupted_s": float(sim["interrupted_steps"][lane] * dt_s),
    }


def stateless(power: np.ndarray, snr_offset_db: float, bandwidth_hz: float, rate_req_bps: float, max_se: float) -> dict[str, np.ndarray]:
    """Outage masks and rate [T] of a scheme with no handover cost (fixed, oracle, beam)."""
    with np.errstate(divide="ignore"):
        snr = 10.0 * np.log10(power) + snr_offset_db
    se = np.minimum(np.log2(1.0 + 10.0 ** (snr / 10.0)), max_se)
    rate = bandwidth_hz * se
    return {"outage0": snr < 0.0, "outage_req": rate < rate_req_bps, "rate": rate}


def ci95(values: list[float]) -> dict[str, float | None]:
    """Mean and 95 % half-width (normal approx.) over jobs."""
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci": None, "n": 0}
    if arr.size == 1:
        return {"mean": float(arr[0]), "ci": 0.0, "n": 1}
    return {"mean": float(arr.mean()), "ci": float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size)), "n": int(arr.size)}
