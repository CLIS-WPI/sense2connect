"""Handover schemes on the 10 ms timeline, batched over lanes.

A lane is one (UE, job, parameter set). All lanes step together through
the T comm steps; every per-step operation is a vector operation over the
lanes, so a whole tuning grid runs in one pass. ``simulate_scalar`` is the
plain-Python reference with the same rules; tests require identical
handovers and outage.

Rules (both implementations):
- Decision at step k switches the cell at step k + 1. The switch interrupts
  the link for ``tau_ho`` (rate 0, counted as outage) for every scheme.
- A3 (all of a3, trend, xapp): L3-filtered SNR per cell (one-pole, ``a``);
  other - serving > offset + hysteresis for TTT steps -> handover. Blocked
  while an xApp hold is active.
- trend: on top of A3; least-squares slope of the raw serving SNR over the
  last ``window`` steps since the last switch predicts a drop >= ``drop``
  within ``horizon`` and the other cell is stronger now -> handover.
- xapp: on top of A3; at step ``report*10 + d`` (d = E2 loop delay in steps)
  the trigger table of that report is read for the serving cell; if true and
  the other cell's filtered SNR exceeds serving - ``block_db`` -> handover,
  then a hold that blocks every further handover until the later of
  switch + ``hold`` and the predicted end of the blockage window (so A3 does
  not hand back before the predicted blockage).
Priority inside one step: xapp, trend, A3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMES = ("a3", "trend", "xapp")
REASON = {"a3": 1, "trend": 2, "xapp": 3, "a5": 4}


@dataclass
class Lanes:
    """Inputs of a batch of lanes. Times are in steps of ``dt_s``."""

    snr_db: np.ndarray  # [L, T, 2]
    scheme: np.ndarray  # [L] int: 1 a3, 2 trend, 3 xapp
    offset_db: np.ndarray
    hysteresis_db: np.ndarray
    ttt_steps: np.ndarray
    filter_a: np.ndarray
    window_steps: np.ndarray
    drop_db: np.ndarray
    trend_horizon_s: np.ndarray
    hold_steps: np.ndarray
    tau_ho_steps: np.ndarray
    e2_delay_steps: np.ndarray
    overhead: np.ndarray
    initial_cell: np.ndarray  # [L]
    trigger: np.ndarray | None = None  # [L, R, 2] bool (xapp lanes)
    trigger_end_steps: np.ndarray | None = None  # [L, R, 2] predicted window end, steps after the report
    block_db: float = 10.0
    report_steps: int = 10
    dt_s: float = 0.01
    extra: dict[str, Any] = field(default_factory=dict)
    # A5 (added after the cost-aware-oracle result): serving filtered SNR < a5_thr1 AND
    # neighbour filtered SNR > a5_thr2 for TTT steps [dB, SNR scale]; used on a5 lanes only
    a5_thr1: np.ndarray | None = None
    a5_thr2: np.ndarray | None = None


def simulate(lanes: Lanes, *, bandwidth_hz: float, rate_req_bps: float, max_se: float) -> dict[str, Any]:
    """Vectorised state machine. Returns per-lane totals, masks and the handover list."""
    snr = lanes.snr_db
    n_l, n_t, _ = snr.shape
    idx = np.arange(n_l)
    serving = lanes.initial_cell.astype(np.int64).copy()
    filt = snr[:, 0, :].copy()
    clock = np.zeros(n_l, dtype=np.int64)
    pending = np.full(n_l, -1, dtype=np.int64)
    pending_reason = np.zeros(n_l, dtype=np.int64)
    interrupt_until = np.zeros(n_l, dtype=np.int64)
    hold_until = np.zeros(n_l, dtype=np.int64)
    pending_hold_end = np.zeros(n_l, dtype=np.int64)
    since_switch = np.zeros(n_l, dtype=np.int64)
    w_max = int(max(2, lanes.window_steps.max()))
    hist = np.zeros((n_l, w_max))
    a = lanes.filter_a
    thresh = lanes.offset_db + lanes.hysteresis_db
    is_a5 = lanes.scheme == REASON["a5"]
    thr1 = lanes.a5_thr1 if lanes.a5_thr1 is not None else np.full(n_l, -np.inf)
    thr2 = lanes.a5_thr2 if lanes.a5_thr2 is not None else np.full(n_l, np.inf)
    is_trend = lanes.scheme == REASON["trend"]
    is_xapp = lanes.scheme == REASON["xapp"]
    n_reports = 0 if lanes.trigger is None else lanes.trigger.shape[1]

    serving_trace = np.zeros((n_l, n_t), dtype=np.int8)
    out0 = np.zeros((n_l, n_t), dtype=bool)
    outr = np.zeros((n_l, n_t), dtype=bool)
    intr = np.zeros((n_l, n_t), dtype=bool)
    rate_sum = np.zeros(n_l)
    rate_trace = np.zeros((n_l, n_t), dtype=np.float32)
    interrupted_steps = np.zeros(n_l, dtype=np.int64)
    ho_lane, ho_step, ho_from, ho_to, ho_reason = [], [], [], [], []
    cap = float(max_se)
    scale = bandwidth_hz * (1.0 - lanes.overhead)

    for k in range(n_t):
        # 1. execute switches decided at k - 1
        go = pending >= 0
        if go.any():
            lanes_go = np.flatnonzero(go)
            ho_lane.append(lanes_go)
            ho_step.append(np.full(lanes_go.size, k))
            ho_from.append(serving[lanes_go].copy())
            ho_to.append(pending[lanes_go].copy())
            ho_reason.append(pending_reason[lanes_go].copy())
            serving[go] = pending[go]
            interrupt_until[go] = k + lanes.tau_ho_steps[go]
            xh = go & (pending_reason == REASON["xapp"])
            hold_until[xh] = np.maximum(k + lanes.hold_steps[xh], pending_hold_end[xh])
            since_switch[go] = 0
            clock[go] = 0
            pending[go] = -1
            pending_reason[go] = 0
        # 2. measurements
        s_now = snr[:, k, :]
        if k > 0:
            filt = (1.0 - a)[:, None] * filt + a[:, None] * s_now
        other = 1 - serving
        s_srv = s_now[idx, serving]
        hist[:, k % w_max] = s_srv
        since_switch += 1
        # 3. accounting
        interrupted = k < interrupt_until
        se = np.minimum(np.log2(1.0 + 10.0 ** (s_srv / 10.0)), cap)
        rate = np.where(interrupted, 0.0, scale * se)
        rate_sum += rate
        rate_trace[:, k] = rate
        interrupted_steps += interrupted
        out0[:, k] = interrupted | (s_srv < 0.0)
        outr[:, k] = interrupted | (rate < rate_req_bps)
        intr[:, k] = interrupted
        serving_trace[:, k] = serving
        # 4. decisions
        free = (pending < 0) & (k >= hold_until)
        f_s = filt[idx, serving]
        f_o = filt[idx, other]
        if n_reports and is_xapp.any():
            rel = k - lanes.e2_delay_steps
            on_report = is_xapp & (rel >= 0) & (rel % lanes.report_steps == 0) & (rel // lanes.report_steps < n_reports)
            if on_report.any():
                lanes_r = np.flatnonzero(on_report & free)
                if lanes_r.size:
                    r_idx = rel[lanes_r] // lanes.report_steps
                    trig = lanes.trigger[lanes_r, r_idx, serving[lanes_r]]
                    ok = trig & (f_o[lanes_r] > f_s[lanes_r] - lanes.block_db)
                    chosen = lanes_r[ok]
                    pending[chosen] = other[chosen]
                    pending_reason[chosen] = REASON["xapp"]
                    ends = lanes.trigger_end_steps[lanes_r, r_idx, serving[lanes_r]][ok]
                    pending_hold_end[chosen] = r_idx[ok] * lanes.report_steps + ends
        free = free & (pending < 0)
        if is_trend.any():
            cand = free & is_trend & (since_switch >= 2)
            if cand.any():
                n_w = np.minimum(lanes.window_steps, since_switch)
                slope = _slopes(hist, k, n_w, w_max, lanes.dt_s)
                fire = cand & (-slope * lanes.trend_horizon_s >= lanes.drop_db) & (s_now[idx, other] > s_srv) & (n_w >= 2)
                pending[fire] = other[fire]
                pending_reason[fire] = REASON["trend"]
        free = free & (pending < 0)
        cond = free & np.where(is_a5, (f_s < thr1) & (f_o > thr2), f_o - f_s > thresh)
        clock = np.where(cond, clock + 1, 0)
        fire = cond & (clock >= lanes.ttt_steps)
        pending[fire] = other[fire]
        pending_reason[fire] = np.where(is_a5[fire], REASON["a5"], REASON["a3"])
        clock[fire] = 0

    if ho_lane:
        hos = {
            "lane": np.concatenate(ho_lane),
            "step": np.concatenate(ho_step),
            "from": np.concatenate(ho_from),
            "to": np.concatenate(ho_to),
            "reason": np.concatenate(ho_reason),
        }
    else:
        hos = {key: np.zeros(0, dtype=np.int64) for key in ("lane", "step", "from", "to", "reason")}
    return {
        "serving": serving_trace,
        "outage0": out0,
        "outage_req": outr,
        "interrupted": intr,
        "rate_sum": rate_sum,
        "rate": rate_trace,
        "interrupted_steps": interrupted_steps,
        "handovers": hos,
    }


def _slopes(hist: np.ndarray, k: int, n_w: np.ndarray, w_max: int, dt_s: float) -> np.ndarray:
    """Least-squares slope [dB/s] over the last ``n_w`` samples ending at step k."""
    ages = np.arange(w_max)  # 0 = newest
    pos = (k - ages) % w_max
    values = hist[:, pos]  # [L, w_max], newest first
    mask = ages[None, :] < n_w[:, None]
    times = -ages[None, :] * dt_s
    n = mask.sum(1)
    n_safe = np.maximum(n, 1)
    mean_t = (times * mask).sum(1) / n_safe
    mean_v = (values * mask).sum(1) / n_safe
    dt = (times - mean_t[:, None]) * mask
    denom = (dt * dt).sum(1)
    num = (dt * (values - mean_v[:, None]) * mask).sum(1)
    return np.where(denom > 0.0, num / np.where(denom > 0.0, denom, 1.0), 0.0)


def simulate_scalar(lanes: Lanes, lane: int, *, bandwidth_hz: float, rate_req_bps: float, max_se: float) -> dict[str, Any]:
    """Plain-Python reference for one lane."""
    snr = lanes.snr_db[lane]
    n_t = snr.shape[0]
    serving = int(lanes.initial_cell[lane])
    filt = [float(snr[0, 0]), float(snr[0, 1])]
    clock = 0
    pending: tuple[int, int] | None = None
    interrupt_until = 0
    hold_until = 0
    hold_end = 0
    history: list[float] = []
    a = float(lanes.filter_a[lane])
    scheme = int(lanes.scheme[lane])
    d = int(lanes.e2_delay_steps[lane])
    out0 = np.zeros(n_t, dtype=bool)
    outr = np.zeros(n_t, dtype=bool)
    rate_sum = 0.0
    hos = []
    for k in range(n_t):
        if pending is not None:
            hos.append((k, serving, pending[0], pending[1]))
            serving = pending[0]
            interrupt_until = k + int(lanes.tau_ho_steps[lane])
            if pending[1] == REASON["xapp"]:
                hold_until = max(k + int(lanes.hold_steps[lane]), hold_end)
            history = []
            clock = 0
            pending = None
        if k > 0:
            filt = [(1.0 - a) * filt[c] + a * float(snr[k, c]) for c in range(2)]
        other = 1 - serving
        s_srv = float(snr[k, serving])
        history.append(s_srv)
        interrupted = k < interrupt_until
        se = min(math.log2(1.0 + 10.0 ** (s_srv / 10.0)), float(max_se))
        rate = 0.0 if interrupted else bandwidth_hz * (1.0 - float(lanes.overhead[lane])) * se
        rate_sum += rate
        out0[k] = interrupted or s_srv < 0.0
        outr[k] = interrupted or rate < rate_req_bps
        free = k >= hold_until
        if free and scheme == REASON["xapp"] and lanes.trigger is not None:
            rel = k - d
            if rel >= 0 and rel % lanes.report_steps == 0 and rel // lanes.report_steps < lanes.trigger.shape[1]:
                r = rel // lanes.report_steps
                if bool(lanes.trigger[lane, r, serving]) and filt[other] > filt[serving] - lanes.block_db:
                    pending = (other, REASON["xapp"])
                    hold_end = r * lanes.report_steps + int(lanes.trigger_end_steps[lane, r, serving])
        if free and pending is None and scheme == REASON["trend"] and len(history) >= 2:
            w = min(int(lanes.window_steps[lane]), len(history))
            if w >= 2:
                values = history[-w:]
                times = [-(w - 1 - i) * lanes.dt_s for i in range(w)]
                mt = sum(times) / w
                mv = sum(values) / w
                den = sum((t - mt) ** 2 for t in times)
                slope = 0.0 if den <= 0.0 else sum((t - mt) * (v - mv) for t, v in zip(times, values)) / den
                if -slope * float(lanes.trend_horizon_s[lane]) >= float(lanes.drop_db[lane]) and float(snr[k, other]) > s_srv:
                    pending = (other, REASON["trend"])
        if free and pending is None:
            if scheme == REASON["a5"]:
                fires = filt[serving] < float(lanes.a5_thr1[lane]) and filt[other] > float(lanes.a5_thr2[lane])
            else:
                fires = filt[other] - filt[serving] > float(lanes.offset_db[lane]) + float(lanes.hysteresis_db[lane])
            if fires:
                clock += 1
            else:
                clock = 0
            if clock >= int(lanes.ttt_steps[lane]):
                pending = (other, REASON["a5"] if scheme == REASON["a5"] else REASON["a3"])
                clock = 0
        else:
            clock = 0
    return {"outage0": out0, "outage_req": outr, "rate_sum": rate_sum, "handovers": hos}
