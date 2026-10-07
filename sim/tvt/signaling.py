"""Failure-aware signaling on the 10 ms timeline (TVT, human decision before the freeze).

``simulate_fa`` is the paper-1 state machine (xapp/schemes.simulate, frozen, not modified) with an
optional handover-failure model, applied identically to every scheme (configs/tvt.yaml signaling:
values and sources):
- radio link monitoring: L1 samples of the serving SNR every 10 ms, Qout / Qin quality = mean of the
  dB samples over the trailing qout_window / qin_window; every indication_interval an out-of-sync
  (quality < Qout) or in-sync (quality > Qin) indication, none in between and none while a handover
  or a re-establishment interrupts the link (TS 38.133 cl. 8.1.6, TR 36.839 cl. 5.2.1.2);
- T310 starts on N310 consecutive out-of-sync and stops on N311 consecutive in-sync indications;
  RLF on expiry (TS 38.331 cl. 5.3.10.1-3). A successful handover stops T310 and resets the counters;
- RLF: RRC re-establishment to the cell with the highest L3-filtered SNR; rate 0 (outage) for tau_re,
  then until a cell has Qout quality >= Qout; then the UE is connected to the best such cell;
- handover-command failure: a commanded switch (reasons a3, a5, trend, xapp) fails at delivery (the
  execution step) if T310 is running or the serving Qout quality is < Qout; the UE stays, no
  interruption, the scheme's TTT clock restarts;
- CHO lanes (``cho`` mask, condition of the a5 lane): execute without a command when the candidate is
  prepared; after an execution or a recovery the candidate is re-prepared t_prep later by a delivery
  that obeys the command-failure rule (retried every step).
With ``sig=None`` the function reproduces xapp.schemes.simulate exactly (CHO lanes = A5 lanes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xapp.schemes import REASON, Lanes, _slopes

CHO_REASON = 5
REEST_REASON = 6


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean over the trailing window of w samples (fewer at the start) along axis 1 of [L, T, C]."""
    c = np.cumsum(x, axis=1, dtype=np.float64)
    out = c.copy()
    out[:, w:] = c[:, w:] - c[:, :-w]
    n = np.minimum(np.arange(1, x.shape[1] + 1), w).astype(np.float64)
    return out / n[None, :, None]


def simulate_fa(lanes: Lanes, *, bandwidth_hz: float, rate_req_bps: float, max_se: float, sig: dict | None = None,
                cho: np.ndarray | None = None) -> dict[str, Any]:
    """Vectorised state machine with optional failure-aware signaling (``sig`` = configs/tvt.yaml signaling)."""
    snr = lanes.snr_db
    n_l, n_t, _ = snr.shape
    idx = np.arange(n_l)
    fa = sig is not None
    cho = np.zeros(n_l, bool) if cho is None else np.asarray(cho, bool)
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
    uses_trig = is_xapp | (lanes.planner_mask if lanes.planner_mask is not None else np.zeros(n_l, dtype=bool))
    n_veto = 0 if lanes.veto is None else lanes.veto.shape[1]
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

    if fa:
        dt = float(lanes.dt_s)
        rlm, rlf_c, rre, cc = sig["rlm"], sig["rlf"], sig["reestablishment"], sig["cho"]
        steps = lambda s: max(1, int(round(float(s) / dt)))  # noqa: E731
        q_out, q_in = float(rlm["qout_db"]), float(rlm["qin_db"])
        qout_f = trailing_mean(snr, steps(rlm["qout_window_s"]))
        qin_f = trailing_mean(snr, steps(rlm["qin_window_s"]))
        ind = steps(rlm["indication_interval_s"])
        n310, n311 = int(rlf_c["n310"]), int(rlf_c["n311"])
        t310 = int(round(float(rlf_c["t310_s"]) / dt))
        tau_re = steps(rre["tau_re_s"])
        t_prep = steps(cc["t_prep_s"])
        cmd_fail_on = bool(sig["handover_command"]["enabled"])
        oos_cnt = np.zeros(n_l, np.int64)
        ins_cnt = np.zeros(n_l, np.int64)
        t310_end = np.full(n_l, -1, np.int64)  # -1: not running
        reest = np.zeros(n_l, bool)  # re-establishment in progress (rate 0)
        reest_min_end = np.zeros(n_l, np.int64)
        prepared = cho.copy()
        prep_at = np.full(n_l, -1, np.int64)
        hof_lane, hof_step, rlf_lane, rlf_step, re_lane, re_start, re_end = [], [], [], [], [], [], []
        reest_trace = np.zeros((n_l, n_t), dtype=bool)
        open_re = np.full(n_l, -1, np.int64)  # index into re_* of the running re-establishment

    for k in range(n_t):
        # 0. re-establishment completion (failure-aware): connect to the best cell once its Qout quality >= Qout
        if fa and reest.any():
            done_t = reest & (k >= reest_min_end)
            if done_t.any():
                best = np.argmax(qout_f[:, k - 1 if k > 0 else 0, :], axis=1)
                ok = done_t & (qout_f[idx, k - 1 if k > 0 else 0, best] >= q_out)
                if ok.any():
                    for l_ in np.flatnonzero(ok):
                        re_end[open_re[l_]] = k
                    serving[ok] = best[ok]
                    reest[ok] = False
                    since_switch[ok] = 0
                    clock[ok] = 0
                    oos_cnt[ok] = 0
                    ins_cnt[ok] = 0
                    t310_end[ok] = -1
                    rec = ok & cho
                    prepared[rec] = False
                    prep_at[rec] = k + t_prep
        # 1. execute switches decided at k - 1
        go = pending >= 0
        if fa and go.any():
            qk = qout_f[idx, k - 1 if k > 0 else 0, serving]
            fail = go & (pending_reason != CHO_REASON) & cmd_fail_on & ((t310_end >= 0) | (qk < q_out))
            if fail.any():
                fl = np.flatnonzero(fail)
                hof_lane.append(fl)
                hof_step.append(np.full(fl.size, k))
                pending[fail] = -1
                pending_reason[fail] = 0
                clock[fail] = 0
                go = go & ~fail
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
            if fa:
                t310_end[go] = -1
                oos_cnt[go] = 0
                ins_cnt[go] = 0
                ce = go & cho
                prepared[ce] = False
                prep_at[ce] = k + t_prep
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
        if fa:
            interrupted = interrupted | reest
            reest_trace[:, k] = reest
        se = np.minimum(np.log2(1.0 + 10.0 ** (s_srv / 10.0)), cap)
        rate = np.where(interrupted, 0.0, scale * se)
        rate_sum += rate
        rate_trace[:, k] = rate
        interrupted_steps += interrupted
        out0[:, k] = interrupted | (s_srv < 0.0)
        outr[:, k] = interrupted | (rate < rate_req_bps)
        intr[:, k] = interrupted
        serving_trace[:, k] = serving
        # 3b. radio link monitoring, RLF, CHO re-preparation (failure-aware)
        if fa:
            if k % ind == 0:
                mon = ~interrupted
                qo = qout_f[idx, k, serving]
                qi = qin_f[idx, k, serving]
                oos = mon & (qo < q_out)
                ins = mon & (qi > q_in)
                oos_cnt = np.where(oos, oos_cnt + 1, np.where(ins, 0, oos_cnt))
                ins_cnt = np.where(ins, ins_cnt + 1, np.where(oos, 0, ins_cnt))
                start = mon & (t310_end < 0) & (oos_cnt >= n310)
                t310_end[start] = k + t310
                stop = (t310_end >= 0) & (ins_cnt >= n311)
                t310_end[stop] = -1
            rlf = (t310_end >= 0) & (k >= t310_end) & ~reest
            if rlf.any():
                rl = np.flatnonzero(rlf)
                rlf_lane.append(rl)
                rlf_step.append(np.full(rl.size, k))
                for l_ in rl:
                    open_re[l_] = len(re_lane)
                    re_lane.append(int(l_))
                    re_start.append(k)
                    re_end.append(-1)
                reest[rlf] = True
                reest_min_end[rlf] = k + 1 + tau_re
                t310_end[rlf] = -1
                oos_cnt[rlf] = 0
                ins_cnt[rlf] = 0
                pending[rlf] = -1
                pending_reason[rlf] = 0
                clock[rlf] = 0
            dl = cho & ~prepared & (prep_at >= 0) & (k >= prep_at) & ~interrupted
            if dl.any():
                qk = qout_f[idx, k, serving]
                ok = dl & ~((t310_end >= 0) | (qk < q_out)) if cmd_fail_on else dl
                prepared[ok] = True
                prep_at[ok] = -1
        # 4. decisions
        free = (pending < 0) & (k >= hold_until)
        if fa:
            free = free & ~reest
        f_s = filt[idx, serving]
        f_o = filt[idx, other]
        if n_reports and uses_trig.any():
            rel = k - lanes.e2_delay_steps
            on_report = uses_trig & (rel >= 0) & (rel % lanes.report_steps == 0) & (rel // lanes.report_steps < n_reports)
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
        if n_veto:
            relv = k - lanes.e2_delay_steps
            rv = np.where(relv >= 0, np.minimum(relv // lanes.report_steps, n_veto - 1), 0)
            vetoed = (relv >= 0) & lanes.veto[idx, rv, serving]
            free = free & ~vetoed
        cond = free & np.where(is_a5, (f_s < thr1) & (f_o > thr2), f_o - f_s > thresh)
        if fa:
            cond = cond & (~cho | prepared)  # a CHO lane executes only towards a prepared candidate
        clock = np.where(cond, clock + 1, 0)
        fire = cond & (clock >= lanes.ttt_steps)
        pending[fire] = other[fire]
        pending_reason[fire] = np.where(cho[fire] & fa, CHO_REASON, np.where(is_a5[fire], REASON["a5"], REASON["a3"]))
        clock[fire] = 0

    if ho_lane:
        hos = {"lane": np.concatenate(ho_lane), "step": np.concatenate(ho_step), "from": np.concatenate(ho_from), "to": np.concatenate(ho_to),
               "reason": np.concatenate(ho_reason)}
    else:
        hos = {key: np.zeros(0, dtype=np.int64) for key in ("lane", "step", "from", "to", "reason")}
    out = {"serving": serving_trace, "outage0": out0, "outage_req": outr, "interrupted": intr, "rate_sum": rate_sum, "rate": rate_trace,
           "interrupted_steps": interrupted_steps, "handovers": hos}
    if fa:
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)  # noqa: E731
        out["failures"] = {"hof_lane": cat(hof_lane), "hof_step": cat(hof_step), "rlf_lane": cat(rlf_lane), "rlf_step": cat(rlf_step),
                           "re_lane": np.array(re_lane, np.int64), "re_start": np.array(re_start, np.int64),
                           "re_end": np.array(re_end, np.int64)}
        out["reestablishing"] = reest_trace
    return out
