"""10 ms comm timeline and the handover schemes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim.comm.phy import apply_sensing_overhead, shannon_bps, snr_db, snr_linear
from sim.scenes.motion import states_at
from xapp.baselines import a3_trigger, l3_filter, pick_oracle, pick_serving_unblocked, trend_trigger
from xapp.policy import should_handover, should_return
from xapp.predict import los_loss_db, predict_link


def interpolate_series(values: np.ndarray, snap_s: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Linear interpolation of a per-snapshot series onto the comm grid."""
    return np.interp(times_s, snap_s, values)


def lerp_loss_db(loss_a: float, loss_b: float, alpha: float) -> float:
    """Interpolate model-B loss through linear power gain."""
    def gain(loss: float) -> float:
        if loss is None or not math.isfinite(float(loss)):
            return 0.0
        return 10.0 ** (-float(loss) / 10.0)

    mixed = (1.0 - alpha) * gain(loss_a) + alpha * gain(loss_b)
    if mixed <= 1e-15:
        return math.inf
    return -10.0 * math.log10(mixed)


def organize_trace(rows: list[dict[str, Any]], n_snap: int, ues: list[str], orus: list[str]) -> dict:
    """Index ue_trace rows by (ue, oru, snapshot)."""
    table: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        table[(str(row["ue"]), str(row["oru"]), int(row["snapshot"]))] = row
    missing = 0
    for ue in ues:
        for oru in orus:
            for snap in range(n_snap):
                if (ue, oru, snap) not in table:
                    missing += 1
    return {"rows": table, "missing": missing}


def link_at(table: dict, ue: str, oru: str, snap: int) -> dict[str, Any]:
    return table.get((ue, oru, snap), {"blocked_power": 0.0, "unblocked_power": 0.0, "los_loss_db": math.inf, "oru": oru})


def rsrp_dbm(blocked_power: float, tx_power_dbm: float) -> float:
    """Received power [dBm] from the channel gain and the transmit power."""
    if blocked_power <= 0.0:
        return -math.inf
    return float(tx_power_dbm) + 10.0 * math.log10(float(blocked_power))


def simulate_job(
    *,
    scheme: str,
    params: dict[str, Any],
    scenario: dict[str, Any],
    trace: dict[str, Any],
    tracks_by_snap: list[list[dict[str, Any]]] | None,
    radio: dict[str, Any],
    bandwidth_hz: float,
    wavelength_m: float,
    dt_comm_s: float,
    dt_sense_s: float,
    uses_sensing: bool,
) -> dict[str, Any]:
    """One (seed, mount, density) run of one scheme."""
    ues = [ue["name"] for ue in scenario["ues"]]
    orus = [oru["name"] for oru in scenario["orus"]]
    n_snap = int(trace["n_snapshots"])
    organized = organize_trace(trace["ue_trace"], n_snap, ues, orus)["rows"]
    snap_s = np.arange(n_snap, dtype=np.float64) * dt_sense_s
    n_comm = int(round((n_snap - 1) * dt_sense_s / dt_comm_s)) + 1
    times = np.arange(n_comm, dtype=np.float64) * dt_comm_s
    oru_pos = {oru["name"]: np.asarray(oru["position_m"], dtype=np.float64) for oru in scenario["orus"]}
    sizes = {
        kind: (float(spec["length_m"]), float(spec["width_m"]), float(spec["height_m"]))
        for kind, spec in scenario["blocker_kinds"].items()
    }
    sizes["vehicle"] = sizes.get("bus", (12.0, 2.5, 3.2))
    rng = np.random.default_rng(int(scenario["seed"]) * 17 + 3)
    sigma = float(params.get("ue_sigma_m", 1.0))
    overhead = float(params.get("overhead", 0.0)) if uses_sensing else 0.0
    tau_e2 = float(params.get("tau_e2_s", 0.0))
    tau_ho = float(params.get("tau_ho_s", 0.0))
    horizon = float(params.get("horizon_s", 1.0))
    hold_s = float(params.get("hold_s", 0.5))
    report_period = float(params.get("report_period_s", 0.1))
    block_db = float(params.get("los_block_db", 10.0))
    clear_db = float(params.get("los_clear_db", 3.0))
    outage_db = float(radio["outage_snr_db"])
    tx = float(radio["tx_power_dbm"])
    nf = float(radio["noise_figure_db"])
    temp = float(radio["temperature_k"])
    max_se = float(radio["max_spectral_efficiency"])

    serving = {ue: pick_serving_unblocked([link_at(organized, ue, oru, 0) | {"oru": oru} for oru in orus]) for ue in ues}
    previous = {ue: None for ue in ues}
    hold = {ue: 0.0 for ue in ues}
    pending: dict[str, tuple[float, str]] = {}
    filt: dict[tuple[str, str], float | None] = {(ue, oru): None for ue in ues for oru in orus}
    a3_clock = {ue: 0.0 for ue in ues}
    history = {ue: [] for ue in ues}
    ho_times: list[dict[str, Any]] = []
    outage_s = {ue: 0.0 for ue in ues}
    rates: dict[str, list[float]] = {ue: [] for ue in ues}
    last_report = -1e9
    last_tracks: list[dict[str, Any]] = []

    events_10 = [ev for ev in trace.get("ue_events", []) if float(ev.get("threshold_db", 0)) == 10.0 and ev.get("metric") == "los"]

    for step, t_s in enumerate(times):
        snap = min(n_snap - 1, int(round(t_s / dt_sense_s)))
        alpha = (t_s / dt_sense_s) - math.floor(t_s / dt_sense_s) if snap + 1 < n_snap else 0.0
        snap0 = min(n_snap - 1, int(math.floor(t_s / dt_sense_s)))
        snap1 = min(n_snap - 1, snap0 + 1)
        if tracks_by_snap is not None and abs(t_s - snap * dt_sense_s) < 0.5 * dt_comm_s:
            if t_s + 1e-9 >= last_report + report_period:
                last_tracks = [row for row in tracks_by_snap[snap] if row["confirmed"]]
                last_report = t_s
        delayed_tracks = last_tracks if t_s >= last_report + tau_e2 else []
        states = states_at(scenario, float(t_s))
        for ue in ues:
            if ue in pending and t_s + 1e-12 >= pending[ue][0]:
                previous[ue] = serving[ue]
                serving[ue] = pending[ue][1]
                hold[ue] = hold_s
                del pending[ue]
            hold[ue] = max(0.0, hold[ue] - dt_comm_s)
            powers = {}
            losses = {}
            rsrps = {}
            for oru in orus:
                a = link_at(organized, ue, oru, snap0)
                b = link_at(organized, ue, oru, snap1)
                power = (1.0 - alpha) * float(a.get("blocked_power") or 0.0) + alpha * float(b.get("blocked_power") or 0.0)
                if scheme == "beam" and serving[ue] == oru:
                    alt_a = a.get("best_available_alt_power")
                    alt_b = b.get("best_available_alt_power")
                    if alt_a is not None and alt_b is not None:
                        power = (1.0 - alpha) * float(alt_a) + alpha * float(alt_b)
                powers[oru] = power
                losses[oru] = lerp_loss_db(a.get("los_loss_db"), b.get("los_loss_db"), alpha)
                rsrps[oru] = rsrp_dbm(power, tx)
                filt[(ue, oru)] = l3_filter(filt[(ue, oru)], rsrps[oru], float(params.get("filter_a", 0.5)))
            if scheme == "oracle":
                serving[ue] = min(orus, key=lambda name: (losses[name], -powers[name], name))
            elif scheme == "none":
                pass
            elif scheme == "a3":
                other = _other(serving[ue], orus)
                s_db = filt[(ue, serving[ue])]
                o_db = filt[(ue, other)]
                if o_db - s_db > float(params["offset_db"]) + float(params["hysteresis_db"]):
                    a3_clock[ue] += dt_comm_s
                else:
                    a3_clock[ue] = 0.0
                if a3_trigger(s_db, o_db, offset_db=params["offset_db"], hysteresis_db=params["hysteresis_db"], above_since_s=a3_clock[ue], ttt_s=params["ttt_s"]):
                    _handover(ue, other, t_s, tau_e2, tau_ho, pending, ho_times, serving, "a3")
                    a3_clock[ue] = 0.0
            elif scheme == "trend":
                history[ue].append(rsrps[serving[ue]])
                window = int(round(float(params["window_s"]) / dt_comm_s))
                hist = history[ue][-max(window, 2) :]
                other = _other(serving[ue], orus)
                if trend_trigger(hist, dt_comm_s, float(params["drop_db"]), float(params["horizon_s"])):
                    if rsrps[other] > rsrps[serving[ue]]:
                        _handover(ue, other, t_s, tau_e2, tau_ho, pending, ho_times, serving, "trend")
            elif scheme == "beam":
                pass
            elif scheme == "xapp" and delayed_tracks is not None and abs(t_s - last_report - tau_e2) < dt_comm_s:
                ue_true = np.asarray(states[ue]["position_m"], dtype=np.float64)
                ue_hat = ue_true + rng.normal(0.0, sigma, size=3)
                ue_hat[2] = ue_true[2]
                preds = {}
                for oru in orus:
                    preds[oru] = predict_link(
                        oru_pos[oru],
                        ue_hat,
                        delayed_tracks,
                        sizes,
                        wavelength_m,
                        horizon,
                        dt_sense_s,
                        block_db,
                        clear_db,
                    )
                other = _other(serving[ue], orus)
                if should_handover(
                    serving[ue],
                    other,
                    preds[serving[ue]],
                    preds[other],
                    tau_e2_s=tau_e2,
                    tau_ho_s=tau_ho,
                    horizon_s=horizon,
                    hold_remaining_s=hold[ue],
                ):
                    _handover(ue, other, t_s, tau_e2, tau_ho, pending, ho_times, serving, "xapp", preds[serving[ue]])
                elif previous[ue] is not None and should_return(preds[serving[ue]], preds[previous[ue]], hold[ue]):
                    _handover(ue, previous[ue], t_s, tau_e2, tau_ho, pending, ho_times, serving, "return")
            cell = serving[ue]
            snr = snr_linear(powers[cell], tx, bandwidth_hz, nf, temp)
            rate = shannon_bps(snr, bandwidth_hz, max_se)
            rate = apply_sensing_overhead(rate, overhead)
            rates[ue].append(rate)
            if snr_db(snr) < outage_db:
                outage_s[ue] += dt_comm_s

    minutes = len(ues) * (times[-1] if len(times) else 0.0) / 60.0
    all_rates = [value for ue in ues for value in rates[ue]]
    ping = _ping_pong(ho_times, 1.0)
    false_ho = _false_handovers(ho_times, events_10, organized, ues)
    pred_stats = _prediction_stats(ho_times, events_10) if scheme == "xapp" else {"precision": None, "recall": None, "lead_s": None}
    return {
        "scheme": scheme,
        "outage_s_per_ue_min": (sum(outage_s.values()) / minutes) if minutes else 0.0,
        "mean_throughput_bps": float(np.mean(all_rates)) if all_rates else 0.0,
        "p5_throughput_bps": float(np.quantile(all_rates, 0.05)) if all_rates else 0.0,
        "handovers_per_ue_min": (len(ho_times) / minutes) if minutes else 0.0,
        "ping_pong_rate": ping,
        "false_handover_rate": false_ho,
        "n_handovers": len(ho_times),
        "prediction": pred_stats,
        "n_ue": len(ues),
        "duration_s": float(times[-1]) if len(times) else 0.0,
        "events_10": events_10,
    }


def _other(serving: str, orus: list[str]) -> str:
    rest = [name for name in orus if name != serving]
    return rest[0] if rest else serving


def _handover(ue, target, t_s, tau_e2, tau_ho, pending, ho_times, serving, reason, pred=None) -> None:
    if ue in pending:
        return
    if target == serving[ue]:
        return
    apply_at = float(t_s) + float(tau_e2) + float(tau_ho)
    pending[ue] = (apply_at, target)
    ho_times.append({"ue": ue, "t_s": apply_at, "from": serving[ue], "to": target, "reason": reason, "pred": pred})


def _ping_pong(ho_times: list[dict[str, Any]], window_s: float) -> float:
    if not ho_times:
        return 0.0
    count = 0
    by_ue: dict[str, list[dict[str, Any]]] = {}
    for row in ho_times:
        by_ue.setdefault(row["ue"], []).append(row)
    for rows in by_ue.values():
        rows = sorted(rows, key=lambda item: item["t_s"])
        for index in range(1, len(rows)):
            if rows[index]["t_s"] - rows[index - 1]["t_s"] <= window_s and rows[index]["to"] == rows[index - 1]["from"]:
                count += 1
    return count / len(ho_times)


def _false_handovers(ho_times, events, organized, ues) -> float:
    if not ho_times:
        return 0.0
    false = 0
    for ho in ho_times:
        matched = False
        for event in events:
            if event.get("ue") != ho["ue"]:
                continue
            start = float(event["start_s"])
            end = float(event["end_s"])
            if start - 1.0 <= ho["t_s"] <= end + 0.2:
                matched = True
                break
        if not matched:
            false += 1
    return false / len(ho_times)


def _prediction_stats(ho_times, events) -> dict[str, float | None]:
    if not events:
        return {"precision": None, "recall": None, "lead_s": None}
    hits = 0
    leads = []
    used = set()
    for ho in ho_times:
        for index, event in enumerate(events):
            if event.get("ue") != ho["ue"]:
                continue
            start = float(event["start_s"])
            if 0.0 <= start - ho["t_s"] <= 2.0:
                hits += 1
                leads.append(start - ho["t_s"])
                used.add(index)
                break
    precision = hits / len(ho_times) if ho_times else None
    recall = len(used) / len(events) if events else None
    return {
        "precision": precision,
        "recall": recall,
        "lead_s": None if not leads else float(np.median(leads)),
    }
