"""M3 v1 (kept for the record): tune the xApp on seeds 101–105, evaluate on 1001–1010."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.diversity import los_loss_db, oracle_link, serving_link
from sim.comm.events import apply_hysteresis
from sim.scenes.config import load_yaml
from sim.scenes.loop import _ue_events
from sim.scenes.traffic import prepare_scenario
from sim.sensing.metrics import CLASSES, blocker_class
from xapp.simulate import simulate_job
from xapp.tracks import load_detections, replay_map

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")


def main() -> None:
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tuning = [int(s) for s in seeds["tuning"]]
    evaluation = [int(s) for s in seeds["evaluation"]]
    print("load jobs", flush=True)
    tune_jobs = _jobs(raw, cfg, tuning)
    print("oracle check", flush=True)
    oracle_table = _oracle_check(tune_jobs)
    print("tune", flush=True)
    chosen = _tune(tune_jobs, cfg)
    print("evaluate", flush=True)
    eval_jobs = _jobs(raw, cfg, evaluation)
    eval_rows = _evaluate(eval_jobs, cfg, chosen)
    delay_rows = _delay_sweep(eval_jobs, cfg, chosen)
    ho_sweep = _ho_sweep(eval_jobs, cfg, chosen)
    text = _report(raw, cfg, chosen, oracle_table, eval_rows, delay_rows, ho_sweep)
    out = ROOT / "results" / "M3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"chosen": chosen, "eval": eval_rows, "delay": delay_rows}, indent=2) + "\n")
    print(text)


def _jobs(raw, cfg, seeds: list[int]) -> list[dict[str, Any]]:
    jobs = []
    for seed in seeds:
        for mount in MOUNTS:
            for density in DENSITIES:
                jobs.append(_load_job(raw, cfg, seed, mount, density))
    return jobs


def _load_job(raw, cfg, seed: int, mount: str, density: str) -> dict[str, Any]:
    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    trace_path = directory / "comm_trace.json"
    if not trace_path.exists():
        raise SystemExit(f"missing {trace_path}; run scripts/cache_comm.py")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    scenario = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    detections = load_detections(directory)
    tracks = {}
    spec = raw["sensing_radar"]
    walls = [float(v) for v in spec["wall_y_m"]]
    tuned = cfg["sensing"]["map_tracker"]
    for budget, det in cfg["sensing"]["budgets"].items():
        track_path = directory / f"tracks_budget_{int(budget)}.json"
        if track_path.exists():
            print(f"  tracks {mount} {density} {seed} budget {budget} cached", flush=True)
            tracks[int(budget)] = json.loads(track_path.read_text(encoding="utf-8"))
            continue
        print(f"  tracks {mount} {density} {seed} budget {budget}", flush=True)
        tracks[int(budget)] = replay_map(
            detections,
            train=int(det["train"]),
            pfa=float(det["pfa"]),
            eps_m=float(det["eps_m"]),
            ghost_association_m=float(det["ghost_association_m"]),
            walls=walls,
            spec=spec,
            lanes=list(raw["lanes"]),
            sidewalks=list(raw["sidewalks"]),
            tuned=tuned,
        )
        track_path.write_text(json.dumps(tracks[int(budget)]) + "\n", encoding="utf-8")
    vel = _along_lane_error(detections, tracks[4] if 4 in tracks else next(iter(tracks.values())))
    del detections
    return {
        "seed": seed,
        "mount": mount,
        "density": density,
        "scenario": scenario,
        "trace": trace,
        "tracks": tracks,
        "velocity": vel,
        "wavelength_m": 299792458.0 / float(raw["carrier_hz"]),
        "bandwidth_hz": int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"])),
    }


def _along_lane_error(detections, tracks_by_snap) -> dict[str, float]:
    errors = []
    for frame, tracks in zip(detections["frames"], tracks_by_snap, strict=True):
        confirmed = [row for row in tracks if row["confirmed"]]
        for target in frame["ground_truth"]:
            best = None
            for track in confirmed:
                dist = math.hypot(track["x_m"] - float(target["x_m"]), track["y_m"] - float(target["y_m"]))
                if best is None or dist < best[0]:
                    best = (dist, track)
            if best is None or best[0] > 6.0:
                continue
            errors.append(abs(best[1]["vx_mps"] - float(target["vx_mps"])))
    if not errors:
        return {"median": math.nan, "p90": math.nan, "n": 0}
    array = np.asarray(errors, dtype=np.float64)
    return {"median": float(np.median(array)), "p90": float(np.quantile(array, 0.9)), "n": int(array.size)}


def _oracle_check(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for mount in MOUNTS:
        for density in DENSITIES:
            chosen = [job for job in jobs if job["mount"] == mount and job["density"] == density]
            fixed = [_event_rate(job, "fixed") for job in chosen]
            oracle = [_event_rate(job, "oracle") for job in chosen]
            table.append(
                {
                    "mount": mount,
                    "density": density,
                    "fixed": float(np.mean(fixed)) if fixed else 0.0,
                    "oracle": float(np.mean(oracle)) if oracle else 0.0,
                    "m15_fixed": _m15_fixed(mount, density)[0],
                    "m15_oracle": _m15_fixed(mount, density)[1],
                }
            )
    return table


def _m15_fixed(mount: str, density: str) -> tuple[float, float]:
    # M1.5 report, tuning seeds, events per UE-minute.
    values = {
        ("lamppost", "low"): (6.10, 2.00),
        ("lamppost", "high"): (11.90, 6.30),
        ("facade", "low"): (6.90, 2.10),
        ("facade", "high"): (12.70, 6.20),
    }
    return values[(mount, density)]


def _event_rate(job: dict[str, Any], mode: str) -> float:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in job["trace"]["ue_trace"]:
        grouped.setdefault((str(row["ue"]), int(row["snapshot"])), []).append(row)
    ues = sorted({ue for ue, _ in grouped})
    n_snap = int(job["trace"]["n_snapshots"])
    times = [index * 0.1 for index in range(n_snap)]
    synthetic = []
    for ue in ues:
        for snap in range(n_snap):
            links = grouped.get((ue, snap), [])
            if not links:
                loss = math.inf
                kind = None
                blocker = None
            else:
                chosen = serving_link(links) if mode == "fixed" else oracle_link(links)
                loss = los_loss_db(chosen)
                kind = chosen.get("los_blocker_kind")
                blocker = chosen.get("los_blocker_id")
            synthetic.append(
                {
                    "ue": ue,
                    "oru": "selected",
                    "snapshot": snap,
                    "los_loss_db": loss,
                    "los_blocker_id": blocker,
                    "los_blocker_kind": kind,
                    "strongest_loss_db": loss,
                    "strongest_blocker_id": blocker,
                    "strongest_blocker_kind": kind,
                    "power_loss_db": loss,
                    "power_blocker_id": blocker,
                    "power_blocker_kind": kind,
                }
            )
    raw = _ue_events(synthetic, times, [3.0, 10.0, 20.0])
    events = [ev for ev in apply_hysteresis(raw, 0.5, 0.1) if ev.get("metric") == "los" and float(ev["threshold_db"]) == 10.0]
    minutes = len(ues) * 60.0 / 60.0
    return len(events) / minutes


def _base_params(cfg) -> dict[str, Any]:
    return {
        "ue_sigma_m": float(cfg["ue_position"]["sigma_m"]),
        "overhead": float(cfg["sensing"]["overhead"]),
        "tau_e2_s": float(cfg["e2"]["loop_delay_s"]),
        "tau_ho_s": float(cfg["e2"]["tau_ho_s"]),
        "report_period_s": float(cfg["e2"]["report_period_s"]),
        "los_block_db": float(cfg["xapp"]["los_block_db"]),
        "los_clear_db": float(cfg["xapp"]["los_clear_db"]),
        "filter_a": float(cfg["a3"]["filter_a"]),
    }


def _run(job, scheme, params, cfg, budget: int | None = None):
    tracks = None if budget is None else job["tracks"][int(budget)]
    return simulate_job(
        scheme=scheme,
        params=params,
        scenario=job["scenario"],
        trace=job["trace"],
        tracks_by_snap=tracks,
        radio=cfg["comm_radio"],
        bandwidth_hz=job["bandwidth_hz"],
        wavelength_m=job["wavelength_m"],
        dt_comm_s=float(cfg["dt_comm_s"]),
        dt_sense_s=float(cfg["dt_sense_s"]),
        uses_sensing=scheme == "xapp",
    )


def _tune(jobs, cfg) -> dict[str, Any]:
    base = _base_params(cfg)
    none = np.mean([_run(job, "none", base, cfg)["outage_s_per_ue_min"] for job in jobs])
    best = None
    best_gain = -1e9
    grid = []
    for budget in (2, 4):
        for horizon in cfg["xapp"]["horizon_s"]:
            for hold in cfg["xapp"]["hold_s"]:
                params = dict(base)
                params["horizon_s"] = float(horizon)
                params["hold_s"] = float(hold)
                outages = []
                false = []
                for job in jobs:
                    result = _run(job, "xapp", params, cfg, budget)
                    outages.append(result["outage_s_per_ue_min"])
                    false.append(result["false_handover_rate"])
                mean_out = float(np.mean(outages))
                gain = none - mean_out
                row = {
                    "budget": budget,
                    "horizon_s": float(horizon),
                    "hold_s": float(hold),
                    "outage": mean_out,
                    "gain": gain,
                    "false_ho": float(np.mean(false)),
                }
                grid.append(row)
                print(f"tune budget {budget} H {horizon} hold {hold} gain {gain:.3f}", flush=True)
                if gain > best_gain:
                    best_gain = gain
                    best = row
    a3_best = _tune_a3(jobs, cfg, base)
    trend_best = _tune_trend(jobs, cfg, base)
    return {"xapp": best, "a3": a3_best, "trend": trend_best, "none_outage": none, "grid": grid}


def _tune_a3(jobs, cfg, base):
    best = None
    best_out = 1e9
    for offset in cfg["a3"]["offset_db"]:
        for hys in cfg["a3"]["hysteresis_db"]:
            for ttt in cfg["a3"]["ttt_s"]:
                params = dict(base)
                params.update({"offset_db": float(offset), "hysteresis_db": float(hys), "ttt_s": float(ttt)})
                out = float(np.mean([_run(job, "a3", params, cfg)["outage_s_per_ue_min"] for job in jobs]))
                print(f"tune a3 {offset} {hys} {ttt} out {out:.3f}", flush=True)
                if out < best_out:
                    best_out = out
                    best = {"offset_db": float(offset), "hysteresis_db": float(hys), "ttt_s": float(ttt), "outage": out}
    return best


def _tune_trend(jobs, cfg, base):
    best = None
    best_out = 1e9
    for window in cfg["trend"]["window_s"]:
        for drop in cfg["trend"]["drop_db"]:
            for horizon in cfg["trend"]["horizon_s"]:
                params = dict(base)
                params.update({"window_s": float(window), "drop_db": float(drop), "horizon_s": float(horizon)})
                out = float(np.mean([_run(job, "trend", params, cfg)["outage_s_per_ue_min"] for job in jobs]))
                if out < best_out:
                    best_out = out
                    best = {"window_s": float(window), "drop_db": float(drop), "horizon_s": float(horizon), "outage": out}
    return best


def _evaluate(jobs, cfg, chosen):
    base = _base_params(cfg)
    schemes = {
        "none": (base, None, "none"),
        "oracle": (base, None, "oracle"),
        "beam": (base, None, "beam"),
        "a3": ({**base, **{k: chosen["a3"][k] for k in ("offset_db", "hysteresis_db", "ttt_s")}}, None, "a3"),
        "trend": ({**base, **{k: chosen["trend"][k] for k in ("window_s", "drop_db", "horizon_s")}}, None, "trend"),
        "xapp": (
            {**base, "horizon_s": chosen["xapp"]["horizon_s"], "hold_s": chosen["xapp"]["hold_s"]},
            int(chosen["xapp"]["budget"]),
            "xapp",
        ),
    }
    rows = []
    for name, (params, budget, scheme) in schemes.items():
        by_mount: dict[str, list] = {mount: [] for mount in MOUNTS}
        by_class: dict[str, list] = {name: [] for name in CLASSES}
        values = []
        extras = []
        for job in jobs:
            result = _run(job, scheme, params, cfg, budget)
            values.append(result)
            extras.append(job["velocity"])
            by_mount[job["mount"]].append(result)
        rows.append({"scheme": name, "all": _ci(values), "by_mount": {m: _ci(v) for m, v in by_mount.items()}, "velocity": _ci_vel(extras)})
    return rows


def _ci(results: list[dict[str, Any]]) -> dict[str, Any]:
    def pack(key):
        numbers = [float(row[key]) for row in results if row.get(key) is not None]
        if not numbers:
            return {"mean": None, "ci": None}
        array = np.asarray(numbers, dtype=np.float64)
        mean = float(np.mean(array))
        if len(array) < 2:
            return {"mean": mean, "ci": 0.0}
        return {"mean": mean, "ci": 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))}

    pred_p = [row["prediction"]["precision"] for row in results if row.get("prediction") and row["prediction"]["precision"] is not None]
    pred_r = [row["prediction"]["recall"] for row in results if row.get("prediction") and row["prediction"]["recall"] is not None]
    pred_l = [row["prediction"]["lead_s"] for row in results if row.get("prediction") and row["prediction"]["lead_s"] is not None]
    return {
        "outage": pack("outage_s_per_ue_min"),
        "mean_tp": pack("mean_throughput_bps"),
        "p5_tp": pack("p5_throughput_bps"),
        "ho": pack("handovers_per_ue_min"),
        "ping": pack("ping_pong_rate"),
        "false_ho": pack("false_handover_rate"),
        "pred_precision": None if not pred_p else float(np.mean(pred_p)),
        "pred_recall": None if not pred_r else float(np.mean(pred_r)),
        "pred_lead": None if not pred_l else float(np.mean(pred_l)),
        "n": len(results),
    }


def _ci_vel(rows: list[dict[str, float]]) -> dict[str, float]:
    med = [row["median"] for row in rows if np.isfinite(row.get("median", np.nan))]
    p90 = [row["p90"] for row in rows if np.isfinite(row.get("p90", np.nan))]
    return {
        "median": None if not med else float(np.median(med)),
        "p90": None if not p90 else float(np.median(p90)),
    }


def _delay_sweep(jobs, cfg, chosen):
    base = _base_params(cfg)
    params = {**base, "horizon_s": chosen["xapp"]["horizon_s"], "hold_s": chosen["xapp"]["hold_s"]}
    budget = int(chosen["xapp"]["budget"])
    rows = []
    for delay in cfg["e2"]["loop_delay_sweep_s"]:
        params = dict(params)
        params["tau_e2_s"] = float(delay)
        by_class = {name: [] for name in ("bus/truck", "pedestrian")}
        outages = []
        for job in jobs:
            result = _run(job, "xapp", params, cfg, budget)
            outages.append(result["outage_s_per_ue_min"])
        rows.append({"tau_e2_s": float(delay), "outage": float(np.mean(outages))})
        print(f"delay {delay} out {rows[-1]['outage']:.3f}", flush=True)
    return rows


def _ho_sweep(jobs, cfg, chosen):
    base = _base_params(cfg)
    params = {**base, "horizon_s": chosen["xapp"]["horizon_s"], "hold_s": chosen["xapp"]["hold_s"]}
    budget = int(chosen["xapp"]["budget"])
    rows = []
    for tau in cfg["e2"]["tau_ho_sweep_s"]:
        trial = dict(params)
        trial["tau_ho_s"] = float(tau)
        out = float(np.mean([_run(job, "xapp", trial, cfg, budget)["outage_s_per_ue_min"] for job in jobs]))
        rows.append({"tau_ho_s": float(tau), "outage": out})
        print(f"tau_ho {tau} out {out:.3f}", flush=True)
    return rows


def _fmt(item, scale=1.0, digits=3) -> str:
    if item is None or item.get("mean") is None:
        return "—"
    return f"{item['mean'] * scale:.{digits}f} ± {item['ci'] * scale:.{digits}f}"


def _report(raw, cfg, chosen, oracle_table, eval_rows, delay_rows, ho_sweep) -> str:
    radio = cfg["comm_radio"]
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    lines = [
        "# M3 — Blockage prediction and xApp handover",
        "",
        "M2 is DONE. This milestone is ready for review and is not marked done.",
        "",
        "Inputs: final M2 configuration (blind clutter, image-method ghosts, map tracker). "
        "Detector budgets 2 and 4 (budget 6 equals 4). Comm carrier numerology 3, 1024 "
        f"subcarriers ({bw / 1e6:.2f} MHz). Channel caches were not re-traced for sensing. "
        "Comm model-B powers were rebuilt with `PathSolver` only (`scripts/cache_comm.py`) "
        "because the sensing cache stores comm coefficients without vertices.",
        "",
        f"Tx power {radio['tx_power_dbm']:.0f} dBm, noise figure {radio['noise_figure_db']:.0f} dB, "
        f"T={radio['temperature_k']:.0f} K. Outage: SNR < {radio['outage_snr_db']:.0f} dB. "
        "Also reported: M1 10 dB LoS-loss events.",
        "",
        "Throughput is Shannon `B log2(1+SNR)` capped at 7.4063 bit/s/Hz (TS 38.214 MCS 27). "
        "Sionna 2.2 `sionna.sys.PHYAbstraction` is present (EESM + NR BLER tables) but needs "
        "per-user MCS and transport-block size; it is not used for the reported rate.",
        "",
        "Sensing overhead is 1/14 of symbols and is applied only to the xApp (the scheme that "
        "uses sensing).",
        "",
        "Time base: comm at 10 ms by linearly interpolating per-link blocked power (and LoS "
        "loss through linear gain) between the 0.1 s snapshots. M1 chose dt = 0.1 s after a "
        "dt = 0.02 s check; 10 dB events last 0.7–1.3 s (p50), so 0.1 s resolves the event "
        "and 10 ms is the E2/HO scale. Sensing and tracks stay at 0.1 s.",
        "",
        f"Assumption: the network knows each UE position with horizontal Gaussian error "
        f"sigma = {cfg['ue_position']['sigma_m']:.1f} m (height is exact).",
        "",
        f"E2 report period {cfg['e2']['report_period_s']:.2f} s. Default loop delay "
        f"{cfg['e2']['loop_delay_s']*1e3:.0f} ms. Default handover interruption "
        f"tau_HO = {cfg['e2']['tau_ho_s']*1e3:.0f} ms (NR interruption; also swept).",
        "",
        "xApp policy: hand over if serving LoS blockage (model B ≥ 10 dB on the predicted "
        "geometry, constant-velocity along the lane/sidewalk) starts in "
        "[tau_E2 + tau_HO, H] and the other cell is predicted clear (loss < 3 dB) for that "
        "duration. Hold timer blocks a reverse HO. Return: after the hold, if the previous "
        "cell is predicted clear and the current cell is predicted blocked.",
        "",
        "## Oracle consistency (tuning seeds, vs M1.5)",
        "",
        "| Mount | Density | Fixed 10 dB | Oracle 10 dB | M1.5 fixed | M1.5 oracle |",
        "|---|---|---|---|---|---|",
    ]
    for row in oracle_table:
        lines.append(
            f"| {row['mount']} | {row['density']} | {row['fixed']:.2f} | {row['oracle']:.2f} | "
            f"{row['m15_fixed']:.2f} | {row['m15_oracle']:.2f} |"
        )
    x = chosen["xapp"]
    lines.extend(
        [
            "",
            "## Operating point (tuning seeds)",
            "",
            f"Chosen on net outage reduction vs no-action, counting false handovers in the table. "
            f"Budget {x['budget']}, H = {x['horizon_s']:.1f} s, hold = {x['hold_s']:.1f} s. "
            f"Tuning outage {x['outage']:.3f} s/UE-min, gain {x['gain']:.3f}, false-HO rate {x['false_ho']:.3f}.",
            "",
            f"A3: offset {chosen['a3']['offset_db']:.1f} dB, hysteresis {chosen['a3']['hysteresis_db']:.1f} dB, "
            f"TTT {chosen['a3']['ttt_s']*1e3:.0f} ms.",
            f"RSRP-trend: window {chosen['trend']['window_s']:.2f} s, drop {chosen['trend']['drop_db']:.0f} dB, "
            f"H {chosen['trend']['horizon_s']:.1f} s.",
            "",
            "## Evaluation seeds (mean ± 95% CI over jobs)",
            "",
            "| Scheme | Outage [s / UE-min] | Mean TP [Mbit/s] | 5th TP [Mbit/s] | HO / UE-min | Ping-pong | False HO |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in eval_rows:
        stats = row["all"]
        lines.append(
            f"| {row['scheme']} | {_fmt(stats['outage'])} | {_fmt(stats['mean_tp'], 1e-6, 2)} | "
            f"{_fmt(stats['p5_tp'], 1e-6, 2)} | {_fmt(stats['ho'], 1, 2)} | {_fmt(stats['ping'], 1, 2)} | "
            f"{_fmt(stats['false_ho'], 1, 2)} |"
        )
    xapp_row = next(row for row in eval_rows if row["scheme"] == "xapp")
    lines.extend(
        [
            "",
            f"xApp prediction precision {xapp_row['all']['pred_precision']}, "
            f"recall {xapp_row['all']['pred_recall']}, median lead {xapp_row['all']['pred_lead']} s.",
            "",
            f"Along-lane velocity error of the map tracker (evaluation jobs): median "
            f"{xapp_row['velocity']['median']} m/s, p90 {xapp_row['velocity']['p90']} m/s. "
            "A larger along-lane error moves the predicted blockage time by "
            "roughly error / (relative closer speed). That is the main source of late or early HOs.",
            "",
            "## H3: outage vs E2 loop delay (xApp, evaluation)",
            "",
            "| tau_E2 [s] | Outage [s / UE-min] |",
            "|---|---|",
        ]
    )
    for row in delay_rows:
        lines.append(f"| {row['tau_e2_s']:.2f} | {row['outage']:.3f} |")
    lines.extend(["", "## Handover interruption sweep", "", "| tau_HO [s] | Outage [s / UE-min] |", "|---|---|"])
    for row in ho_sweep:
        lines.append(f"| {row['tau_ho_s']:.2f} | {row['outage']:.3f} |")
    lines.extend(
        [
            "",
            "## H2 / H3",
            "",
            "H2: compare xApp outage and throughput to A3, RSRP-trend, same-cell beam switch, "
            "no-action, and oracle in the evaluation table. H3: the delay sweep is the "
            "break-even curve; read it per the table (not split by class in this first cut "
            "because the predictor is not class-conditional at decision time; serving-event "
            "blocker class is in the M2 lead-time tables).",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
