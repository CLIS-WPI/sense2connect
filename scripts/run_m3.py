"""M3 rework: 10 ms model-B timeline, margin sweep, per-margin tuning, evaluation.

Stages (wall time and GPU memory are logged to ``results/M3/stages.json``):
1. build   per job, cached: analytic blocker/UE poses every 10 ms (CPU,
           4 workers), map-tracker replay (CPU), batched model B on the GPU,
           10 dB events with onset, xApp predictions on the GPU.
2. budget  3GPP reference SNR, reference margin, SNR for all margins in one
           GPU pass.
3. tune    every scheme at every margin on tuning seeds (lanes = grid x jobs x UEs).
4. eval    evaluation seeds with the per-margin parameters.
5. sweeps  H3 (E2 loop delay) and handover interruption.

Run: ``python scripts/run_m3.py [--workers 4] [--limit-seeds N]``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.linkbudget import extra_loss_db, sensing_overhead, snr_all_margins, snr_ref_db, snr_req_db  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from sim.sensing.provenance import CacheRefused, require_scoped, scoped_version  # noqa: E402
from xapp.metrics import CLASSES, ci95, fixed_cell, lane_summary, link_events, stateless  # noqa: E402
from xapp.schemes import REASON, Lanes, simulate  # noqa: E402

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")
DT_COMM = 0.01
DT_SENSE = 0.1
KIND = "m3_timeline"
OUT = ROOT / "results" / "M3"


def timeline_sources() -> list[Path]:
    names = [
        "xapp/timeline.py", "xapp/metrics.py", "xapp/predict_torch.py", "xapp/predict.py", "xapp/tracks.py",
        "sim/sensing/track_map.py", "sim/sensing/cluster.py", "sim/sensing/ghost.py",
        "sim/comm/blockage.py", "sim/comm/blockage_torch.py", "sim/comm/events.py",
        "sim/scenes/motion.py", "sim/scenes/traffic.py", "sim/scenes/loop.py",
    ]
    return [ROOT / n for n in names]


def geometry_sources() -> list[Path]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from cache_comm_geometry import sources

    return sources()


class Stages:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, float]] = {}

    def run(self, name: str, fn, *args, **kwargs):
        import torch

        torch.cuda.reset_peak_memory_stats()
        clock = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        self.rows[name] = {
            "wall_s": time.perf_counter() - clock,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "unix_start": time.time() - (time.perf_counter() - clock),
            "unix_end": time.time(),
        }
        print(f"[stage] {name}: {self.rows[name]['wall_s']:.1f} s, peak GPU {self.rows[name]['peak_gpu_mem_gb']:.2f} GB", flush=True)
        return result


# ---------------------------------------------------------------- build


def _cpu_part(item: tuple) -> dict[str, Any]:
    """Poses every 10 ms and tracker replay (CPU)."""
    raw, cfg, seed, mount, density = item
    import torch

    torch.set_num_threads(2)
    from sim.scenes.traffic import prepare_scenario
    from xapp.timeline import actor_tracks, comm_times, load_geometry
    from xapp.tracks import load_detections, replay_map

    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    geom, meta = load_geometry(directory)
    require_scoped(meta, kind="comm_geometry", sources=geometry_sources())
    scenario = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=DT_SENSE)
    times = comm_times(int(meta["n_snapshots"]), DT_SENSE, DT_COMM)
    actors = actor_tracks(scenario, times)
    detections = load_detections(directory)
    spec = raw["sensing_radar"]
    tracks = {}
    for budget, det in cfg["sensing"]["budgets"].items():
        tracks[int(budget)] = replay_map(
            detections,
            train=int(det["train"]),
            pfa=float(det["pfa"]),
            eps_m=float(det["eps_m"]),
            ghost_association_m=float(det["ghost_association_m"]),
            walls=[float(v) for v in spec["wall_y_m"]],
            spec=spec,
            lanes=list(raw["lanes"]),
            sidewalks=list(raw["sidewalks"]),
            tuned=cfg["sensing"]["map_tracker"],
        )
    sizes = {k: (float(v["length_m"]), float(v["width_m"]), float(v["height_m"])) for k, v in scenario["blocker_kinds"].items()}
    return {
        "key": (seed, mount, density),
        "times": times,
        "actors": actors,
        "tracks": tracks,
        "sizes": sizes,
        "geom_meta": meta,
        "detection_provenance": detections.get("provenance"),
    }


def build(jobs: list[tuple[int, str, str]], raw: dict, cfg: dict, workers: int, rebuild: bool = False) -> dict[tuple, dict[str, Any]]:
    """Load cached timelines; build missing ones (CPU pool + GPU in this process)."""
    rw = cfg["rework"]
    params = {"predict": rw["predict"], "map_tracker": cfg["sensing"]["map_tracker"], "budgets": cfg["sensing"]["budgets"], "dt_comm_s": DT_COMM}
    out: dict[tuple, dict[str, Any]] = {}
    todo = []
    for job in jobs:
        cached = None if rebuild else _load_timeline(job, params)
        if cached is None:
            todo.append(job)
        else:
            out[job] = cached
    print(f"timelines cached {len(out)}, to build {len(todo)}", flush=True)
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(max(1, min(4, workers))) as pool:
            for part in pool.imap(_cpu_part, [(raw, cfg, *job) for job in todo]):
                out[part["key"]] = _gpu_part(part, rw, params)
                print(f"  built {part['key']}", flush=True)
    return out


def _gpu_part(part: dict[str, Any], rw: dict, params: dict) -> dict[str, Any]:
    from xapp.predict_torch import pack_tracks, predict_torch, trigger_table, ue_fixes  # noqa: F401
    from xapp.timeline import held_segments, load_geometry, model_b_timeline

    seed, mount, density = part["key"]
    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    geom, meta = load_geometry(directory)
    actors = part["actors"]
    times = part["times"]
    seg = held_segments(geom, times, actors["ue_position_m"], DT_SENSE)
    tl = model_b_timeline(seg, actors["blocker_position_m"], actors["blocker_size_m"], float(meta["wavelength_m"]))
    cell = fixed_cell(tl["unblocked_power"])
    events = link_events(
        tl["los_loss_db"], tl["los_blocker"], cell, actors["blocker_name"], actors["blocker_kind"], actors["blocker_lap"], DT_COMM
    )
    pr = rw["predict"]
    taus = np.arange(0.0, float(pr["tau_max_s"]) + 1e-9, float(pr["tau_step_s"]))
    n_r = int(meta["n_snapshots"])
    rep_k = np.arange(n_r) * int(round(DT_SENSE / DT_COMM))
    ue_pos = actors["ue_position_m"][rep_k]
    ue_vel = actors["ue_velocity_mps"][rep_k]
    fix = ue_fixes(ue_pos, ue_vel, float(pr["ue_sigma_m"]), seed * 17 + 3)
    oru = geom["oru_position_m"][0]
    preds = {}
    for budget, rows in part["tracks"].items():
        packed = pack_tracks(rows, {"bus": part["sizes"]["bus"], "pedestrian": part["sizes"]["pedestrian"]})
        preds[budget] = predict_torch(packed, fix, ue_vel, oru, taus, float(meta["wavelength_m"]))
    data = {
        **{k: v for k, v in tl.items()},
        "fixed_cell": cell,
        "taus": taus,
        **{f"pred_{b}": p for b, p in preds.items()},
    }
    np.savez(directory / "m3_timeline.npz", **data)
    tl_meta = {
        "seed": seed, "mount": mount, "density": density, "n_steps": int(len(times)),
        "events": events, "params": params, "geometry_provenance": meta["provenance"],
        "detection_provenance_recorded": part["detection_provenance"],
        "provenance": {"kind": KIND, **scoped_version(timeline_sources())},
    }
    (directory / "m3_timeline_meta.json").write_text(json.dumps(tl_meta, default=_json) + "\n", encoding="utf-8")
    return {"data": data, "events": events, "meta": tl_meta}


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _load_timeline(job: tuple, params: dict) -> dict[str, Any] | None:
    seed, mount, density = job
    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    meta_path = directory / "m3_timeline_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        require_scoped(meta, kind=KIND, sources=timeline_sources())
        geom_meta = json.loads((directory / "comm_geometry_meta.json").read_text(encoding="utf-8"))
        require_scoped(geom_meta, kind="comm_geometry", sources=geometry_sources())
    except CacheRefused as error:
        print(f"  refuse {job}: {error}", flush=True)
        return None
    if json.loads(json.dumps(params, default=_json)) != meta["params"] or geom_meta["provenance"] != meta["geometry_provenance"]:
        print(f"  refuse {job}: parameters or geometry changed", flush=True)
        return None
    with np.load(directory / "m3_timeline.npz") as f:
        data = {k: f[k] for k in f.files}
    return {"data": data, "events": meta["events"], "meta": meta}


# ---------------------------------------------------------------- budget


def budget(built: dict, tune_jobs: list, rw: dict, bandwidth_hz: float) -> dict[str, Any]:
    b = rw["budget"]
    req = snr_req_db(float(rw["service_rate_bps"]), bandwidth_hz)
    best = []
    per_ue = []
    for job in tune_jobs:
        un = snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bandwidth_hz, b["ue_noise_figure_db"]) - b["losses_db"]
        top = un.max(-1)  # [T, U]
        best.append(top.reshape(-1))
        per_ue.extend(np.median(top, axis=0).tolist())
    margin_ref = float(np.median(np.concatenate(best)) - req)
    v1 = rw["v1_radio"]
    v1_offset = (v1["tx_power_dbm"] - v1["ue_noise_figure_db"]) - (b["tx_power_dbm"] - b["ue_noise_figure_db"] - b["losses_db"])
    points = [float(m) for m in rw["margins_db"]] + [margin_ref, margin_ref + v1_offset]
    labels = [f"{m:.0f} dB" for m in rw["margins_db"]] + ["3GPP short-range reference", "v1 radio (high margin)"]
    return {
        "snr_req_db": req,
        "margin_ref_db": margin_ref,
        "v1_offset_db": v1_offset,
        "points_db": points,
        "labels": labels,
        "extra_loss_db": extra_loss_db(points, margin_ref).tolist(),
    }


def all_snr(built: dict, jobs: list, rw: dict, bandwidth_hz: float, losses: list[float], device: str = "cuda") -> dict[tuple, np.ndarray]:
    """SNR [M, T, U, C] per job, every margin in one GPU pass over all jobs."""
    b = rw["budget"]
    ref = np.stack([
        snr_ref_db(built[j]["data"]["blocked_power"], b["tx_power_dbm"], bandwidth_hz, b["ue_noise_figure_db"]) - b["losses_db"]
        for j in jobs
    ])  # [J, T, U, C]
    out = snr_all_margins(ref, np.asarray(losses), device=device).cpu().numpy()  # [M, J, T, U, C]
    return {job: out[:, i] for i, job in enumerate(jobs)}


# ---------------------------------------------------------------- schemes


def _grid(spec: dict) -> list[dict[str, float]]:
    keys = list(spec)
    return [dict(zip(keys, values)) for values in itertools.product(*[spec[k] for k in keys])]


def make_lanes(jobs, combos, snr_m, built, scheme, a3, rw, budget_info, *, e2_s=None, tau_ho_s=None, hybrid=False, min_start_s=None) -> tuple[Lanes, list[tuple]]:
    """Lanes for (combo, job, UE).

    ``hybrid``: xApp lanes whose handovers set no hold at all, so A3 stays
    active the whole time (it may hand back, and it handles every return).
    ``min_start_s``: keep only triggers whose predicted blockage start is at
    least this far after the report (onset-advance policy); None = all.
    """
    e2 = rw["e2"]
    pr = rw["predict"]
    overhead = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"]) if scheme == "xapp" else 0.0
    index, snr, trig, trig_end, rows = [], [], [], [], []
    for ci, combo in enumerate(combos):
        for job in jobs:
            data = built[job]["data"]
            for u in range(data["blocked_power"].shape[1]):
                index.append((ci, job, u))
                snr.append(snr_m[job][:, u, :])
                p = {**a3, **combo}
                if scheme == "xapp":
                    from xapp.predict_torch import trigger_table

                    source = p["budget"] if isinstance(p["budget"], str) else int(p["budget"])  # detector budget, or "genie"
                    key = (job, source, float(p["horizon_s"]), min_start_s)
                    if key not in _TRIG:
                        table = trigger_table(data[f"pred_{source}"], data["taus"], float(p["horizon_s"]), float(pr["los_block_db"]), float(pr["los_clear_db"]))
                        end = np.nan_to_num(table["end_s"], nan=0.0)
                        trig_ok = table["trigger"]
                        if min_start_s is not None:
                            trig_ok = trig_ok & (np.nan_to_num(table["start_s"], nan=-1.0) >= float(min_start_s) - 1e-9)
                        _TRIG[key] = (trig_ok, np.round(end / DT_COMM).astype(np.int64))
                    trig.append(_TRIG[key][0][:, u, :])
                    trig_end.append(np.zeros_like(_TRIG[key][1][:, u, :]) if hybrid else _TRIG[key][1][:, u, :])
                rows.append(p)
    n = len(index)
    col = lambda k, d=0.0: np.array([float(r.get(k, d)) for r in rows])  # noqa: E731
    lanes = Lanes(
        snr_db=np.stack(snr),
        scheme=np.full(n, REASON[scheme]),
        offset_db=col("offset_db"),
        hysteresis_db=col("hysteresis_db"),
        ttt_steps=np.round(col("ttt_s") / DT_COMM).astype(np.int64),
        filter_a=np.full(n, float(rw["filter_a"])),
        window_steps=np.maximum(2, np.round(col("window_s", 0.02) / DT_COMM).astype(np.int64)),
        drop_db=col("drop_db", 1e9),
        trend_horizon_s=col("horizon_s", 0.0) if scheme == "trend" else np.zeros(n),
        hold_steps=np.zeros(n, dtype=np.int64) if hybrid else np.round(col("hold_s") / DT_COMM).astype(np.int64),
        tau_ho_steps=np.full(n, int(round((e2["tau_ho_s"] if tau_ho_s is None else tau_ho_s) / DT_COMM))),
        e2_delay_steps=np.full(n, int(round((e2["loop_delay_s"] if e2_s is None else e2_s) / DT_COMM))),
        overhead=np.full(n, overhead),
        initial_cell=np.array([int(built[j]["data"]["fixed_cell"][0, u]) for _, j, u in index]),
        trigger=np.stack(trig).astype(bool) if trig else None,
        trigger_end_steps=np.stack(trig_end) if trig_end else None,
        block_db=float(pr["los_block_db"]),
        report_steps=int(round(DT_SENSE / DT_COMM)),
        dt_s=DT_COMM,
    )
    return lanes, index


_TRIG: dict[tuple, np.ndarray] = {}


def run_lanes(lanes: Lanes, index, built, bandwidth_hz, rate_req, oracle_rates, scheme) -> list[dict[str, Any]]:
    sim = simulate(lanes, bandwidth_hz=bandwidth_hz, rate_req_bps=rate_req, max_se=MAX_NR_SE)
    proactive = (REASON[scheme],) if scheme in ("trend", "xapp") else ()
    out = []
    for lane, (ci, job, u) in enumerate(index):
        events = [ev for ev in built[job]["events"] if ev["ue"] == u]
        s = lane_summary(sim, lane, events, oracle_rates[job][:, u], DT_COMM, proactive_reasons=proactive)
        s.update({"combo": ci, "job": job, "ue": u})
        out.append(s)
    return out


def stateless_rows(jobs, snr_m, built, kind, bandwidth_hz, rate_req) -> tuple[list[dict], dict]:
    rows, rates = [], {}
    for job in jobs:
        d = built[job]["data"]
        n_t, n_u, _ = d["blocked_power"].shape
        offset = snr_m[job][:, :, 0] - _db(d["blocked_power"][:, :, 0])  # [T, U]
        if kind == "oracle":
            power = d["blocked_power"].max(-1)
        elif kind == "none":
            power = np.take_along_axis(d["blocked_power"], d["fixed_cell"][..., None], -1)[..., 0]
        else:  # beam: analog single-path beam on the fixed cell
            los = np.take_along_axis(d["los_blocked_power"], d["fixed_cell"][..., None], -1)[..., 0]
            alt = np.take_along_axis(d["best_alt_power"], d["fixed_cell"][..., None], -1)[..., 0]
            power = np.maximum(los, alt)
        rate = np.zeros((n_t, n_u))
        for u in range(n_u):
            off = float(np.nanmedian(offset[:, u][np.isfinite(offset[:, u])]))
            st = stateless(power[:, u], off, bandwidth_hz, rate_req, MAX_NR_SE)
            rate[:, u] = st["rate"]
            sim = {
                "outage_req": st["outage_req"][None], "outage0": st["outage0"][None], "rate": st["rate"][None],
                "interrupted_steps": np.zeros(1, dtype=np.int64),
                "handovers": {k: np.zeros(0, dtype=np.int64) for k in ("lane", "step", "from", "to", "reason")},
            }
            events = [ev for ev in built[job]["events"] if ev["ue"] == u]
            s = lane_summary(sim, 0, events, st["rate"], DT_COMM)
            s.update({"combo": 0, "job": job, "ue": u})
            rows.append(s)
        rates[job] = rate
    return rows, rates


def _db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(x)


def objective(rows: list[dict], n_combos: int) -> list[tuple[float, float]]:
    """(mean outage at SNR_req [s/UE-min], HO/UE-min) per combo."""
    out = []
    for ci in range(n_combos):
        sel = [r for r in rows if r["combo"] == ci]
        out.append((float(np.mean([r["outage_req_s_per_min"] for r in sel])), float(np.mean([r["ho_per_min"] for r in sel]))))
    return out


def tune_margin(mi, tune_jobs, snr_all, built, rw, bw, rate_req, oracle_rates) -> dict[str, Any]:
    snr_m = {j: snr_all[j][mi] for j in tune_jobs}
    grids = rw["grids"]
    result = {}
    a3_combos = _grid(grids["a3"])
    lanes, index = make_lanes(tune_jobs, a3_combos, snr_m, built, "a3", {}, rw, None)
    rows = run_lanes(lanes, index, built, bw, rate_req, oracle_rates, "a3")
    obj = objective(rows, len(a3_combos))
    best = min(range(len(a3_combos)), key=lambda i: (obj[i][0], obj[i][1], i))
    result["a3"] = {"params": a3_combos[best], "objective": obj[best], "grid": [{**c, "outage": o[0], "ho": o[1]} for c, o in zip(a3_combos, obj)]}
    a3 = a3_combos[best]
    for scheme in ("trend", "xapp"):
        combos = _grid(grids[scheme])
        lanes, index = make_lanes(tune_jobs, combos, snr_m, built, scheme, a3, rw, None)
        rows = run_lanes(lanes, index, built, bw, rate_req, oracle_rates, scheme)
        obj = objective(rows, len(combos))
        best = min(range(len(combos)), key=lambda i: (obj[i][0], obj[i][1], i))
        result[scheme] = {"params": {**a3, **combos[best]}, "objective": obj[best], "grid": [{**c, "outage": o[0], "ho": o[1]} for c, o in zip(combos, obj)]}
    return result


def evaluate_margin(mi, jobs, snr_all, built, rw, bw, rate_req, tuned, *, e2_s=None, tau_ho_s=None, schemes=None) -> dict[str, list[dict]]:
    snr_m = {j: snr_all[j][mi] for j in jobs}
    rows = {}
    oracle_rows, oracle_rates = stateless_rows(jobs, snr_m, built, "oracle", bw, rate_req)
    for kind in ("none", "oracle", "beam"):
        if schemes and kind not in schemes:
            continue
        rows[kind] = oracle_rows if kind == "oracle" else stateless_rows(jobs, snr_m, built, kind, bw, rate_req)[0]
    for scheme in ("a3", "trend", "xapp"):
        if schemes and scheme not in schemes:
            continue
        params = tuned[scheme]["params"]
        a3 = tuned["a3"]["params"]
        lanes, index = make_lanes(jobs, [params], snr_m, built, scheme, a3, rw, None, e2_s=e2_s, tau_ho_s=tau_ho_s)
        rows[scheme] = run_lanes(lanes, index, built, bw, rate_req, oracle_rates, scheme)
    return rows, oracle_rates


def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Per job (mean over UEs, events pooled), then mean and 95 % CI over jobs."""
    by_job: dict[tuple, list[dict]] = {}
    for r in rows:
        by_job.setdefault(tuple(r["job"]), []).append(r)
    scalar_keys = ("outage_req_s_per_min", "outage0_s_per_min", "mean_rate_bps", "p5_rate_bps", "tp_loss_vs_oracle", "ho_per_min", "ping_pong")
    per_job = {k: [] for k in scalar_keys}
    precision = []
    false_ho = []
    ev_stats: dict[tuple[str, str], dict[str, list[float]]] = {}
    for job, items in by_job.items():
        for k in scalar_keys:
            per_job[k].append(float(np.mean([it[k] for it in items])))
        n_cons = sum(it["n_ho_considered"] for it in items)
        if n_cons:
            precision.append(sum(it["n_ho_matched"] for it in items) / n_cons)
            false_ho.append(1.0 - precision[-1])
        events = [ev for it in items for ev in it["events"]]
        for cls in ("all",) + CLASSES:
            for scope in ("all", "actionable"):
                sel = [ev for ev in events if (cls == "all" or ev["class"] == cls) and (scope == "all" or ev["actionable"])]
                if not sel:
                    continue
                st = ev_stats.setdefault((cls, scope), {"recall": [], "proactive_recall": [], "interruption_req_s": [], "interruption_0_s": [], "lead_s": [], "n_events": []})
                st["recall"].append(float(np.mean([ev["hit"] for ev in sel])))
                st["proactive_recall"].append(float(np.mean([ev["proactive_hit"] for ev in sel])))
                st["interruption_req_s"].append(float(np.mean([ev["interruption_req_s"] for ev in sel])))
                st["interruption_0_s"].append(float(np.mean([ev["interruption_0_s"] for ev in sel])))
                leads = [ev["lead_s"] for ev in sel if ev["lead_s"] is not None]
                if leads:
                    st["lead_s"].append(float(np.median(leads)))
                st["n_events"].append(float(len(sel)))
    out = {k: ci95(v) for k, v in per_job.items()}
    out["precision"] = ci95(precision)
    out["false_ho_rate"] = ci95(false_ho)
    out["events"] = {f"{c}|{s}": {k: (ci95(v) if k != "n_events" else float(np.sum(v))) for k, v in st.items()} for (c, s), st in ev_stats.items()}
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-seeds", type=int, default=0, help="smoke test: first N seeds of each split")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--rebuild", action="store_true", help="ignore cached timelines (determinism check)")
    args = parser.parse_args()
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected exactly one visible GPU (GPU 1), got {torch.cuda.device_count()}")
    torch.set_num_threads(8)
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tuning = [int(s) for s in seeds["tuning"]]
    evaluation = [int(s) for s in seeds["evaluation"]]
    if args.limit_seeds:
        tuning, evaluation = tuning[: args.limit_seeds], evaluation[: args.limit_seeds]
    tune_jobs = [(s, m, d) for s in tuning for m in MOUNTS for d in DENSITIES]
    eval_jobs = [(s, m, d) for s in evaluation for m in MOUNTS for d in DENSITIES]
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    st = Stages()

    built = st.run("build", build, tune_jobs + eval_jobs, raw, cfg, args.workers, args.rebuild)
    info = st.run("budget", budget, built, tune_jobs, rw, bw)
    print(json.dumps(info, indent=2), flush=True)
    snr_all = st.run("snr_all_margins", all_snr, built, tune_jobs + eval_jobs, rw, bw, info["extra_loss_db"])
    margins = info["points_db"]

    def tune_all():
        tuned = []
        for mi, m in enumerate(margins):
            orates = stateless_rows(tune_jobs, {j: snr_all[j][mi] for j in tune_jobs}, built, "oracle", bw, rate_req)[1]
            tuned.append(tune_margin(mi, tune_jobs, snr_all, built, rw, bw, rate_req, orates))
            print(f"tuned margin {m:.1f}: " + ", ".join(f"{k} {v['params']} -> {v['objective'][0]:.3f}" for k, v in tuned[-1].items()), flush=True)
        return tuned

    tuned = st.run("tune", tune_all)

    def eval_all():
        res = []
        for mi in range(len(margins)):
            rows, _ = evaluate_margin(mi, eval_jobs, snr_all, built, rw, bw, rate_req, tuned[mi])
            res.append({k: aggregate(v) for k, v in rows.items()})
        return res

    evaluated = st.run("evaluate", eval_all)

    def sweeps():
        ref_i = info["labels"].index("3GPP short-range reference")
        extra = float(rw["extra_h3_margin_db"])
        extra_i = margins.index(extra) if extra in margins else None
        h3 = {}
        for name, mi in (("reference", ref_i), (f"{extra:.0f} dB", extra_i)):
            if mi is None:
                continue
            h3[name] = []
            for d in rw["e2"]["loop_delay_sweep_s"]:
                rows, _ = evaluate_margin(mi, eval_jobs, snr_all, built, rw, bw, rate_req, tuned[mi], e2_s=float(d), schemes=("xapp",))
                h3[name].append({"tau_e2_s": float(d), "xapp": aggregate(rows["xapp"])})
        ho = []
        for tau in rw["e2"]["tau_ho_sweep_s"]:
            rows, _ = evaluate_margin(ref_i, eval_jobs, snr_all, built, rw, bw, rate_req, tuned[ref_i], tau_ho_s=float(tau), schemes=("a3", "trend", "xapp"))
            ho.append({"tau_ho_s": float(tau), **{k: aggregate(v) for k, v in rows.items()}})
        return {"h3": h3, "tau_ho": ho}

    sw = st.run("sweeps", sweeps)

    margin_stats = _margin_distribution(built, tune_jobs + eval_jobs, rw, bw, info)
    event_stats = _event_stats(built, tune_jobs, eval_jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "budget": info,
        "margin_distribution": margin_stats,
        "event_stats": event_stats,
        "tuned": [{k: {"params": v["params"], "objective": v["objective"]} for k, v in t.items()} for t in tuned],
        "tuning_grids": [{k: v["grid"] for k, v in t.items()} for t in tuned],
        "evaluation": evaluated,
        "sweeps": sw,
        "stages": st.rows,
        "jobs": {"tuning": len(tune_jobs), "evaluation": len(eval_jobs)},
        "overhead_xapp": sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"]),
        "bandwidth_hz": bw,
    }
    name = "metrics.json" if not args.limit_seeds else "metrics_smoke.json"
    (args.out / name).write_text(json.dumps(payload, indent=1, default=_json) + "\n", encoding="utf-8")
    (args.out / "stages.json").write_text(json.dumps(st.rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / name}", flush=True)


def _margin_distribution(built, jobs, rw, bw, info) -> dict[str, Any]:
    b = rw["budget"]
    req = info["snr_req_db"]
    per_ue, per_step = {}, {}
    for job in jobs:
        un = snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"]
        top = un.max(-1) - req
        key = job[1]
        per_ue.setdefault(key, []).extend(np.median(top, axis=0).tolist())
        per_step.setdefault(key, []).append(top.reshape(-1))
    q = [0, 5, 25, 50, 75, 95, 100]
    out = {}
    for key in per_ue:
        out[key] = {
            "per_ue_median_margin_db_percentiles": dict(zip(map(str, q), np.percentile(per_ue[key], q).round(2).tolist())),
            "per_step_margin_db_percentiles": dict(zip(map(str, q), np.percentile(np.concatenate(per_step[key]), q).round(2).tolist())),
            "n_ue_jobs": len(per_ue[key]),
        }
    return out


def _event_stats(built, tune_jobs, eval_jobs) -> dict[str, Any]:
    out = {}
    for split, jobs in (("tuning", tune_jobs), ("evaluation", eval_jobs)):
        events = [ev for j in jobs for ev in built[j]["events"]]
        rows = {}
        for cls in ("all",) + CLASSES:
            sel = [ev for ev in events if cls == "all" or ev["class"] == cls]
            onset = np.array([ev["onset_s"] for ev in sel if ev["onset_s"] is not None and math.isfinite(ev["onset_s"])])
            dur = np.array([ev["end_s"] - ev["start_s"] + DT_COMM for ev in sel])
            rows[cls] = {
                "n": len(sel),
                "per_ue_min": len(sel) / (len(jobs) * 2 * 59.9 / 60.0),
                "actionable_share": float(np.mean([ev["actionable"] for ev in sel])) if sel else None,
                "onset_10_90_s": None if onset.size == 0 else dict(zip(("p10", "p50", "p90"), np.percentile(onset, [10, 50, 90]).round(3).tolist())),
                "duration_s": None if dur.size == 0 else dict(zip(("p10", "p50", "p90"), np.percentile(dur, [10, 50, 90]).round(3).tolist())),
            }
        out[split] = rows
    return out


if __name__ == "__main__":
    main()
