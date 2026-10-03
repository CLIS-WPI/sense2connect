"""60 s M2 study from the channel cache.

Tuning seeds 101–105 choose two parameter sets (ghost off, ghost on).
Evaluation seeds 1001–1010 run both ghost methods at both sets, so the
method and the parameters are not confounded. Blind and twin clutter are
both reported.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.events import apply_hysteresis
from sim.scenes.config import load_yaml
from sim.scenes.loop import _ue_events
from sim.scenes.traffic import prepare_scenario
from sim.sensing.cache import cache_dir, load_frame, load_meta, load_static, split_frame
from sim.sensing.cfar import ca_cfar, local_maxima
from sim.sensing.channel import localize_cells, n_lags, power_map
from sim.sensing.cluster import cluster_detections
from sim.sensing.ghost import reject_ghosts
from sim.sensing.metrics import CLASSES, empty_score, scaled_gates, summarize, update_score
from sim.sensing.provenance import CacheRefused, attach, require
from sim.sensing.process_torch import add_noise_batch, cpi_from_paths, delay_doppler_batch, frequency_responses
from sim.sensing.radar import build_radar, ground_truth
from sim.sensing.trace import trace_job
from sim.sensing.track import Tracker
from sim.sensing.waveform import Waveform, waveform_from_config

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")
LEADS_S = (0.1, 0.3, 0.5, 1.0)
CFAR_GRID = [{"guard": 2, "train": train, "pfa": pfa} for train in (4, 8) for pfa in (1e-3, 1e-4, 1e-5)]
EPS_M = (3.0, 5.0)
COAST = (2, 4)
ASSOCIATION_M = (6.0, 10.0)
FA_BUDGETS = (2.0, 4.0, 6.0)
GATE_SCALES = (0.5, 1.0, 2.0)
_EVENT_EXTRA = {"min_gap_s": 0.5, "los_threshold_db": 10.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tuning = [int(seed) for seed in seeds["tuning"]]
    evaluation = [int(seed) for seed in seeds["evaluation"]]
    if args.seeds:
        tuning = [seed for seed in tuning if seed in args.seeds]
        evaluation = [seed for seed in evaluation if seed in args.seeds]
    jobs = [(mount, density, seed) for seed in tuning + evaluation for mount in MOUNTS for density in DENSITIES]
    frames = None if args.max_frames <= 0 else int(args.max_frames)
    if args.workers <= 1:
        for mount, density, seed in jobs:
            _ensure(raw, mount, density, seed, frames)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            pool.map(_ensure_worker, [(mount, density, seed, frames) for mount, density, seed in jobs])
    print("caches ready", len(jobs), flush=True)
    if frames is not None:
        return
    spec = raw["sensing_radar"]
    wave = waveform_from_config(raw)
    wide = waveform_from_config(raw, n_subcarriers=int(spec["waveform"]["n_subcarriers_alt"]))
    geometries = {mount: _geometry(raw, mount) for mount in MOUNTS}
    centroid_note = _centroid_note(geometries["lamppost"])
    sizes = raw["blocker_kinds"]
    tune_cases = [
        _detections(raw, mount, density, seed, wave, geometries[mount], CFAR_GRID)
        for mount, density, seed in jobs
        if seed in tuning
    ]
    grid = _grid(tune_cases, spec, sizes)
    picks = _budgets(grid)
    eval_cases = []
    for mount, density, seed in jobs:
        if seed not in evaluation:
            continue
        narrow = _detections(raw, mount, density, seed, wave, geometries[mount], CFAR_GRID)
        broad = _detections(raw, mount, density, seed, wide, geometries[mount], CFAR_GRID)
        eval_cases.append({"narrow": narrow, "broad": broad, "mount": mount, "density": density, "seed": seed})
    text, payload = _report(raw, wave, wide, grid, picks, eval_cases, spec, sizes, centroid_note)
    output = ROOT / "results" / "M2"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(text, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(payload, default=_json) + "\n", encoding="utf-8")
    print(text)


def _cache_fresh(directory: Path, events_path: Path, last: int) -> bool:
    if not (directory / f"frame_{last:04d}.npz").exists() or not events_path.exists():
        return False
    try:
        load_meta(directory)
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        require(payload, kind="events", extra=_EVENT_EXTRA)
    except (CacheRefused, FileNotFoundError, json.JSONDecodeError, OSError, KeyError):
        return False
    return True


def _event_list(directory: Path) -> list[dict[str, Any]]:
    payload = json.loads((directory / "events.json").read_text(encoding="utf-8"))
    require(payload, kind="events", extra=_EVENT_EXTRA)
    return list(payload["events"])


def _detection_extra(waveform: Waveform, cfar_grid: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "n_subcarriers": int(waveform.n_subcarriers),
        "cfar": [[int(item["guard"]), int(item["train"]), float(item["pfa"])] for item in cfar_grid],
    }


def _ensure_worker(job: tuple) -> None:
    mount, density, seed, frames = job
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    _ensure(raw, mount, density, seed, frames)


def _ensure(raw: dict, mount: str, density: str, seed: int, frames: int | None = None) -> None:
    directory = cache_dir(ROOT / "results" / "cache", mount, density, seed)
    last = 599 if frames is None else frames - 1
    events_path = directory / "events.json"
    if _cache_fresh(directory, events_path, last):
        return
    if events_path.exists():
        events_path.unlink()
    scenario = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    print(f"trace {mount} {density} {seed}", flush=True)
    summary = trace_job(
        scenario,
        mount=mount,
        density=density,
        cache_root=ROOT / "results" / "cache",
        process=False,
        record_blockage=True,
        max_frames=frames,
    )
    times = [index * 0.1 for index in range(int(summary["n_frames"]))]
    raw_events = _ue_events(summary["blockage_rows"], times, [3.0, 10.0, 20.0])
    events = apply_hysteresis(raw_events, 0.5, 0.1)
    los = [event for event in events if event["metric"] == "los" and float(event["threshold_db"]) == 10.0 and event.get("oru") == "oru-0"]
    record = attach(
        {
            "seed": seed,
            "mount": mount,
            "density": density,
            "dt_s": 0.1,
            "n_frames": int(summary["n_frames"]),
            "events": los,
        },
        kind="events",
        extra=_EVENT_EXTRA,
    )
    events_path.write_text(json.dumps(record, default=_json) + "\n", encoding="utf-8")


def _centroid_note(radar: dict[str, Any]) -> str:
    """Offset of the scattering points from the mesh origin, if the solver exposes them."""
    offsets: list[float] = []
    try:
        for target in radar["targets"]:
            model = getattr(target, "scattering_model", None)
            owners = [model, getattr(model, "spst", None) if model is not None else None]
            positions = None
            for owner in owners:
                if owner is None:
                    continue
                positions = getattr(owner, "lcs_positions", None)
                if positions is not None:
                    break
            if positions is None:
                return "Scattering-point positions were not exposed by the solver. Gates use the mesh origin."
            array = np.asarray(positions.numpy() if hasattr(positions, "numpy") else positions, dtype=np.float64)
            array = np.reshape(array, (-1, 3))
            offsets.append(float(np.linalg.norm(array.mean(axis=0))))
    except Exception as exc:
        return f"Scattering-point positions were not read ({type(exc).__name__}). Gates use the mesh origin."
    if not offsets:
        return "No sensing targets were present when the centroid was checked. Gates use the mesh origin."
    return (
        "Mean scattering-point offset from the mesh origin, over the lamppost targets: "
        f"max {max(offsets):.3f} m, mean {float(np.mean(offsets)):.3f} m. Gates use the mesh origin."
    )


def _geometry(raw: dict, mount: str) -> dict[str, Any]:
    scenario = prepare_scenario(raw, seed=101, mount=mount, density="low", duration_s=0.1, dt_s=0.1)
    radar = build_radar(scenario)
    return radar


def _detections(
    raw: dict,
    mount: str,
    density: str,
    seed: int,
    waveform: Waveform,
    radar: dict[str, Any],
    cfar_grid: list[dict[str, float]],
) -> dict[str, Any]:
    directory = cache_dir(ROOT / "results" / "cache", mount, density, seed)
    tag = f"{waveform.n_subcarriers}"
    cached = directory / f"detections_{tag}.json"
    wanted = {f"blind:{int(item['train'])}:{float(item['pfa'])}" for item in cfar_grid}
    wanted |= {key.replace("blind:", "twin:", 1) for key in wanted}
    extra = _detection_extra(waveform, cfar_grid)
    if cached.exists():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            require(payload, kind="detections", extra=extra)
            have = set(payload["frames"][0]["detections"])
            if wanted <= have:
                payload["events"] = _event_list(directory)
                payload["radar_position_m"] = [float(v) for v in radar["radar_position_m"]]
                return payload
        except (CacheRefused, KeyError, json.JSONDecodeError, OSError):
            pass
    spec = raw["sensing_radar"]
    noise = spec["noise"]
    lags = n_lags(waveform, float(spec["max_range_m"]))
    static = load_static(directory)
    meta = load_meta(directory)
    st_c, st_d = cpi_from_paths(static, waveform, float(noise["tx_power_dbm"]), max_paths=1024)
    static_h = frequency_responses(st_c, st_d, waveform)
    frames = []
    clutter_abs = 0.0
    batch = 4
    index = 0
    n_frames = int(meta["n_frames"])
    while index < n_frames:
        count = min(batch, n_frames - index)
        rows = [load_frame(directory, index + offset) for offset in range(count)]
        measured = []
        for frame in rows:
            rcs_c, rcs_d = cpi_from_paths(split_frame(frame, "rcs"), waveform, float(noise["tx_power_dbm"]), max_paths=1024)
            bg_c, bg_d = cpi_from_paths(split_frame(frame, "bg"), waveform, float(noise["tx_power_dbm"]), max_paths=1024)
            measured.append(frequency_responses(rcs_c, rcs_d, waveform) + frequency_responses(bg_c, bg_d, waveform))
        batch_h = torch.cat(measured, dim=0)
        seeds = [int(seed) * 100000 + index + offset for offset in range(count)]
        batch_h = add_noise_batch(
            batch_h,
            waveform,
            float(noise["noise_figure_db"]),
            float(noise["temperature_k"]),
            seeds,
        )
        blind = batch_h - batch_h.mean(dim=-2, keepdim=True)
        twin = batch_h - static_h
        for offset, frame in enumerate(rows):
            detections: dict[str, list[dict]] = {}
            for mode, response in (("blind", blind[offset]), ("twin", twin[offset])):
                cube = delay_doppler_batch(response[None], lags, waveform.window)[0]
                power = power_map(cube)
                if offset == 0 and index == 0 and mode == "twin":
                    blind_power = power_map(delay_doppler_batch(blind[offset][None], lags, waveform.window)[0])
                    clutter_abs = float(np.max(np.abs(power - blind_power)))
                for setting in cfar_grid:
                    mask, _alpha = ca_cfar(
                        power,
                        guard=int(setting["guard"]),
                        train=int(setting["train"]),
                        pfa=float(setting["pfa"]),
                        noise_applied=True,
                    )
                    hits = local_maxima(mask, power)
                    detections[f"{mode}:{int(setting['train'])}:{float(setting['pfa'])}"] = _locate(
                        cube, hits, waveform, radar
                    )
            frames.append({"t_s": (index + offset) * 0.1, "ground_truth": _truth(frame), "detections": detections})
        index += count
        print(f"detect {mount} {density} {seed} {tag} {index}/{n_frames}", flush=True)
    payload = attach(
        {
            "mount": mount,
            "density": density,
            "seed": seed,
            "dt_s": 0.1,
            "n_frames": n_frames,
            "radar_position_m": [float(v) for v in radar["radar_position_m"]],
            "clutter_power_abs": clutter_abs,
            "frames": frames,
        },
        kind="detections",
        extra=extra,
    )
    cached.write_text(json.dumps(payload, default=_json), encoding="utf-8")
    payload["events"] = _event_list(directory)
    return payload


def _truth(frame: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for index, identifier in enumerate(frame["gt_id"].tolist()):
        rows.append(
            {
                "id": str(identifier),
                "kind": str(frame["gt_kind"][index]),
                "x_m": float(frame["gt_x"][index]),
                "y_m": float(frame["gt_y"][index]),
                "z_m": float(frame["gt_z"][index]),
                "vx_mps": float(frame["gt_vx"][index]),
                "vy_mps": float(frame["gt_vy"][index]),
            }
        )
    return rows


def _locate(cube: torch.Tensor, hits: list[tuple[int, int, float]], waveform: Waveform, radar: dict[str, Any]) -> list[dict]:
    if not hits:
        return []
    doppler = torch.tensor([item[0] for item in hits], device=cube.device)
    lag = torch.tensor([item[1] for item in hits], device=cube.device)
    x_m, y_m, z_m, _theta, _phi = localize_cells(
        cube, doppler, lag, waveform, radar["positions_m"], radar["orientation_rad"], radar["radar_position_m"]
    )
    from sim.sensing.channel import doppler_hz, radial_velocity_mps, range_m

    rows = []
    for index, (doppler_i, lag_i, peak) in enumerate(hits):
        rows.append(
            {
                "x_m": float(x_m[index]),
                "y_m": float(y_m[index]),
                "z_m": float(z_m[index]),
                "range_m": range_m(int(lag_i), waveform),
                "doppler_hz": doppler_hz(int(doppler_i), waveform),
                "radial_velocity_mps": radial_velocity_mps(int(doppler_i), waveform),
                "power": float(peak),
            }
        )
    return rows


def _candidates(ghost: bool) -> list[dict[str, Any]]:
    rows = []
    for cfar in CFAR_GRID:
        for eps_m in EPS_M:
            for coast in COAST:
                for association_m in ASSOCIATION_M:
                    rows.append(
                        {
                            "ghost": ghost,
                            "cfar": cfar,
                            "eps_m": eps_m,
                            "coast": coast,
                            "association_m": association_m,
                        }
                    )
    return rows


def _replay(
    case: dict[str, Any],
    params: dict[str, Any],
    gates: dict[str, float],
    spec: dict[str, Any],
    sizes: dict[str, Any],
) -> tuple[dict, list[dict], list[list[dict]]]:
    score = empty_score()
    tracker = Tracker(
        dt_s=float(case["dt_s"]),
        association_gate_m=float(params["association_m"]),
        coast_frames=int(params["coast"]),
        confirm_hits=int(spec["tracker"]["confirm_hits"]),
        radar_position_m=np.asarray(case["radar_position_m"], dtype=np.float64),
    )
    clutter = str(params.get("clutter", "blind"))
    key = f"{clutter}:{int(params['cfar']['train'])}:{float(params['cfar']['pfa'])}"
    walls = [float(value) for value in spec["wall_y_m"]]
    tracks_by_frame: list[list[dict]] = []
    for frame in case["frames"]:
        detections = list(frame["detections"][key])
        if params["ghost"]:
            detections = reject_ghosts(detections, walls, float(params["association_m"]))
        clusters = cluster_detections(detections, float(params["eps_m"]), int(spec["cluster"]["min_samples"]))
        alive = tracker.step(clusters)
        rows = [
            {
                "x_m": float(track.state[0]),
                "y_m": float(track.state[1]),
                "z_m": float(track.state[2]),
                "vx_mps": float(track.state[3]),
                "vy_mps": float(track.state[4]),
                "confirmed": bool(track.confirmed),
                "age_s": float(track.age_s),
                "id": int(track.identifier),
            }
            for track in alive
        ]
        tracks_by_frame.append(rows)
        update_score(
            score,
            frame["ground_truth"],
            clusters,
            rows,
            gates,
            t_s=float(frame["t_s"]),
            sizes_m=sizes,
            wall_y_m=[float(value) for value in spec["wall_y_m"]],
        )
    return score, case.get("events", []), tracks_by_frame


def _grid(cases: list[dict[str, Any]], spec: dict[str, Any], sizes: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tuning grid point. Nothing is selected here."""
    gates = scaled_gates(1.0)
    rows = []
    for clutter in ("blind", "twin"):
        for ghost in (False, True):
            for params in _candidates(ghost):
                params = dict(params)
                params["clutter"] = clutter
                total = empty_score()
                for case in cases:
                    part, _events, _tracks = _replay(case, params, gates, spec, sizes)
                    _add(total, part)
                summary = summarize(total)
                pd = summary["classes"]["bus/truck"]["track_pd"]
                rows.append(
                    {
                        "params": params,
                        "track_pd": None if pd is None else float(pd),
                        "false_clusters_per_cpi": float(summary["false_clusters_per_cpi"]),
                        "false_alarms_per_cpi": float(summary["false_alarms_per_cpi"]),
                        "summary": summary,
                    }
                )
            print(f"grid {clutter} {'image' if ghost else 'none'} {len(rows)}", flush=True)
    return rows


def _budgets(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best tuning point under each false-alarm budget. Infeasible budgets stay empty."""
    picks = []
    for budget in FA_BUDGETS:
        for clutter in ("blind", "twin"):
            for ghost in (False, True):
                pool = [
                    row
                    for row in grid
                    if row["params"]["clutter"] == clutter and bool(row["params"]["ghost"]) is ghost
                ]
                feasible = [row for row in pool if row["false_alarms_per_cpi"] <= budget]
                best = None
                if feasible:
                    best = max(feasible, key=lambda row: (row["track_pd"] or -1.0, -row["false_alarms_per_cpi"]))
                picks.append(
                    {
                        "budget": budget,
                        "clutter": clutter,
                        "method": "image" if ghost else "none",
                        "point": best,
                    }
                )
                print(
                    f"budget {budget} {clutter} {'image' if ghost else 'none'} "
                    f"{'infeasible' if best is None else best['false_alarms_per_cpi']}",
                    flush=True,
                )
    return picks


def _add(total: dict[str, Any], part: dict[str, Any]) -> None:
    total["frames"] += part["frames"]
    for key in (
        "false_clusters",
        "cluster_fragment",
        "cluster_ghost",
        "cluster_other",
        "false_alarms",
        "confirmed_updates",
        "unmatched_confirmed",
        "track_fragment",
        "track_ghost",
        "track_other",
    ):
        total[key] += part[key]
    for name in CLASSES:
        for key, value in part["classes"][name].items():
            total["classes"][name][key] += value


def _coverage(case: dict[str, Any], tracks_by_frame: list[list[dict]], gates: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    dt = float(case["dt_s"])
    for event in case.get("events", []):
        if event.get("blocker_id") is None:
            continue
        start = int(event["start_snapshot"])
        age = _age_at(case["frames"], tracks_by_frame, start, str(event["blocker_id"]), gates)
        leads = {}
        for lead in LEADS_S:
            index = int(round((float(event["start_s"]) - lead) / dt))
            leads[str(lead)] = _tracked(case["frames"], tracks_by_frame, index, str(event["blocker_id"]), gates)
        rows.append(
            {
                "class": str(event.get("blocker_kind")),
                "mount": case["mount"],
                "age_s": age,
                "leads": leads,
            }
        )
    return rows


def _tracked(frames, tracks_by_frame, index, blocker_id, gates) -> bool:
    if index < 0 or index >= len(frames):
        return False
    target = next((row for row in frames[index]["ground_truth"] if row["id"] == blocker_id), None)
    if target is None:
        return False
    gate = gates[_class(str(target["kind"]))]
    for track in tracks_by_frame[index]:
        if not track["confirmed"]:
            continue
        if np.hypot(track["x_m"] - target["x_m"], track["y_m"] - target["y_m"]) <= gate:
            return True
    return False


def _age_at(frames, tracks_by_frame, index, blocker_id, gates) -> float | None:
    if index < 0 or index >= len(frames):
        return None
    target = next((row for row in frames[index]["ground_truth"] if row["id"] == blocker_id), None)
    if target is None:
        return None
    gate = gates[_class(str(target["kind"]))]
    ages = [
        float(track["age_s"])
        for track in tracks_by_frame[index]
        if track["confirmed"] and np.hypot(track["x_m"] - target["x_m"], track["y_m"] - target["y_m"]) <= gate
    ]
    return min(ages) if ages else None


def _class(kind: str) -> str:
    if kind in ("bus", "truck", "bus/truck"):
        return "bus/truck"
    if kind == "car":
        return "car"
    return "pedestrian"


def _report(raw, wave, wide, grid, picks, eval_cases, spec, sizes, centroid_note: str) -> tuple[str, dict]:
    from sim.sensing.provenance import version

    gates = scaled_gates(1.0)
    tables = []
    events = []
    sensitivity = []
    for label, key in (("1024", "narrow"), ("2048", "broad")):
        for pick in picks:
            point = pick["point"]
            if point is None:
                tables.append({**pick, "bandwidth": label, "mount": None, "summary": None})
                continue
            params = dict(point["params"])
            for mount in MOUNTS:
                total = empty_score()
                for case in eval_cases:
                    if case["mount"] != mount:
                        continue
                    view = dict(case[key])
                    view["events"] = case["narrow"]["events"]
                    view["mount"] = mount
                    part, _events, tracks = _replay(view, params, gates, spec, sizes)
                    _add(total, part)
                    if label == "1024":
                        covered = _coverage(view, tracks, gates)
                        for row in covered:
                            row["method"] = pick["method"]
                            row["clutter"] = pick["clutter"]
                            row["budget"] = pick["budget"]
                        events.extend(covered)
                tables.append({**pick, "bandwidth": label, "mount": mount, "summary": summarize(total)})
        for pick in picks:
            if pick["point"] is None or pick["clutter"] != "blind" or pick["method"] != "image":
                continue
            params = dict(pick["point"]["params"])
            for scale in GATE_SCALES:
                total = empty_score()
                for case in eval_cases:
                    part, _events, _tracks = _replay(dict(case[key]), params, scaled_gates(scale), spec, sizes)
                    _add(total, part)
                sensitivity.append(
                    {"bandwidth": label, "budget": pick["budget"], "scale": scale, "summary": summarize(total)}
                )
    prov = version()
    clutter_abs = max((float(case["narrow"].get("clutter_power_abs", 0.0)) for case in eval_cases), default=0.0)
    lines = [
        "# M2 report",
        "",
        "Status: ready for review. This milestone is not marked done. No operating point is selected.",
        "",
        "Regenerate this table from the channel caches with "
        "`docker compose run --rm sionna python -u scripts/run_m2_review.py --workers 4`. "
        "The loader refuses a cache whose git commit, uncommitted-change hash, or config hash differs.",
        "",
        f"git {prov['git_commit']} dirty {prov['dirty_hash']} config {prov['config_hash']}",
        "",
        "Runs are 60 s at dt = 0.1 s, one CPI per snapshot. "
        "The tracker is an EKF on range, azimuth, elevation, and radial velocity. "
        "Velocity starts from Doppler and, on the second hit, two-point differencing. "
        "Velocity RMSE is after 1 s of track age.",
        "",
        "The constraint of at most 2 unmatched clusters per CPI is infeasible on the tuning seeds. "
        "It is not changed. Unmatched clusters are split into fragments (inside a true target's "
        "bounding box expanded by 1 m), ghosts (mirror of that box across a known wall), and other. "
        "Only ghosts and other count as false alarms. "
        "Budgets of 2, 4 and 6 false alarms per CPI are the best tuning point under each budget, labelled as such. "
        "The operating point waits on the false-handover cost.",
        "",
        "Blind subtraction removes the slow-time mean. Twin subtraction removes the empty-scene path. "
        "No impairment was added.",
        f"Largest absolute power difference between the blind and twin maps on the first snapshot of each evaluation job, 1024 subcarriers: {clutter_abs:.3e}.",
        "",
        "Match gates are horizontal distances from the mesh origin: "
        f"pedestrian {scaled_gates(1)['pedestrian']} m, car {scaled_gates(1)['car']} m, bus/truck {scaled_gates(1)['bus/truck']} m. "
        "The 0.5× and 2× rows are sensitivity only. "
        + centroid_note,
        "",
        "## Tuning trade-off, all grid points",
        "",
        "Seeds 101–105, 1024 subcarriers. False alarms are ghost clusters plus other clusters per CPI.",
        "",
        "| Clutter | Method | Pfa | Train | Eps [m] | Coast | Assoc. [m] | Bus/truck track Pd | Unmatched / CPI | Fragment | Ghost | Other | False alarms / CPI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in grid:
        params = row["params"]
        summary = row["summary"]
        lines.append(
            "| {clutter} | {method} | {pfa:.0e} | {train} | {eps:.0f} | {coast} | {assoc:.0f} | {pd} | {unmatched} | {frag} | {ghost} | {other} | {fa} |".format(
                clutter=params["clutter"],
                method="image" if params["ghost"] else "none",
                pfa=float(params["cfar"]["pfa"]),
                train=int(params["cfar"]["train"]),
                eps=float(params["eps_m"]),
                coast=int(params["coast"]),
                assoc=float(params["association_m"]),
                pd=_fmt(row["track_pd"]),
                unmatched=_fmt(row["false_clusters_per_cpi"], 2),
                frag=_fmt(summary["cluster_fragment_per_cpi"], 2),
                ghost=_fmt(summary["cluster_ghost_per_cpi"], 2),
                other=_fmt(summary["cluster_other_per_cpi"], 2),
                fa=_fmt(row["false_alarms_per_cpi"], 2),
            )
        )
    lines.extend(
        [
            "",
            "## Labelled budgets",
            "",
            "Each row is the tuning point with the highest bus/truck track Pd whose false alarms per CPI are at most that budget. "
            "A budget with no such point is infeasible and is not evaluated.",
            "",
            "| Budget | Clutter | Method | Feasible | Pfa | Train | Eps | Coast | Assoc. | Tuning track Pd | Tuning FA / CPI |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for pick in picks:
        point = pick["point"]
        if point is None:
            lines.append(
                f"| {pick['budget']:.0f} | {pick['clutter']} | {pick['method']} | no | — | — | — | — | — | — | — |"
            )
            continue
        params = point["params"]
        lines.append(
            "| {budget:.0f} | {clutter} | {method} | yes | {pfa:.0e} | {train} | {eps:.0f} | {coast} | {assoc:.0f} | {pd} | {fa} |".format(
                budget=float(pick["budget"]),
                clutter=pick["clutter"],
                method=pick["method"],
                pfa=float(params["cfar"]["pfa"]),
                train=int(params["cfar"]["train"]),
                eps=float(params["eps_m"]),
                coast=int(params["coast"]),
                assoc=float(params["association_m"]),
                pd=_fmt(point["track_pd"]),
                fa=_fmt(point["false_alarms_per_cpi"], 2),
            )
        )
    lines.extend(
        [
            "",
            "## Evaluation at those budgets",
            "",
            "Seeds 1001–1010. The budget is a label, not a selected operating point. "
            "Evaluation false alarms are measured and are not forced under the budget.",
            "",
            "| Bandwidth | Budget | Clutter | Method | Mount | Class | GT | Track Pd | FA / CPI | Fragment | Ghost | Other | Vel. RMSE [m/s] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in tables:
        if row["summary"] is None:
            lines.append(
                f"| {row['bandwidth']} | {row['budget']:.0f} | {row['clutter']} | {row['method']} | — | — | — | infeasible | — | — | — | — | — |"
            )
            continue
        for name in CLASSES:
            stats = row["summary"]["classes"][name]
            lines.append(
                "| {bw} | {budget:.0f} | {clutter} | {method} | {mount} | {klass} | {gt} | {pd} | {fa} | {frag} | {ghost} | {other} | {vel} |".format(
                    bw=row["bandwidth"],
                    budget=float(row["budget"]),
                    clutter=row["clutter"],
                    method=row["method"],
                    mount=row["mount"],
                    klass=name,
                    gt=stats["gt_samples"],
                    pd=_fmt(stats["track_pd"]),
                    fa=_fmt(row["summary"]["false_alarms_per_cpi"], 2),
                    frag=_fmt(row["summary"]["cluster_fragment_per_cpi"], 2),
                    ghost=_fmt(row["summary"]["cluster_ghost_per_cpi"], 2),
                    other=_fmt(row["summary"]["cluster_other_per_cpi"], 2),
                    vel=_fmt(stats["velocity_rmse_mps"], 2),
                )
            )
    lines.extend(["", "## Gate sensitivity, blind clutter, image method, labelled budget", ""])
    lines.append("| Bandwidth | Budget | Scale | Class | Track Pd |")
    lines.append("|---|---|---|---|---|")
    for row in sensitivity:
        for name in CLASSES:
            stats = row["summary"]["classes"][name]
            lines.append(
                f"| {row['bandwidth']} | {row['budget']:.0f} | {row['scale']} | {name} | {_fmt(stats['track_pd'])} |"
            )
    lines.extend(["", "## Confirmed track before a 10 dB LoS event", ""])
    lines.append(
        "1024 subcarriers, at the labelled budget. Age is at event start. "
        "A blocker that was never confirmed inside its class gate is a miss and has no age."
    )
    lines.append("")
    lines.append("| Budget | Class | Mount | Method | Clutter | Events | Lead 0.1 s | 0.3 s | 0.5 s | 1.0 s | Age p50 [s] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for budget in FA_BUDGETS:
        for mount in MOUNTS:
            for name in CLASSES:
                for method in ("none", "image"):
                    for clutter in ("blind", "twin"):
                        subset = [
                            row
                            for row in events
                            if row["mount"] == mount
                            and _class(row["class"]) == name
                            and row["method"] == method
                            and row["clutter"] == clutter
                            and float(row["budget"]) == float(budget)
                        ]
                        if not subset:
                            continue
                        rates = []
                        for lead in LEADS_S:
                            hits = sum(1 for row in subset if row["leads"][str(lead)])
                            rates.append(f"{hits}/{len(subset)}")
                        ages = sorted(row["age_s"] for row in subset if row["age_s"] is not None)
                        p50 = "—" if not ages else f"{ages[len(ages)//2]:.2f}"
                        lines.append(
                            f"| {budget:.0f} | {name} | {mount} | {method} | {clutter} | {len(subset)} | "
                            + " | ".join(rates)
                            + f" | {p50} |"
                        )
    text = "\n".join(lines) + "\n"
    record = {
        "provenance": prov,
        "tables": tables,
        "sensitivity": sensitivity,
        "events": events,
        "grid": grid,
        "picks": picks,
    }
    return text, record


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


if __name__ == "__main__":
    main()
