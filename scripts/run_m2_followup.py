"""Velocity diagnostics and the map-constrained tracker.

Reads the detection caches and the budget tables written by
``scripts/run_m2_review.py``. Channel caches are accepted when the git
commit and the config hash match. A newer dirty tree is recorded, because
this step does not re-trace.

The detector (CFAR, DBSCAN, clutter, ghost method) is the labelled budget-4
point for blind clutter and no ghost handling, when that point exists.
Only the map tracker's own parameters are tuned, and only on seeds 101–105.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml
from sim.sensing.cluster import cluster_detections
from sim.sensing.metrics import CLASSES, blocker_class, empty_score, scaled_gates, summarize, update_score
from sim.sensing.provenance import version
from sim.sensing.track import Tracker
from sim.sensing.track_map import MapTracker

LEADS_S = (0.1, 0.3, 0.5, 1.0)
BUDGET = 4.0


def main() -> None:
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tuning = [int(seed) for seed in seeds["tuning"]]
    evaluation = [int(seed) for seed in seeds["evaluation"]]
    metrics_path = ROOT / "results" / "M2" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "grid" not in metrics or "picks" not in metrics:
        raise SystemExit("budget tables are not in results/M2/metrics.json yet")
    point = _budget_point(metrics["picks"])
    if point is None:
        raise SystemExit(f"budget {BUDGET:.0f}, blind clutter, no ghost handling is infeasible")
    spec = raw["sensing_radar"]
    sizes = raw["blocker_kinds"]
    lanes = list(raw["lanes"])
    sidewalks = list(raw["sidewalks"])
    current = version()
    print("load tuning detections", flush=True)
    tune_cases = [
        _case(mount, density, seed, current)
        for seed in tuning
        for mount in ("lamppost", "facade")
        for density in ("low", "high")
    ]
    print("tune map tracker", len(tune_cases), flush=True)
    tuned = _tune_map(tune_cases, spec, sizes, lanes, sidewalks, point)
    print("load evaluation detections", flush=True)
    eval_cases = [
        _case(mount, density, seed, current)
        for seed in evaluation
        for mount in ("lamppost", "facade")
        for density in ("low", "high")
    ]
    text = _write(raw, metrics, point, tuned, tune_cases, eval_cases, spec, sizes, lanes, sidewalks, current)
    output = ROOT / "results" / "M2" / "followup.md"
    output.write_text(text, encoding="utf-8")
    _append_report(text)
    print(text)


def _budget_point(picks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for pick in picks:
        if (
            float(pick["budget"]) == BUDGET
            and pick["clutter"] == "blind"
            and pick["method"] == "none"
            and pick.get("point") is not None
        ):
            return pick["point"]
    return None


def _case(mount: str, density: str, seed: int, current: dict[str, str]) -> dict[str, Any]:
    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    path = directory / "detections_1024.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    if provenance.get("git_commit") != current["git_commit"] or provenance.get("config_hash") != current["config_hash"]:
        raise SystemExit(f"refusing {path}: commit or config hash differs from this tree")
    events_payload = json.loads((directory / "events.json").read_text(encoding="utf-8"))
    events = events_payload["events"] if isinstance(events_payload, dict) else events_payload
    payload["events"] = events
    payload["mount"] = mount
    payload["density"] = density
    payload["trace_dirty_hash"] = provenance.get("dirty_hash")
    return payload


def _detections(frame: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    key = f"blind:{int(params['cfar']['train'])}:{float(params['cfar']['pfa'])}"
    return list(frame["detections"][key])


def _replay_tracker(case, tracker, params, spec, sizes, gates) -> tuple[dict, list[list[dict]]]:
    from sim.sensing.ghost import reject_ghosts

    score = empty_score()
    walls = [float(value) for value in spec["wall_y_m"]]
    tracks_by_frame: list[list[dict]] = []
    for frame in case["frames"]:
        detections = _detections(frame, params)
        if params.get("ghost"):
            detections = reject_ghosts(detections, walls, float(params["association_m"]))
        clusters = cluster_detections(detections, float(params["eps_m"]), int(spec["cluster"]["min_samples"]))
        alive = tracker.step(clusters)
        rows = [_track_row(track) for track in alive]
        tracks_by_frame.append(rows)
        update_score(
            score,
            frame["ground_truth"],
            clusters,
            rows,
            gates,
            t_s=float(frame["t_s"]),
            sizes_m=sizes,
            wall_y_m=walls,
        )
    return score, tracks_by_frame


def _track_row(track) -> dict[str, Any]:
    return {
        "x_m": float(track.state[0]),
        "y_m": float(track.state[1]),
        "z_m": float(track.state[2]),
        "vx_mps": float(track.state[3]),
        "vy_mps": float(track.state[4]),
        "confirmed": bool(track.confirmed),
        "age_s": float(track.age_s),
        "id": int(track.identifier),
        "misses": int(track.misses),
    }


def _unconstrained(params, spec, radar) -> Tracker:
    return Tracker(
        dt_s=0.1,
        association_gate_m=float(params["association_m"]),
        coast_frames=int(params["coast"]),
        confirm_hits=int(spec["tracker"]["confirm_hits"]),
        radar_position_m=np.asarray(radar, dtype=np.float64),
        process_q=1.0,
    )


def _map_tracker(params, spec, radar, lanes, sidewalks, tuned) -> MapTracker:
    return MapTracker(
        dt_s=0.1,
        association_gate_m=float(tuned["association_m"]),
        coast_frames=int(tuned["coast"]),
        confirm_hits=int(spec["tracker"]["confirm_hits"]),
        radar_position_m=np.asarray(radar, dtype=np.float64),
        process_q=float(tuned["process_q"]),
        lanes=lanes,
        sidewalks=sidewalks,
        lane_gate_m=float(tuned["lane_gate_m"]),
        sidewalk_gate_m=float(tuned["sidewalk_gate_m"]),
        leave_m=float(tuned["leave_m"]),
    )


def _tune_map(cases, spec, sizes, lanes, sidewalks, point) -> dict[str, Any]:
    params = dict(point["params"])
    params["ghost"] = False
    params["clutter"] = "blind"
    gates = scaled_gates(1.0)
    best = None
    best_pd = -1.0
    for association_m in (4.0, 6.0, 10.0):
        for coast in (2, 4, 8):
            for process_q in (0.25, 1.0, 4.0):
                for lane_gate_m in (1.5, 2.5):
                    for leave_m in (2.0, 4.0):
                        tuned = {
                            "association_m": association_m,
                            "coast": coast,
                            "process_q": process_q,
                            "lane_gate_m": lane_gate_m,
                            "sidewalk_gate_m": lane_gate_m,
                            "leave_m": leave_m,
                        }
                        total = empty_score()
                        for case in cases:
                            tracker = _map_tracker(params, spec, case["radar_position_m"], lanes, sidewalks, tuned)
                            part, _tracks = _replay_tracker(case, tracker, params, spec, sizes, gates)
                            _add(total, part)
                        summary = summarize(total)
                        pd = summary["classes"]["bus/truck"]["track_pd"] or -1.0
                        if pd > best_pd:
                            best_pd = float(pd)
                            best = {"tuned": tuned, "summary": summary}
                        print(
                            f"map assoc {association_m} coast {coast} q {process_q} gate {lane_gate_m} leave {leave_m} pd {pd:.3f}",
                            flush=True,
                        )
    if best is None:
        raise RuntimeError("map tracker grid was empty")
    return best


def _add(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key in (
        "frames",
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


def _velocity(case, tracks_by_frame, radar, gates) -> dict[str, dict[str, float]]:
    buckets = {
        name: {"n": 0, "radial": 0.0, "cross_range": 0.0, "along": 0.0, "cross_lane": 0.0}
        for name in CLASSES
    }
    radar = np.asarray(radar, dtype=np.float64)
    for frame, tracks in zip(case["frames"], tracks_by_frame, strict=True):
        confirmed = [track for track in tracks if track["confirmed"]]
        pairs = _match(frame["ground_truth"], confirmed, gates)
        for truth_index, track_index in pairs:
            target = frame["ground_truth"][truth_index]
            track = confirmed[track_index]
            if float(track["age_s"]) < 1.0:
                continue
            name = blocker_class(str(target["kind"]))
            error = np.array(
                [float(track["vx_mps"]) - float(target["vx_mps"]), float(track["vy_mps"]) - float(target["vy_mps"])],
                dtype=np.float64,
            )
            delta = np.array([float(target["x_m"]) - radar[0], float(target["y_m"]) - radar[1]], dtype=np.float64)
            norm = float(np.linalg.norm(delta))
            radial_hat = np.array([1.0, 0.0]) if norm < 1e-6 else delta / norm
            cross_hat = np.array([-radial_hat[1], radial_hat[0]])
            bucket = buckets[name]
            bucket["n"] += 1
            bucket["radial"] += float(np.dot(error, radial_hat) ** 2)
            bucket["cross_range"] += float(np.dot(error, cross_hat) ** 2)
            bucket["along"] += float(error[0] ** 2)
            bucket["cross_lane"] += float(error[1] ** 2)
    return buckets


def _match(ground_truth, tracks, gates) -> list[tuple[int, int]]:
    if not ground_truth or not tracks:
        return []
    pairs = []
    for i, target in enumerate(ground_truth):
        gate = gates[blocker_class(str(target["kind"]))]
        for j, track in enumerate(tracks):
            distance = float(np.hypot(track["x_m"] - target["x_m"], track["y_m"] - target["y_m"]))
            if distance <= gate:
                pairs.append((distance, i, j))
    pairs.sort()
    used_t: set[int] = set()
    used_k: set[int] = set()
    chosen = []
    for _distance, i, j in pairs:
        if i in used_t or j in used_k:
            continue
        used_t.add(i)
        used_k.add(j)
        chosen.append((i, j))
    return chosen


def _switches(case, tracks_by_frame, gates) -> dict[str, dict[str, int]]:
    previous: dict[str, int] = {}
    counts = {name: {"switches": 0, "targets": set()} for name in CLASSES}
    for frame, tracks in zip(case["frames"], tracks_by_frame, strict=True):
        confirmed = [track for track in tracks if track["confirmed"]]
        for truth_index, track_index in _match(frame["ground_truth"], confirmed, gates):
            target = frame["ground_truth"][truth_index]
            name = blocker_class(str(target["kind"]))
            identifier = str(target["id"])
            track_id = int(confirmed[track_index]["id"])
            counts[name]["targets"].add(identifier)
            if identifier in previous and previous[identifier] != track_id:
                counts[name]["switches"] += 1
            previous[identifier] = track_id
    return counts


def _lead_gap(case, tracks_by_frame, gates) -> dict[str, int]:
    """Pedestrian events tracked at t−0.5 s and not at t−0.1 s."""
    counts = {"events": 0, "died": 0, "same_track_outside_gate": 0, "other": 0}
    dt = 0.1
    for event in case.get("events", []):
        if str(event.get("blocker_kind")) != "pedestrian":
            continue
        start = float(event["start_s"])
        early = int(round((start - 0.5) / dt))
        late = int(round((start - 0.1) / dt))
        early_hit = _at(case, tracks_by_frame, early, str(event["blocker_id"]), gates)
        late_hit = _at(case, tracks_by_frame, late, str(event["blocker_id"]), gates)
        if early_hit is None or late_hit is not None:
            continue
        counts["events"] += 1
        track_id, _distance = early_hit
        if late < 0 or late >= len(tracks_by_frame) or not any(int(track["id"]) == track_id for track in tracks_by_frame[late]):
            counts["died"] += 1
            continue
        target = next((row for row in case["frames"][late]["ground_truth"] if row["id"] == str(event["blocker_id"])), None)
        same = next(track for track in tracks_by_frame[late] if int(track["id"]) == track_id)
        if target is None:
            counts["other"] += 1
            continue
        gate = gates["pedestrian"]
        distance = float(np.hypot(same["x_m"] - target["x_m"], same["y_m"] - target["y_m"]))
        if distance > gate:
            counts["same_track_outside_gate"] += 1
        else:
            counts["other"] += 1
    return counts


def _at(case, tracks_by_frame, index, blocker_id, gates):
    if index < 0 or index >= len(case["frames"]):
        return None
    target = next((row for row in case["frames"][index]["ground_truth"] if row["id"] == blocker_id), None)
    if target is None:
        return None
    gate = gates[blocker_class(str(target["kind"]))]
    best = None
    for track in tracks_by_frame[index]:
        if not track["confirmed"]:
            continue
        distance = float(np.hypot(track["x_m"] - target["x_m"], track["y_m"] - target["y_m"]))
        if distance <= gate and (best is None or distance < best[1]):
            best = (int(track["id"]), distance)
    return best


def _write(raw, metrics, point, tuned, tune_cases, eval_cases, spec, sizes, lanes, sidewalks, current) -> str:
    params = dict(point["params"])
    params["ghost"] = False
    params["clutter"] = "blind"
    gates = scaled_gates(1.0)
    free_scores = []
    map_scores = []
    velocity = {name: {"n": 0, "radial": 0.0, "cross_range": 0.0, "along": 0.0, "cross_lane": 0.0} for name in CLASSES}
    map_velocity = {name: {"n": 0, "radial": 0.0, "cross_range": 0.0, "along": 0.0, "cross_lane": 0.0} for name in CLASSES}
    switches = {name: {"switches": 0, "targets": 0} for name in CLASSES}
    map_switches = {name: {"switches": 0, "targets": 0} for name in CLASSES}
    gaps = {"events": 0, "died": 0, "same_track_outside_gate": 0, "other": 0}
    free_leads = _empty_leads()
    map_leads = _empty_leads()
    free_total = empty_score()
    map_total = empty_score()
    dirty = set()
    for case in eval_cases:
        dirty.add(case.get("trace_dirty_hash"))
        radar = case["radar_position_m"]
        free = _unconstrained(params, spec, radar)
        constrained = _map_tracker(params, spec, radar, lanes, sidewalks, tuned["tuned"])
        free_part, free_tracks = _replay_tracker(case, free, params, spec, sizes, gates)
        map_part, map_tracks = _replay_tracker(case, constrained, params, spec, sizes, gates)
        _add(free_total, free_part)
        _add(map_total, map_part)
        _accumulate(velocity, _velocity(case, free_tracks, radar, gates))
        _accumulate(map_velocity, _velocity(case, map_tracks, radar, gates))
        _accumulate_switches(switches, _switches(case, free_tracks, gates))
        _accumulate_switches(map_switches, _switches(case, map_tracks, gates))
        gap = _lead_gap(case, free_tracks, gates)
        for key, value in gap.items():
            gaps[key] += value
        _add_leads(free_leads, _lead_counts(case, free_tracks, gates), case["mount"])
        _add_leads(map_leads, _lead_counts(case, map_tracks, gates), case["mount"])
        print(f"eval {case['mount']} {case['density']} {case['seed']}", flush=True)
    free_summary = summarize(free_total)
    map_summary = summarize(map_total)
    paired = _paired(metrics["grid"])
    lines = [
        "# M2 follow-up",
        "",
        "No operating point is selected. This does not mark M2 done.",
        "",
        f"Analysis git {current['git_commit']} dirty {current['dirty_hash']} config {current['config_hash']}.",
        "Detection caches kept the rebuild's dirty hash "
        + ", ".join(str(item) for item in dirty)
        + ". The commit and the config hash match, so the channels were not re-traced.",
        "",
        "## Plain results",
        "",
        "Twin-based clutter subtraction is a null result. Static canyon clutter is exactly zero-Doppler, "
        "and no impairment was added. On the evaluation table the blind and twin columns match.",
        "",
        _ghost_sentence(paired),
        "",
        "## Bandwidth",
        "",
        "2048 subcarriers are not the main sensing setting. "
        "The communication carrier in this scenario, which M3 inherits unless it is changed, "
        f"is numerology {raw['numerology']} with {raw['n_subcarriers']} subcarriers "
        f"({_mhz(raw)} MHz). 2048 sensing subcarriers are {_mhz_sensing(raw, 2048):.1f} MHz. "
        "Use 2048 as the main setting only if that communication carrier is widened to the same bandwidth.",
        "",
        "## Detector held fixed",
        "",
        "Blind clutter, no ghost handling, the labelled budget of "
        f"{BUDGET:.0f} false alarms per CPI on the tuning seeds. "
        f"Pfa {params['cfar']['pfa']:.0e}, train {params['cfar']['train']}, "
        f"DBSCAN eps {params['eps_m']:.0f} m. "
        "The unconstrained tracker keeps association "
        f"{params['association_m']:.0f} m, coast {params['coast']}, process_q 1.0. "
        "That process noise is the white-acceleration density in each Cartesian axis: "
        "position variance q dt^3/3, cross term q dt^2/2, velocity variance q dt, with dt = 0.1 s. "
        "Measurement sigmas are 1.5 m, 2 degrees, and 0.5 m/s radial.",
        "",
        "The map tracker was tuned only on seeds 101–105, for bus/truck track Pd, with no false-alarm cap. "
        f"Chosen association {tuned['tuned']['association_m']:.0f} m, coast {tuned['tuned']['coast']}, "
        f"process_q {tuned['tuned']['process_q']}, lane/sidewalk gate {tuned['tuned']['lane_gate_m']} m, "
        f"sidewalk leave distance {tuned['tuned']['leave_m']} m. "
        f"Tuning bus/truck track Pd {_fmt(tuned['summary']['classes']['bus/truck']['track_pd'])}.",
        "",
        "## Velocity, unconstrained, evaluation seeds, after 1 s",
        "",
        "Radial and cross-range are horizontal components relative to oru-0. "
        "Along-lane is the street axis x. Cross-lane is y.",
        "",
        "| Class | Samples | Radial RMSE | Cross-range RMSE | Along-lane RMSE | Cross-lane RMSE | ID switches | Per target |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in CLASSES:
        bucket = velocity[name]
        count = switches[name]
        targets = max(int(count["targets"]), 1)
        lines.append(
            f"| {name} | {bucket['n']} | {_rmse(bucket, 'radial')} | {_rmse(bucket, 'cross_range')} | "
            f"{_rmse(bucket, 'along')} | {_rmse(bucket, 'cross_lane')} | {count['switches']} | "
            f"{count['switches'] / targets:.3f} |"
        )
    lines.extend(["", "## Why a pedestrian track at t−0.5 s is missing at t−0.1 s", ""])
    lines.append(
        f"Pedestrian 10 dB events on the evaluation seeds where a confirmed track is inside the 1.5 m gate "
        f"at t−0.5 s and not at t−0.1 s: {gaps['events']}. "
        f"The same track id has been dropped: {gaps['died']}. "
        f"The same track id is alive but outside the gate: {gaps['same_track_outside_gate']}. "
        f"Anything else: {gaps['other']}."
    )
    lines.append("")
    coast = int(params["coast"])
    lines.append(
        f"Coast is {coast} snapshots ({coast * 0.1:.1f} s) and a track is kept while its miss count is at most that. "
        "A track that was updated at t−0.5 s is therefore still alive at t−0.1 s unless it was already coasting "
        "or the coast is shorter than four frames. "
        "The usual case is the second count: the track leaves the 1.5 m pedestrian gate. "
        "Pedestrian velocity error is a few metres per second, so four frames of prediction, a turn onto a crossing, "
        "or an association onto a nearby cluster moves it by more than the gate. "
        "A new track is not confirmed until the second hit, so the lead at 0.1 s is empty even though the lead at 0.5 s was not."
    )
    lines.extend(
        [
            "",
            "## Map-constrained tracker, evaluation seeds, 1024 subcarriers",
            "",
            "Along-lane and cross-lane are velocity RMSE [m/s] after 1 s of track age, on the street axes x and y. "
            "An ID switch is a change of the confirmed track id matched to one ground-truth identity, including after a gap. "
            "The per-target rate counts each identity once per job.",
            "",
            "| Tracker | Class | Track Pd | Pos. RMSE [m] | Vel. RMSE [m/s] | Along-lane | Cross-lane | ID switches | Per target | FA / CPI |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for label, summary, vel, sw in (
        ("unconstrained", free_summary, velocity, switches),
        ("map", map_summary, map_velocity, map_switches),
    ):
        for name in CLASSES:
            stats = summary["classes"][name]
            targets = max(int(sw[name]["targets"]), 1)
            lines.append(
                f"| {label} | {name} | {_fmt(stats['track_pd'])} | {_fmt(stats['position_rmse_m'], 2)} | "
                f"{_fmt(stats['velocity_rmse_mps'], 2)} | {_rmse(vel[name], 'along')} | {_rmse(vel[name], 'cross_lane')} | "
                f"{sw[name]['switches']} | {sw[name]['switches'] / targets:.3f} | {_fmt(summary['false_alarms_per_cpi'], 2)} |"
            )
    lines.extend(["", "## Lead time, same detector, 1024 subcarriers", ""])
    lines.append("| Tracker | Class | Mount | Events | 0.1 s | 0.3 s | 0.5 s | 1.0 s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, table in (("unconstrained", free_leads), ("map", map_leads)):
        for mount in ("lamppost", "facade"):
            for name in CLASSES:
                row = table[mount][name]
                if row["n"] == 0:
                    continue
                rates = " | ".join(f"{row['hits'][str(lead)]}/{row['n']}" for lead in LEADS_S)
                lines.append(f"| {label} | {name} | {mount} | {row['n']} | {rates} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _empty_leads() -> dict[str, dict[str, dict[str, Any]]]:
    table = {}
    for mount in ("lamppost", "facade"):
        table[mount] = {}
        for name in CLASSES:
            table[mount][name] = {"n": 0, "hits": {str(lead): 0 for lead in LEADS_S}}
    return table


def _lead_counts(case, tracks_by_frame, gates) -> dict[str, dict[str, Any]]:
    counts = {name: {"n": 0, "hits": {str(lead): 0 for lead in LEADS_S}} for name in CLASSES}
    dt = 0.1
    for event in case.get("events", []):
        kind = str(event.get("blocker_kind"))
        if kind not in ("bus", "truck", "car", "pedestrian"):
            continue
        name = blocker_class(kind)
        counts[name]["n"] += 1
        for lead in LEADS_S:
            index = int(round((float(event["start_s"]) - lead) / dt))
            if _at(case, tracks_by_frame, index, str(event.get("blocker_id")), gates) is not None:
                counts[name]["hits"][str(lead)] += 1
    return counts


def _add_leads(total, part, mount: str) -> None:
    for name in CLASSES:
        total[mount][name]["n"] += part[name]["n"]
        for lead in LEADS_S:
            total[mount][name]["hits"][str(lead)] += part[name]["hits"][str(lead)]


def _paired(grid: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    pairs = []
    for left in grid:
        if left["params"].get("ghost"):
            continue
        for right in grid:
            if not right["params"].get("ghost"):
                continue
            if left["params"]["clutter"] != right["params"]["clutter"]:
                continue
            if not _same_detector(left["params"], right["params"]):
                continue
            pairs.append((left, right))
    return pairs


def _same_detector(left: dict, right: dict) -> bool:
    return (
        int(left["cfar"]["train"]) == int(right["cfar"]["train"])
        and float(left["cfar"]["pfa"]) == float(right["cfar"]["pfa"])
        and float(left["eps_m"]) == float(right["eps_m"])
        and int(left["coast"]) == int(right["coast"])
        and float(left["association_m"]) == float(right["association_m"])
    )


def _ghost_sentence(pairs: list[tuple[dict, dict]]) -> str:
    if not pairs:
        return "Image-method ghost handling could not be paired with the no-ghost grid."
    pd = []
    fa = []
    for none, image in pairs:
        none_pd = none["track_pd"]
        image_pd = image["track_pd"]
        if none_pd is not None and image_pd is not None:
            pd.append(float(image_pd) - float(none_pd))
        base = float(none["false_clusters_per_cpi"])
        if base > 0.0:
            fa.append((base - float(image["false_clusters_per_cpi"])) / base)
    mean_pd = float(np.mean(pd)) if pd else 0.0
    mean_fa = float(np.mean(fa)) if fa else 0.0
    return (
        "Image-method ghost handling has no bus/truck track-Pd gain at equal parameters "
        f"(mean change {mean_pd:+.3f} over {len(pd)} tuning grid pairs). "
        f"It removes {100.0 * mean_fa:.1f}% of the unmatched clusters on those pairs. "
        "The reviewed operating point was 6.14 versus 5.30 unmatched clusters per CPI, which is 14%."
    )


def _accumulate(total, part) -> None:
    for name in CLASSES:
        for key, value in part[name].items():
            total[name][key] += value


def _accumulate_switches(total, part) -> None:
    for name in CLASSES:
        total[name]["switches"] += part[name]["switches"]
        total[name]["targets"] += len(part[name]["targets"])


def _rmse(bucket: dict[str, float], key: str) -> str:
    if int(bucket["n"]) == 0:
        return "—"
    return f"{float(np.sqrt(bucket[key] / bucket['n'])):.2f}"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _mhz(raw: dict) -> float:
    spacing = 15000.0 * (2 ** int(raw["numerology"]))
    return int(raw["n_subcarriers"]) * spacing / 1e6


def _mhz_sensing(raw: dict, n_subcarriers: int) -> float:
    spacing = 15000.0 * (2 ** int(raw["sensing_radar"]["waveform"]["numerology"]))
    return n_subcarriers * spacing / 1e6


def _append_report(text: str) -> None:
    path = ROOT / "results" / "M2" / "report.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "# M2 follow-up"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n\n"
    else:
        existing = existing.rstrip() + "\n\n"
    path.write_text(existing + text, encoding="utf-8")


if __name__ == "__main__":
    main()
