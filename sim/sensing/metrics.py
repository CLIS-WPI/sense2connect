"""Detection and track scores against ground-truth identities."""

from __future__ import annotations

from typing import Any

import math

import numpy as np

CLASSES = ("bus/truck", "pedestrian", "car")
_GROUP = {"bus": "bus/truck", "truck": "bus/truck", "pedestrian": "pedestrian", "car": "car"}
# Horizontal gates from the scattering-point centroid. The five-point
# vehicle model and the human model are centered on the mesh origin, so
# the centroid used here is the ground-truth center.
CLASS_GATES_M = {"pedestrian": 1.5, "car": 3.0, "bus/truck": 6.0}
BOX_MARGIN_M = 1.0


def blocker_class(kind: str) -> str:
    """Map a target kind onto the reported class."""
    if kind not in _GROUP:
        raise KeyError(f"unknown blocker kind {kind}")
    return _GROUP[kind]


def match_gate(points_a: np.ndarray, points_b: np.ndarray, gate_m: float) -> list[tuple[int, int]]:
    """Greedy one-to-one pairs whose horizontal distance is within ``gate_m``."""
    if len(points_a) == 0 or len(points_b) == 0:
        return []
    distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
    pairs: list[tuple[float, int, int]] = []
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if distances[i, j] <= gate_m:
                pairs.append((float(distances[i, j]), i, j))
    pairs.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    chosen: list[tuple[int, int]] = []
    for _distance, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        chosen.append((i, j))
    return chosen


def ospa_m(ground_truth: np.ndarray, estimates: np.ndarray, cutoff_m: float) -> float | None:
    """First-order OSPA [m]. ``None`` when there is no ground truth."""
    n_truth = len(ground_truth)
    if n_truth == 0:
        return None
    n_est = len(estimates)
    order = max(n_truth, n_est)
    if n_est == 0:
        return float(cutoff_m)
    pairs = match_gate(ground_truth, estimates, cutoff_m)
    cost = 0.0
    for i, j in pairs:
        cost += float(np.linalg.norm(ground_truth[i] - estimates[j]))
    unmatched = order - len(pairs)
    cost += unmatched * float(cutoff_m)
    return cost / float(order)


def empty_score() -> dict[str, Any]:
    """Accumulators for one variant."""
    per_class = {
        name: {
            "gt": 0,
            "det_hit": 0,
            "track_hit": 0,
            "pos_err_sq": 0.0,
            "pos_n": 0,
            "vel_err_sq": 0.0,
            "vel_n": 0,
            "ospa_sum": 0.0,
            "ospa_n": 0,
        }
        for name in CLASSES
    }
    return {
        "frames": 0,
        "false_clusters": 0,
        "cluster_fragment": 0,
        "cluster_ghost": 0,
        "cluster_other": 0,
        "false_alarms": 0,
        "confirmed_updates": 0,
        "unmatched_confirmed": 0,
        "track_fragment": 0,
        "track_ghost": 0,
        "track_other": 0,
        "classes": per_class,
    }


def scaled_gates(scale: float) -> dict[str, float]:
    """Class gates [m] multiplied by ``scale``. Nominal scale is 1."""
    return {name: float(CLASS_GATES_M[name]) * float(scale) for name in CLASSES}


def update_score(
    score: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    gate_m: float | dict[str, float],
    *,
    t_s: float = 0.0,
    warmup_s: float = 1.0,
    sizes_m: dict[str, dict[str, float]] | None = None,
    wall_y_m: list[float] | None = None,
) -> None:
    """Fold one CPI into ``score``.

    Each ground-truth identity uses its class gate [m]. A float ``gate_m``
    is that same radius for every class. Velocity error starts after
    ``warmup_s`` of track age. Wrapped identities are already distinct.
    """
    if isinstance(gate_m, dict):
        gates = {name: float(gate_m[name]) for name in CLASSES}
    else:
        gates = {name: float(gate_m) for name in CLASSES}
    score["frames"] += 1
    cluster_xy = _xy(clusters)
    confirmed = [track for track in tracks if track.get("confirmed")]
    track_xy = _xy(confirmed)
    truth_xy = _xy(ground_truth)
    cluster_pairs = _match_classes(ground_truth, truth_xy, cluster_xy, gates)
    track_pairs = _match_classes(ground_truth, truth_xy, track_xy, gates)
    unmatched_clusters = [clusters[j] for j in range(len(clusters)) if j not in {pair[1] for pair in cluster_pairs}]
    unmatched_tracks = [confirmed[j] for j in range(len(confirmed)) if j not in {pair[1] for pair in track_pairs}]
    score["false_clusters"] += len(unmatched_clusters)
    score["confirmed_updates"] += len(confirmed)
    score["unmatched_confirmed"] += len(unmatched_tracks)
    if sizes_m is not None:
        walls = [] if wall_y_m is None else wall_y_m
        for prefix, rows in (("cluster", unmatched_clusters), ("track", unmatched_tracks)):
            fragment, ghost, other = split_false(rows, ground_truth, sizes_m, walls)
            score[f"{prefix}_fragment"] += fragment
            score[f"{prefix}_ghost"] += ghost
            score[f"{prefix}_other"] += other
            if prefix == "cluster":
                score["false_alarms"] += ghost + other
    det_hit = {i for i, _j in cluster_pairs}
    track_of = {i: j for i, j in track_pairs}
    by_class: dict[str, list[int]] = {name: [] for name in CLASSES}
    for index, target in enumerate(ground_truth):
        by_class[blocker_class(str(target["kind"]))].append(index)
    for name, indices in by_class.items():
        bucket = score["classes"][name]
        bucket["gt"] += len(indices)
        if not indices:
            continue
        target_xy = truth_xy[indices]
        matched_rows = []
        for index in indices:
            if index in det_hit:
                bucket["det_hit"] += 1
            if index not in track_of:
                continue
            bucket["track_hit"] += 1
            j = track_of[index]
            error = truth_xy[index] - track_xy[j]
            bucket["pos_err_sq"] += float(error @ error)
            bucket["pos_n"] += 1
            age_s = float(confirmed[j].get("age_s", t_s))
            if age_s >= warmup_s:
                velocity = np.array([float(confirmed[j]["vx_mps"]), float(confirmed[j]["vy_mps"])], dtype=np.float64)
                truth_v = np.array(
                    [float(ground_truth[index]["vx_mps"]), float(ground_truth[index]["vy_mps"])],
                    dtype=np.float64,
                )
                delta = velocity - truth_v
                bucket["vel_err_sq"] += float(delta @ delta)
                bucket["vel_n"] += 1
            matched_rows.append(track_xy[j])
        matched = np.array(matched_rows, dtype=np.float64) if matched_rows else np.zeros((0, 2))
        value = ospa_m(target_xy, matched, gates[name])
        if value is not None:
            bucket["ospa_sum"] += value
            bucket["ospa_n"] += 1


def _match_classes(
    ground_truth: list[dict[str, Any]],
    truth_xy: np.ndarray,
    other_xy: np.ndarray,
    gates_m: dict[str, float],
) -> list[tuple[int, int]]:
    """Greedy pairs. The gate is the ground-truth identity's class gate."""
    if len(truth_xy) == 0 or len(other_xy) == 0:
        return []
    distances = np.linalg.norm(truth_xy[:, None, :] - other_xy[None, :, :], axis=2)
    pairs: list[tuple[float, int, int]] = []
    for i in range(distances.shape[0]):
        gate = gates_m[blocker_class(str(ground_truth[i]["kind"]))]
        for j in range(distances.shape[1]):
            if distances[i, j] <= gate:
                pairs.append((float(distances[i, j]), i, j))
    pairs.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    chosen: list[tuple[int, int]] = []
    for _distance, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        chosen.append((i, j))
    return chosen


def _xy(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([[float(row["x_m"]), float(row["y_m"])] for row in rows], dtype=np.float64)


def summarize(score: dict[str, Any]) -> dict[str, Any]:
    """Rates and RMSEs from accumulators."""
    frames = max(int(score["frames"]), 1)
    classes = {}
    for name, bucket in score["classes"].items():
        gt = int(bucket["gt"])
        classes[name] = {
            "gt_samples": gt,
            "detection_pd": None if gt == 0 else bucket["det_hit"] / gt,
            "track_pd": None if gt == 0 else bucket["track_hit"] / gt,
            "position_rmse_m": None if bucket["pos_n"] == 0 else float(np.sqrt(bucket["pos_err_sq"] / bucket["pos_n"])),
            "velocity_rmse_mps": None if bucket["vel_n"] == 0 else float(np.sqrt(bucket["vel_err_sq"] / bucket["vel_n"])),
            "ospa_m": None if bucket["ospa_n"] == 0 else bucket["ospa_sum"] / bucket["ospa_n"],
        }
    confirmed = int(score["confirmed_updates"])
    return {
        "false_clusters_per_cpi": score["false_clusters"] / frames,
        "false_alarms_per_cpi": score["false_alarms"] / frames,
        "cluster_fragment_per_cpi": score["cluster_fragment"] / frames,
        "cluster_ghost_per_cpi": score["cluster_ghost"] / frames,
        "cluster_other_per_cpi": score["cluster_other"] / frames,
        "track_fragment_per_cpi": score["track_fragment"] / frames,
        "track_ghost_per_cpi": score["track_ghost"] / frames,
        "track_other_per_cpi": score["track_other"] / frames,
        "ghost_track_rate": None if confirmed == 0 else score["unmatched_confirmed"] / confirmed,
        "classes": classes,
    }


def split_false(
    rows: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    sizes_m: dict[str, dict[str, float]],
    wall_y_m: list[float],
) -> tuple[int, int, int]:
    """Count unmatched rows as fragment, ghost, or other.

    A fragment lies in a true target's horizontal bounding box expanded by
    1 m. A ghost is the mirror of such a box across a known wall and is not
    a fragment. Only ghost and other are false alarms.
    """
    fragment = ghost = other = 0
    for row in rows:
        point = np.array([float(row["x_m"]), float(row["y_m"])], dtype=np.float64)
        if any(_in_box(point, target, sizes_m) for target in ground_truth):
            fragment += 1
        elif any(_is_mirror(point, target, sizes_m, wall_y_m) for target in ground_truth):
            ghost += 1
        else:
            other += 1
    return fragment, ghost, other


def _in_box(point: np.ndarray, target: dict[str, Any], sizes_m: dict[str, dict[str, float]]) -> bool:
    kind = str(target["kind"])
    spec = sizes_m[kind]
    half_length = 0.5 * float(spec["length_m"]) + BOX_MARGIN_M
    half_width = 0.5 * float(spec["width_m"]) + BOX_MARGIN_M
    velocity = np.array([float(target.get("vx_mps", 0.0)), float(target.get("vy_mps", 0.0))], dtype=np.float64)
    if float(np.linalg.norm(velocity)) < 0.3:
        heading = 0.0
    else:
        heading = math.atan2(float(velocity[1]), float(velocity[0]))
    delta = point - np.array([float(target["x_m"]), float(target["y_m"])], dtype=np.float64)
    cosine, sine = float(np.cos(heading)), float(np.sin(heading))
    along = cosine * delta[0] + sine * delta[1]
    across = -sine * delta[0] + cosine * delta[1]
    return abs(along) <= half_length and abs(across) <= half_width


def _is_mirror(
    point: np.ndarray,
    target: dict[str, Any],
    sizes_m: dict[str, dict[str, float]],
    wall_y_m: list[float],
) -> bool:
    for wall in wall_y_m:
        mirrored = np.array([point[0], 2.0 * float(wall) - point[1]], dtype=np.float64)
        if _in_box(mirrored, target, sizes_m):
            return True
    return False
