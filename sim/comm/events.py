"""UE-event hysteresis, LoS headroom, and sensing-range warning time."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim.comm.blockage import headroom_db
from sim.scenes.motion import states_at, track_identity


def apply_hysteresis(
    events: list[dict[str, Any]],
    min_gap_s: float,
    dt_s: float,
) -> list[dict[str, Any]]:
    """Merge short gaps, then keep higher thresholds nested in the 3 dB event.

    Gap merge joins same-threshold events whose inactive gap is shorter than
    ``min_gap_s``. Nesting then collapses every 10 dB piece that falls inside
    one 3 dB event into a single event, and likewise 20 dB inside 10 dB.
    A dip that stays at or above 3 dB therefore cannot produce more 10 dB
    events than 3 dB events.
    """
    merged = merge_ue_events(events, min_gap_s, dt_s)
    return nest_thresholds(merged)


def nest_thresholds(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse higher-threshold fragments that lie inside one lower-threshold event."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for event in events:
        key = (event["ue"], event.get("oru"), event["metric"])
        groups.setdefault(key, []).append(event)
    nested: list[dict[str, Any]] = []
    for pieces in groups.values():
        by_threshold: dict[float, list[dict[str, Any]]] = {}
        for event in pieces:
            by_threshold.setdefault(float(event["threshold_db"]), []).append(event)
        base = sorted(by_threshold.get(3.0, []), key=lambda item: int(item["start_snapshot"]))
        nested.extend(base)
        middle = _collapse_into(by_threshold.get(10.0, []), base)
        nested.extend(middle)
        nested.extend(_collapse_into(by_threshold.get(20.0, []), middle or base))
        known = {3.0, 10.0, 20.0}
        for threshold, items in by_threshold.items():
            if threshold not in known:
                nested.extend(items)
    nested.sort(key=lambda item: (item["metric"], item["threshold_db"], item["ue"], item.get("oru", ""), item["start_s"]))
    return nested


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> int:
    start = max(int(left["start_snapshot"]), int(right["start_snapshot"]))
    end = min(int(left["end_snapshot"]), int(right["end_snapshot"]))
    return max(0, end - start + 1)


def _collapse_into(children: list[dict[str, Any]], parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not children:
        return []
    if not parents:
        return [dict(child) for child in children]
    buckets: list[list[dict[str, Any]]] = [[] for _ in parents]
    orphans: list[dict[str, Any]] = []
    for child in children:
        best_index = None
        best_overlap = 0
        for index, parent in enumerate(parents):
            overlap = _overlap(child, parent)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is None:
            orphans.append(dict(child))
        else:
            buckets[best_index].append(child)
    collapsed: list[dict[str, Any]] = []
    for kids in buckets:
        if not kids:
            continue
        collapsed.append(_join_events(kids))
    collapsed.extend(orphans)
    collapsed.sort(key=lambda item: int(item["start_snapshot"]))
    return collapsed


def _join_events(pieces: list[dict[str, Any]]) -> dict[str, Any]:
    pieces = sorted(pieces, key=lambda item: int(item["start_snapshot"]))
    joined = dict(pieces[0])
    for nxt in pieces[1:]:
        if _peak(nxt) > _peak(joined) or (
            _peak(nxt) == _peak(joined) and str(nxt.get("blocker_id")) < str(joined.get("blocker_id"))
        ):
            joined["blocker_id"] = nxt.get("blocker_id")
            joined["blocker_kind"] = nxt.get("blocker_kind")
            joined["max_loss_db"] = nxt.get("max_loss_db")
        joined["end_s"] = nxt["end_s"]
        joined["end_snapshot"] = nxt["end_snapshot"]
    return joined


def merge_ue_events(
    events: list[dict[str, Any]],
    min_gap_s: float,
    dt_s: float,
) -> list[dict[str, Any]]:
    """Merge same-link events whose inactive gap is shorter than ``min_gap_s``.

    The gap is the time from the snapshot after one event to the snapshot
    before the next. Merging keeps the earlier start, the later end, and the
    blocker of the piece with the larger peak loss.
    """
    if min_gap_s <= 0.0 or not events:
        return list(events)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for event in events:
        key = (event["ue"], event.get("oru"), event["metric"], float(event["threshold_db"]))
        groups.setdefault(key, []).append(event)
    merged: list[dict[str, Any]] = []
    for pieces in groups.values():
        pieces = sorted(pieces, key=lambda item: int(item["start_snapshot"]))
        current = dict(pieces[0])
        for nxt in pieces[1:]:
            gap_snapshots = int(nxt["start_snapshot"]) - int(current["end_snapshot"]) - 1
            gap_s = gap_snapshots * dt_s
            if gap_s < min_gap_s:
                if _peak(nxt) > _peak(current) or (
                    _peak(nxt) == _peak(current) and str(nxt.get("blocker_id")) < str(current.get("blocker_id"))
                ):
                    current["blocker_id"] = nxt.get("blocker_id")
                    current["blocker_kind"] = nxt.get("blocker_kind")
                    current["max_loss_db"] = nxt.get("max_loss_db")
                current["end_s"] = nxt["end_s"]
                current["end_snapshot"] = nxt["end_snapshot"]
            else:
                merged.append(current)
                current = dict(nxt)
        merged.append(current)
    merged.sort(key=lambda item: (item["metric"], item["threshold_db"], item["ue"], item["start_s"]))
    return merged


def _peak(event: dict[str, Any]) -> float:
    value = event.get("max_loss_db")
    if value is None or not math.isfinite(float(value)):
        return math.inf
    return float(value)


def entry_warning_s(
    distances_m: list[float] | np.ndarray,
    event_index: int,
    dt_s: float,
    range_m: float,
) -> dict[str, Any]:
    """Time from entering ``range_m`` to the event, for one blocker.

    ``distances_m[k]`` is the blocker–O-RU distance [m] at snapshot ``k``.
    The entry is the most recent crossing from outside the range to inside
    it at or before ``event_index``. Samples that are not finite are outside
    this identity's lifetime (before the scenario, or before a wrap created
    this id). The returned duration is an upper bound on time in view, not
    a predictor lead time. It is censored when the identity is already
    inside the range on its first sample, because the entry was not observed.
    """
    distances = np.asarray(distances_m, dtype=np.float64)
    if event_index < 0 or event_index >= len(distances):
        raise IndexError("event_index is outside the distance series")
    if not math.isfinite(float(distances[event_index])) or float(distances[event_index]) > range_m:
        return {"warning_s": None, "in_range": False, "censored": False, "upper_bound": True}
    enter = event_index
    while enter > 0 and math.isfinite(float(distances[enter - 1])) and float(distances[enter - 1]) <= range_m:
        enter -= 1
    if enter == 0:
        censored = float(distances[0]) <= range_m
    else:
        censored = not math.isfinite(float(distances[enter - 1]))
    return {
        "warning_s": (event_index - enter) * dt_s,
        "in_range": True,
        "censored": censored,
        "upper_bound": True,
    }


def blocker_oru_distances(
    scenario: dict[str, Any],
    blocker_ids: list[str],
    n_snapshots: int,
    dt_s: float,
) -> dict[tuple[str, str], np.ndarray]:
    """Distance [m] from each ground-truth identity to each O-RU.

    An identity exists only between street wraps. Samples outside that
    lifetime are non-finite, so a later lap does not inherit the previous
    one's time in range.
    """
    orus = scenario["orus"]
    oru_pos = {str(oru["name"]): np.asarray(oru["position_m"], dtype=np.float64) for oru in orus}
    actors = list(scenario.get("vehicles", [])) + list(scenario.get("pedestrians", []))
    series: dict[tuple[str, str], np.ndarray] = {}
    wanted = set(blocker_ids)
    for index in range(n_snapshots):
        t_s = index * dt_s
        states = states_at(scenario, t_s)
        for actor in actors:
            identity = track_identity(scenario, actor, t_s)
            if wanted and identity not in wanted:
                continue
            position = states[actor["name"]]["position_m"]
            for oru_name, origin in oru_pos.items():
                key = (identity, oru_name)
                if key not in series:
                    series[key] = np.full(n_snapshots, np.inf, dtype=np.float64)
                series[key][index] = float(np.linalg.norm(position - origin))
    return series


def annotate_los_events(
    events: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    distances_m: dict[tuple[str, str], np.ndarray],
    dt_s: float,
    range_m: float,
    threshold_db: float = 10.0,
) -> None:
    """Add headroom and warning fields to 10 dB LoS events, in place.

    Unblocked headroom is the worst ``10 log10(P_alt / P_LoS)`` over
    snapshots in the event that are still at or above the threshold.
    Available headroom uses the same alternative after model B on that
    snapshot, over the unblocked LoS power. ``alt_same_blocker`` is true
    when the event's blocker also blocks that alternative (model-B loss
    at or above the per-path threshold) on one of those snapshots.
    Warning uses ``distances_m[(blocker_id, oru)]`` and is an upper bound.
    """
    by_link: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trace:
        by_link.setdefault((row["ue"], row["oru"]), []).append(row)
    for event in events:
        if event["metric"] != "los" or abs(float(event["threshold_db"]) - threshold_db) > 1e-9:
            continue
        rows = by_link.get((event["ue"], event["oru"]), [])
        headrooms: list[float] = []
        available: list[float] = []
        same = False
        for row in rows:
            snap = int(row["snapshot"])
            if snap < int(event["start_snapshot"]) or snap > int(event["end_snapshot"]):
                continue
            loss = row.get("los_loss_db")
            if loss is None or float(loss) < threshold_db:
                continue
            alt = row.get("alt_power")
            alt_power = None if alt is None else float(alt)
            los_power = float(row.get("los_power") or 0.0)
            headrooms.append(headroom_db(los_power, alt_power))
            if alt_power is not None:
                available_power = row.get("alt_available_power")
                available.append(
                    headroom_db(los_power, None if available_power is None else float(available_power))
                )
            if row.get("alt_blocked_by_los_blocker") == event.get("blocker_id"):
                same = True
        finite = [value for value in headrooms if math.isfinite(value)]
        event["headroom_db"] = None if not finite else float(min(finite))
        event["headroom_available_db"] = None if not available else float(min(available))
        event["alt_same_blocker"] = same
        event["n_snapshots_without_alternative"] = sum(1 for value in headrooms if not math.isfinite(value))
        blocker_id = event.get("blocker_id")
        series = None if blocker_id is None else distances_m.get((str(blocker_id), str(event["oru"])))
        if series is None:
            event["warning_s"] = None
            event["in_range"] = False
            event["censored"] = False
            continue
        warning = entry_warning_s(series, int(event["start_snapshot"]), dt_s, range_m)
        event["warning_s"] = warning["warning_s"]
        event["in_range"] = warning["in_range"]
        event["censored"] = warning["censored"]
