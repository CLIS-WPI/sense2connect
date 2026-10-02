"""Build the urban scene and step it.

Comm paths are solved on a copy that has no sensing targets. Those targets
are perfect absorbers, so a comm solve in the sensing scene would replace
model B with a binary on/off loss. Model B is applied to the geometric
comm paths afterwards.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from sim.comm.blockage import (
    available_power,
    dominant_blocker,
    linear_amplitude_gain,
    path_blocker_loss,
    power_ratio_loss_db,
    sum_blockage_db,
)
from sim.comm.events import annotate_los_events, apply_hysteresis, blocker_oru_distances
from sim.scenes.config import subcarrier_spacing_hz
from sim.scenes.motion import move_radios, move_targets, read_vec3, states_at, track_identity
from sim.scenes.traffic import prepare_scenario

_NONE = 0  # sionna.rt.constants.InteractionType.NONE
_INVALID_OBJECT = 2**32 - 1


def _scene_by_name(name: str) -> Any:
    import sionna.rt as rt

    try:
        return getattr(rt.scene, name)
    except AttributeError as exc:
        raise ValueError(f"unknown built-in scene {name}") from exc


def _make_array(spec: dict[str, Any]) -> Any:
    """Planar array. Spacing is in wavelengths. Omitted spacing keeps the solver default."""
    from sionna.rt import PlanarArray

    kwargs: dict[str, Any] = {
        "num_rows": int(spec["num_rows"]),
        "num_cols": int(spec["num_cols"]),
        "pattern": str(spec["pattern"]),
        "polarization": str(spec["polarization"]),
    }
    if "vertical_spacing" in spec:
        kwargs["vertical_spacing"] = float(spec["vertical_spacing"])
    if "horizontal_spacing" in spec:
        kwargs["horizontal_spacing"] = float(spec["horizontal_spacing"])
    return PlanarArray(**kwargs)


def _add_radios(scene: Any, scenario: dict[str, Any], include_targets: bool) -> tuple[list[Any], list[Any]]:
    from sionna.rt import Receiver, Transmitter
    from sionna.rt.rcs import TR38901SensingTarget

    array_cfg = scenario["array"]
    if "oru" in array_cfg:
        tx_array = _make_array(array_cfg["oru"])
        rx_array = _make_array(array_cfg["ue"])
    else:
        tx_array = _make_array(array_cfg)
        rx_array = tx_array
    scene.frequency = float(scenario["carrier_hz"])
    scene.tx_array = tx_array
    scene.rx_array = rx_array

    states0 = states_at(scenario, 0.0)
    transmitters = []
    receivers = []
    for oru in scenario["orus"]:
        position = [float(v) for v in oru["position_m"]]
        transmitters.append(Transmitter(oru["name"], position=position, velocity=(0.0, 0.0, 0.0)))
    for ue in scenario["ues"]:
        state = states0[ue["name"]]
        receivers.append(
            Receiver(
                ue["name"],
                position=state["position_m"].tolist(),
                velocity=state["velocity_mps"].tolist(),
            )
        )

    targets = []
    if include_targets:
        for spec in list(scenario["vehicles"]) + list(scenario["pedestrians"]):
            state = states0[spec["name"]]
            targets.append(
                TR38901SensingTarget(
                    spec["name"],
                    object_type=str(spec["object_type"]),
                    length=float(spec["length_m"]),
                    width=float(spec["width_m"]),
                    height=float(spec["height_m"]),
                    position=state["position_m"].tolist(),
                    velocity=state["velocity_mps"].tolist(),
                    random_sigma_s=False,
                    random_phases=False,
                    random_xpr=False,
                )
            )
    batch = transmitters + receivers + targets
    scene.add(batch)
    return targets, receivers


def build_scenes(scenario: dict[str, Any]) -> tuple[Any, Any, list[Any]]:
    """Return ``(comm_scene, sensing_scene, sensing_targets)``."""
    from sionna.rt import load_scene

    filename = _scene_by_name(str(scenario["scene"]))
    comm = load_scene(filename)
    sensing = load_scene(filename)
    _add_radios(comm, scenario, include_targets=False)
    targets, _ = _add_radios(sensing, scenario, include_targets=True)
    return comm, sensing, targets


def _solver_kwargs(scenario: dict[str, Any], kind: str) -> dict[str, Any]:
    solvers = scenario["solvers"]
    synthetic = bool(scenario.get("array", {}).get("synthetic_array", True))
    if kind == "rcs":
        return {
            "max_depth": int(solvers["max_depth"]),
            "specular_reflection": bool(solvers["specular_reflection"]),
            "refraction": bool(solvers["refraction"]),
            "seed": int(scenario["seed"]),
            "samples_per_sp": int(solvers["samples_per_sp"]),
            "buffer_size_per_sp": int(solvers["buffer_size_per_sp"]),
            "los": False,
            "synthetic_array": synthetic,
        }
    return {
        "max_depth": int(solvers["max_depth"]),
        "specular_reflection": bool(solvers["specular_reflection"]),
        "diffraction": bool(solvers["diffraction"]),
        "refraction": bool(solvers["refraction"]),
        "seed": int(scenario["seed"]),
        "los": True,
        "samples_per_src": int(solvers["samples_per_src"]),
        "max_num_paths_per_src": int(solvers["max_num_paths_per_src"]),
        "diffuse_reflection": False,
        "edge_diffraction": False,
        "synthetic_array": synthetic,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _canonicalize_paths(packed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort paths of each link so a solver permutation does not change the file.

    Path slots are ordered independently for every receiver and transmitter.
    The key is that link's delay, geometry, and coefficient bytes.
    """
    coefficients = np.array(packed["a"], copy=True)
    delays = np.array(packed["tau"], copy=True)
    doppler = np.array(packed["doppler"], copy=True)
    vertices = np.array(packed["vertices"], copy=True)
    interactions = np.array(packed["interactions"], copy=True)
    objects = np.array(packed["objects"], copy=True)
    valid = np.array(packed["valid"], copy=True)
    n_rx = int(coefficients.shape[0])
    n_tx = int(coefficients.shape[2])
    n_paths = int(coefficients.shape[-2])
    for rx_index in range(n_rx):
        for tx_index in range(n_tx):
            keys: list[bytes] = []
            for path_index in range(n_paths):
                pieces = [
                    np.ascontiguousarray(delays[rx_index, tx_index, path_index]),
                    np.ascontiguousarray(vertices[:, rx_index, tx_index, path_index, :]),
                    np.ascontiguousarray(interactions[:, rx_index, tx_index, path_index]),
                    np.ascontiguousarray(np.bool_(valid[rx_index, tx_index, path_index])),
                    np.ascontiguousarray(doppler[rx_index, tx_index, path_index]),
                    np.ascontiguousarray(coefficients[rx_index, :, tx_index, :, path_index, :]),
                ]
                keys.append(b"".join(piece.tobytes() for piece in pieces))
            order = np.array(
                sorted(range(n_paths), key=lambda index: keys[index]),
                dtype=np.int64,
            )
            coefficients[rx_index, :, tx_index, :, :, :] = np.take(
                coefficients[rx_index, :, tx_index, :, :, :], order, axis=-2
            )
            delays[rx_index, tx_index, :] = delays[rx_index, tx_index, order]
            doppler[rx_index, tx_index, :] = doppler[rx_index, tx_index, order]
            vertices[:, rx_index, tx_index, :, :] = np.take(
                vertices[:, rx_index, tx_index, :, :], order, axis=-2
            )
            interactions[:, rx_index, tx_index, :] = interactions[:, rx_index, tx_index, order]
            objects[:, rx_index, tx_index, :] = objects[:, rx_index, tx_index, order]
            valid[rx_index, tx_index, :] = valid[rx_index, tx_index, order]
    return {
        "a": coefficients,
        "tau": delays,
        "doppler": doppler,
        "vertices": vertices,
        "interactions": interactions,
        "objects": objects,
        "valid": valid,
    }


def pack_paths(paths: Any) -> dict[str, np.ndarray]:
    """Coefficients, delays [s], Doppler [Hz], and geometry of one solve."""
    coefficients, delays = paths.cir(out_type="numpy", normalize_delays=False)
    packed = {
        "a": np.asarray(coefficients),
        "tau": np.asarray(delays),
        "doppler": _to_numpy(paths.doppler),
        "vertices": _to_numpy(paths.vertices),
        "interactions": _to_numpy(paths.interactions),
        "objects": _to_numpy(paths.objects),
        "valid": _to_numpy(paths.valid),
    }
    return _canonicalize_paths(packed)


def _device_positions(scene: Any, names: list[str]) -> dict[str, np.ndarray]:
    return {name: read_vec3(scene.get(name).position) for name in names}


def _object_catalog(scene: Any) -> dict[int, dict[str, str]]:
    """Map ``Paths.objects`` ids to a shape name and a load-stable token.

    Mitsuba reassigns integer object ids on each scene load. The token is
    the scene-object name when that name is stable (``floor``, ``building_*``),
    otherwise a bounding-box key. Path classes use the token. The path id
    itself keeps the integer from ``Paths.objects``.
    """
    catalog: dict[int, dict[str, str]] = {}
    for name, obj in scene.objects.items():
        box = obj.mi_mesh.bbox()
        if name == "floor" or str(name).startswith("building"):
            token = str(name)
        else:
            token = f"mesh-{box.min.x:.2f}-{box.min.y:.2f}-{box.max.z:.2f}"
        catalog[int(obj.object_id)] = {"name": str(name), "token": token}
    return catalog


def _path_class(tokens: list[str]) -> str:
    """LoS, ground reflection, wall reflection, or double bounce."""
    if not tokens:
        return "los"
    if len(tokens) >= 2:
        return "double"
    if tokens[0] == "floor":
        return "ground"
    return "wall"


def _path_segments(
    vertices: np.ndarray,
    interactions: np.ndarray,
    objects: np.ndarray,
    valid: np.ndarray,
    path_index: int,
    rx_index: int,
    tx_index: int,
    tx_m: np.ndarray,
    rx_m: np.ndarray,
    catalog: dict[int, dict[str, str]],
) -> tuple[str, str, str, list[tuple[np.ndarray, np.ndarray]]] | None:
    """Return ``(path_id, path_key, path_class, segments)`` or ``None``.

    ``path_id`` is the interaction-type sequence plus the ``Paths.objects``
    id sequence. Delay is not part of it. ``path_key`` uses the load-stable
    shape token so two scene loads can be compared. LoS is one segment.
    Every hop of a reflected path is included.
    """
    if not bool(valid[rx_index, tx_index, path_index]):
        return None
    kinds = interactions[:, rx_index, tx_index, path_index]
    object_ids = objects[:, rx_index, tx_index, path_index]
    points = [np.asarray(tx_m, dtype=np.float64)]
    bounce_kinds: list[int] = []
    bounce_ids: list[int] = []
    tokens: list[str] = []
    if np.all(kinds == _NONE):
        points.append(np.asarray(rx_m, dtype=np.float64))
        return "los", "los", "los", [(points[0], points[1])]
    for depth, kind in enumerate(kinds):
        if int(kind) == _NONE:
            continue
        vertex = np.round(vertices[depth, rx_index, tx_index, path_index].astype(np.float64), 3)
        points.append(vertex)
        bounce_kinds.append(int(kind))
        object_id = int(object_ids[depth])
        bounce_ids.append(object_id)
        if object_id == _INVALID_OBJECT or object_id not in catalog:
            tokens.append(f"id{object_id}")
        else:
            tokens.append(catalog[object_id]["token"])
    points.append(np.asarray(rx_m, dtype=np.float64))
    kind_text = ".".join(str(kind) for kind in bounce_kinds)
    id_text = ".".join(str(object_id) for object_id in bounce_ids)
    token_text = ".".join(tokens)
    segments = [(points[index], points[index + 1]) for index in range(len(points) - 1)]
    return (
        f"nlos-{kind_text}-{id_text}",
        f"nlos-{kind_text}-{token_text}",
        _path_class(tokens),
        segments,
    )


def _path_power(coefficients: np.ndarray, rx_index: int, tx_index: int, path_index: int) -> float:
    """Unblocked received power of one path, summed over antennas [linear]."""
    coeff = coefficients[rx_index, :, tx_index, :, path_index, :]
    return float(np.sum(np.abs(coeff) ** 2))


def blockage_for_snapshot(
    scenario: dict[str, Any],
    comm_paths: dict[str, np.ndarray],
    tx_positions: dict[str, np.ndarray],
    rx_positions: dict[str, np.ndarray],
    target_states: dict[str, dict[str, np.ndarray]],
    target_sizes: dict[str, tuple[float, float, float, str]],
    wavelength_m: float,
    catalog: dict[int, dict[str, str]],
    *,
    store_segments: bool = True,
    identities: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply model B to every segment of every comm path.

    Returns the attenuated coefficients, one record per (UE, O-RU, path,
    blocker), and one UE-metric row per UE. The metric row uses the
    unblocked powers and the model-B losses.
    """
    oru_names = [oru["name"] for oru in scenario["orus"]]
    ue_names = [ue["name"] for ue in scenario["ues"]]
    coefficients = np.array(comm_paths["a"], copy=True)
    interactions = comm_paths["interactions"]
    vertices = comm_paths["vertices"]
    objects = comm_paths["objects"]
    valid = comm_paths["valid"]
    n_rx, _n_rx_ant, n_tx, _n_tx_ant, n_paths, _n_time = coefficients.shape
    if interactions.ndim != 4:
        raise RuntimeError(
            "expected synthetic-array interactions "
            "[depth, rx, tx, paths]; got shape "
            f"{interactions.shape}"
        )
    records: list[dict[str, Any]] = []
    paths_by_link: dict[tuple[str, str], list[dict[str, Any]]] = {}
    threshold_db = float(scenario["blockage"]["event_loss_db"])

    for rx_index in range(n_rx):
        for tx_index in range(n_tx):
            ue_name = ue_names[rx_index]
            oru_name = oru_names[tx_index]
            for path_index in range(n_paths):
                parsed = _path_segments(
                    vertices,
                    interactions,
                    objects,
                    valid,
                    path_index,
                    rx_index,
                    tx_index,
                    tx_positions[oru_name],
                    rx_positions[ue_name],
                    catalog,
                )
                if parsed is None:
                    continue
                path_id, path_key, path_class, segments = parsed
                power = _path_power(coefficients, rx_index, tx_index, path_index)
                per_blocker: list[float] = []
                blocker_rows: list[dict[str, Any]] = []
                for blocker_name, state in target_states.items():
                    length, width, height, kind = target_sizes[blocker_name]
                    loss_db, segment_losses = path_blocker_loss(
                        segments,
                        state["position_m"],
                        length,
                        width,
                        height,
                        wavelength_m,
                    )
                    per_blocker.append(loss_db)
                    blocked = (not math.isfinite(loss_db)) or loss_db >= threshold_db
                    blocker_id = blocker_name if identities is None else identities.get(blocker_name, blocker_name)
                    row = {
                        "ue": ue_name,
                        "oru": oru_name,
                        "path": path_id,
                        "path_key": path_key,
                        "path_class": path_class,
                        "power": power,
                        "blocker": blocker_id,
                        "blocker_kind": kind,
                        "loss_db": loss_db,
                        "intersects": any(item.intersects for item in segment_losses),
                        "blocked": blocked,
                    }
                    if store_segments:
                        row["segments"] = [
                            {
                                "index": index,
                                "loss_db": item.loss_db,
                                "intersects": bool(item.intersects),
                            }
                            for index, item in enumerate(segment_losses)
                        ]
                    records.append(row)
                    blocker_rows.append(row)
                path_loss = sum_blockage_db(per_blocker)
                paths_by_link.setdefault((ue_name, oru_name), []).append(
                    {
                        "path": path_id,
                        "path_key": path_key,
                        "path_class": path_class,
                        "power": power,
                        "loss_db": path_loss,
                        "blockers": blocker_rows,
                    }
                )
                gain = linear_amplitude_gain(path_loss)
                coefficients[rx_index, :, tx_index, :, path_index, :] *= gain
    for path_rows in paths_by_link.values():
        total_power = sum(float(item["power"]) for item in path_rows)
        for item in path_rows:
            item_share = 0.0 if total_power <= 0.0 else float(item["power"]) / total_power
            for blocker in item["blockers"]:
                blocker["power_share"] = item_share
    ue_metrics = [
        _ue_metric_row(ue_name, oru_name, path_rows) for (ue_name, oru_name), path_rows in paths_by_link.items()
    ]
    return coefficients, records, ue_metrics


def _ue_metric_row(ue_name: str, oru_name: str, path_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """LoS loss, strongest-path loss, and total power loss for one UE."""
    powers = [float(item["power"]) for item in path_rows]
    losses = [float(item["loss_db"]) for item in path_rows]
    blocker_ids = sorted({row["blocker"] for item in path_rows for row in item["blockers"]})
    blockers = []
    for blocker_id in blocker_ids:
        kind = next(
            row["blocker_kind"]
            for item in path_rows
            for row in item["blockers"]
            if row["blocker"] == blocker_id
        )
        aligned = []
        for item in path_rows:
            match = next(row for row in item["blockers"] if row["blocker"] == blocker_id)
            aligned.append(float(match["loss_db"]))
        blockers.append({"id": blocker_id, "kind": kind, "losses_db": aligned})
    power_blocker, power_kind = dominant_blocker(powers, blockers) if blockers else (None, None)

    los_rows = [item for item in path_rows if item["path_class"] == "los"]
    if los_rows:
        los = max(los_rows, key=lambda item: float(item["power"]))
        los_loss = float(los["loss_db"])
        los_blocker, los_kind = _strongest_screen(los)
        los_present = True
        los_power = float(los["power"])
    else:
        los_loss = None
        los_blocker, los_kind = None, None
        los_present = False
        los_power = 0.0
    alternatives = [item for item in path_rows if item["path_class"] != "los"]
    if alternatives:
        best_alt = max(alternatives, key=lambda item: float(item["power"]))
        alt_power = float(best_alt["power"])
        alt_available = available_power(alt_power, float(best_alt["loss_db"]))
        alt_same = False
        if los_blocker is not None:
            match = next((row for row in best_alt["blockers"] if row["blocker"] == los_blocker), None)
            alt_same = match is not None and bool(match["blocked"])
    else:
        alt_power = None
        alt_available = None
        alt_same = False
    unblocked_power = float(sum(float(item["power"]) for item in path_rows))
    blocked_power = float(
        sum(available_power(float(item["power"]), float(item["loss_db"])) for item in path_rows)
    )
    nlos_left = [
        available_power(float(item["power"]), float(item["loss_db"]))
        for item in path_rows
        if item["path_class"] != "los"
    ]
    best_available_alt_power = None if not nlos_left else float(max(nlos_left))

    if path_rows:
        strongest = max(path_rows, key=lambda item: (float(item["power"]), item["path"]))
        strongest_loss = float(strongest["loss_db"])
        strongest_blocker, strongest_kind = _strongest_screen(strongest)
        strongest_path = str(strongest["path"])
        strongest_class = str(strongest["path_class"])
    else:
        strongest_loss = None
        strongest_blocker, strongest_kind = None, None
        strongest_path = None
        strongest_class = None

    return {
        "ue": ue_name,
        "oru": oru_name,
        "los_present": los_present,
        "los_power": los_power,
        "alt_power": alt_power,
        "alt_available_power": alt_available,
        "best_available_alt_power": best_available_alt_power,
        "unblocked_power": unblocked_power,
        "blocked_power": blocked_power,
        "alt_blocked_by_los_blocker": los_blocker if alt_same else None,
        "los_loss_db": los_loss,
        "los_blocker_id": los_blocker,
        "los_blocker_kind": los_kind,
        "strongest_loss_db": strongest_loss,
        "strongest_path": strongest_path,
        "strongest_path_class": strongest_class,
        "strongest_blocker_id": strongest_blocker,
        "strongest_blocker_kind": strongest_kind,
        "power_loss_db": power_ratio_loss_db(powers, losses) if path_rows else 0.0,
        "power_blocker_id": power_blocker,
        "power_blocker_kind": power_kind,
        "path_ids": sorted({str(item["path"]) for item in path_rows}),
    }


def _strongest_screen(path_row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Blocker with the largest model-B loss on this path. Ties break on id."""
    best_id = None
    best_kind = None
    best_loss = -1.0
    for row in path_row["blockers"]:
        loss = float(row["loss_db"])
        if not math.isfinite(loss):
            loss = math.inf
        blocker_id = str(row["blocker"])
        if best_id is None or loss > best_loss or (loss == best_loss and blocker_id < best_id):
            best_loss = loss
            best_id = blocker_id
            best_kind = str(row["blocker_kind"])
    return best_id, best_kind


def _events(snapshot_rows: list[dict[str, Any]], times_s: list[float]) -> list[dict[str, Any]]:
    """Contiguous blockage intervals with start, end, and blocker id."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in snapshot_rows:
        if not row["blocked"]:
            continue
        key = (row["ue"], row["oru"], row["path"], row["blocker"])
        grouped.setdefault(key, []).append(row)

    events = []
    for (ue, oru, path, blocker), blocked_rows in grouped.items():
        blocked_rows = sorted(blocked_rows, key=lambda item: int(item["snapshot"]))
        unique: list[dict[str, Any]] = []
        seen: set[int] = set()
        for row in blocked_rows:
            snap = int(row["snapshot"])
            if snap in seen:
                continue
            seen.add(snap)
            unique.append(row)
        start_row = 0
        for index in range(1, len(unique) + 1):
            if index < len(unique) and int(unique[index]["snapshot"]) == int(unique[index - 1]["snapshot"]) + 1:
                continue
            window = unique[start_row:index]
            finite = [float(row["loss_db"]) for row in window if math.isfinite(float(row["loss_db"]))]
            sample = window[0]
            shares = [float(row["power_share"]) for row in window]
            events.append(
                {
                    "ue": ue,
                    "oru": oru,
                    "path": path,
                    "path_key": sample["path_key"],
                    "path_class": sample["path_class"],
                    "blocker_id": blocker,
                    "blocker_kind": sample["blocker_kind"],
                    "start_s": times_s[int(window[0]["snapshot"])],
                    "end_s": times_s[int(window[-1]["snapshot"])],
                    "max_loss_db": None if not finite else max(finite),
                    "power_weight": float(np.mean(shares)) if shares else 0.0,
                }
            )
            start_row = index
    events.sort(key=lambda item: (item["ue"], item["oru"], item["path"], item["blocker_id"], item["start_s"]))
    return events


_UE_METRICS = (
    ("los", "los_loss_db", "los_blocker_id", "los_blocker_kind"),
    ("strongest", "strongest_loss_db", "strongest_blocker_id", "strongest_blocker_kind"),
    ("power", "power_loss_db", "power_blocker_id", "power_blocker_kind"),
)


def _ue_events(
    metric_rows: list[dict[str, Any]],
    times_s: list[float],
    thresholds_db: list[float],
) -> list[dict[str, Any]]:
    """Contiguous UE-level intervals where a metric stays at or above a threshold."""
    events = []
    links = sorted({(row["ue"], row.get("oru", "")) for row in metric_rows})
    for metric, loss_key, blocker_key, kind_key in _UE_METRICS:
        for threshold in thresholds_db:
            for ue, oru in links:
                rows = [row for row in metric_rows if row["ue"] == ue and row.get("oru", "") == oru]
                rows.sort(key=lambda item: int(item["snapshot"]))
                active = [
                    int(row["snapshot"])
                    for row in rows
                    if row[loss_key] is not None and float(row[loss_key]) >= threshold
                ]
                if not active:
                    continue
                start = previous = active[0]
                for index in active[1:] + [None]:
                    if index is not None and index == previous + 1:
                        previous = index
                        continue
                    window = [
                        row
                        for row in rows
                        if start <= int(row["snapshot"]) <= previous and float(row[loss_key]) >= threshold
                    ]
                    peak = max(window, key=lambda item: float(item[loss_key]))
                    finite = [float(row[loss_key]) for row in window if math.isfinite(float(row[loss_key]))]
                    events.append(
                        {
                            "ue": ue,
                            "oru": oru,
                            "metric": metric,
                            "threshold_db": float(threshold),
                            "blocker_id": peak[blocker_key],
                            "blocker_kind": peak[kind_key],
                            "start_snapshot": start,
                            "end_snapshot": previous,
                            "start_s": times_s[start],
                            "end_s": times_s[previous],
                            "max_loss_db": None if not finite else max(finite),
                        }
                    )
                    if index is None:
                        break
                    start = previous = index
    events.sort(key=lambda item: (item["metric"], item["threshold_db"], item["ue"], item.get("oru", ""), item["start_s"]))
    return events


def _path_id_stability(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often a path id seen at one snapshot is still present at the next.

    The id is interaction types plus ``Paths.objects`` integers. Those
    integers are constant for a shape inside one scene load. A value near
    1 means the same ids persist as the radios move. Delay is not in the id.
    """
    by_ue: dict[tuple[str, str], list[set[str]]] = {}
    for row in metric_rows:
        by_ue.setdefault((row["ue"], row.get("oru", "")), []).append(set(row["path_ids"]))
    persisted = 0
    considered = 0
    distinct: set[str] = set()
    for snaps in by_ue.values():
        for ids in snaps:
            distinct.update(ids)
        for left, right in zip(snaps, snaps[1:]):
            if not left:
                continue
            considered += len(left)
            persisted += len(left & right)
    return {
        "mean_persistence": None if considered == 0 else persisted / considered,
        "n_id_occurrences": considered,
        "n_distinct_ids": len(distinct),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _render_frame(scene: Any, scenario: dict[str, Any], filename: Path, paths: Any = None) -> None:
    """Save one frame. ``paths`` draws the UE links when the solver returns them."""
    from sionna.rt import Camera

    render = scenario["render"]
    camera = Camera(
        position=render["camera_position_m"],
        look_at=render["camera_look_at_m"],
    )
    resolution = render["resolution"]
    kwargs = {
        "camera": camera,
        "filename": str(filename),
        "resolution": (int(resolution[0]), int(resolution[1])),
        "num_samples": int(render["num_samples"]),
        "show_devices": True,
        "show_orientations": False,
    }
    if paths is not None:
        kwargs["paths"] = paths
    scene.render_to_file(**kwargs)


def _write_gif(frame_paths: list[Path], gif_path: Path) -> None:
    from PIL import Image

    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    for frame in frames:
        frame.close()


def write_offset_plot(path: Path, carrier_hz: float) -> None:
    """Save loss [dB] against the lateral offset [m] of a vehicle screen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sim.comm.blockage import lateral_offset_sweep

    wavelength = 299_792_458.0 / carrier_hz
    offsets = np.linspace(-8.0, 8.0, 401)
    losses = lateral_offset_sweep(offsets, wavelength)
    finite = np.isfinite(losses)
    figure, axis = plt.subplots(figsize=(6.5, 3.5))
    axis.plot(offsets[finite], losses[finite], color="black", linewidth=1.5)
    axis.set_xlabel("Lateral offset of blocker center [m]")
    axis.set_ylabel("Model B loss [dB]")
    axis.set_title("Blockage model B versus lateral offset")
    axis.grid(True, linewidth=0.4, alpha=0.6)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _target_catalog(scenario: dict[str, Any]) -> dict[str, tuple[float, float, float, str]]:
    catalog = {}
    for spec in list(scenario["vehicles"]) + list(scenario["pedestrians"]):
        catalog[spec["name"]] = (
            float(spec["length_m"]),
            float(spec["width_m"]),
            float(spec["height_m"]),
            str(spec["kind"]),
        )
    return catalog


def run_scenario(
    scenario: dict[str, Any],
    output_dir: Path,
    render: bool | None = None,
    *,
    comm_only: bool = False,
    write_outputs: bool = True,
    keep_snapshot_rows: bool = True,
) -> dict[str, Any]:
    """Step the scenario and, by default, write channels, ground truth, and frames.

    ``comm_only`` skips the sensing-scene solves. Model B still uses the
    analytic target poses. ``PathSolver`` and ``RCSSolver`` are constructed
    with ``deterministic=True``.
    """
    from sionna.rt.rcs import RCSSolver
    from sionna.rt import PathSolver

    if "oru_height_m" not in scenario:
        scenario = prepare_scenario(scenario)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    do_render = (not comm_only) and (scenario["render"]["enabled"] if render is None else render)
    if comm_only:
        from sionna.rt import load_scene

        comm = load_scene(_scene_by_name(str(scenario["scene"])))
        _add_radios(comm, scenario, include_targets=False)
        sensing = None
        targets: list[Any] = []
    else:
        comm, sensing, targets = build_scenes(scenario)
    target_sizes = _target_catalog(scenario)
    catalog = _object_catalog(comm)
    oru_names = [oru["name"] for oru in scenario["orus"]]
    ue_names = [ue["name"] for ue in scenario["ues"]]
    radios = [comm.get(name) for name in oru_names + ue_names]
    sensing_radios = [] if sensing is None else [sensing.get(name) for name in oru_names + ue_names]

    rcs_solver = RCSSolver(deterministic=True)
    background_solver = PathSolver(deterministic=True)
    comm_solver = PathSolver(deterministic=True)
    wavelength_m = float(np.asarray(comm.wavelength.numpy()).reshape(-1)[0])

    dt_s = float(scenario["dt_s"])
    n_snapshots = int(scenario["n_snapshots"])
    clock = time.perf_counter()
    snapshot_s: list[float] = []
    packed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    ue_rows: list[dict[str, Any]] = []
    times_s: list[float] = []
    frame_paths: list[Path] = []
    frame_dir = output_dir / "frames"

    for index in range(n_snapshots):
        snap_clock = time.perf_counter()
        t_s = index * dt_s
        times_s.append(t_s)
        states = states_at(scenario, t_s)
        move_radios(radios, states)
        if sensing is not None:
            move_targets(sensing, targets, states)
            move_radios(sensing_radios, states)
            sensing_paths = pack_paths(rcs_solver(sensing, **_solver_kwargs(scenario, "rcs")))
            background_paths = pack_paths(
                background_solver(sensing, **_solver_kwargs(scenario, "background"))
            )
            readback = {
                name: {
                    "position_m": read_vec3(sensing.get(name).position),
                    "velocity_mps": read_vec3(sensing.get(name).velocity),
                }
                for name in states
            }
        else:
            sensing_paths = None
            background_paths = None
            readback = states
        comm_solved = comm_solver(comm, **_solver_kwargs(scenario, "comm"))
        comm_geometric = pack_paths(comm_solved)
        attenuated, records, ue_metrics = blockage_for_snapshot(
            scenario,
            comm_geometric,
            _device_positions(comm, oru_names),
            _device_positions(comm, ue_names),
            {name: states[name] for name in target_sizes},
            target_sizes,
            wavelength_m,
            catalog,
            store_segments=keep_snapshot_rows,
            identities={
                spec["name"]: track_identity(scenario, spec, t_s)
                for spec in list(scenario["vehicles"]) + list(scenario["pedestrians"])
            },
        )
        for record in records:
            record["snapshot"] = index
            record["t_s"] = t_s
            rows.append(record)
        for metric in ue_metrics:
            metric["snapshot"] = index
            metric["t_s"] = t_s
            ue_rows.append(metric)
        if write_outputs:
            packed.append(
                {
                    "t_s": t_s,
                    "sensing": sensing_paths,
                    "background": background_paths,
                    "comm_geometric": comm_geometric,
                    "comm_model_b": attenuated,
                    "states": readback,
                }
            )
        if do_render and index % int(scenario["render"]["every"]) == 0:
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / f"frame_{index:03d}.png"
            _render_frame(sensing, scenario, frame_path, paths=comm_solved)
            frame_paths.append(frame_path)
        snapshot_s.append(time.perf_counter() - snap_clock)

    events = _events(rows, times_s)
    thresholds = [float(value) for value in scenario["blockage"]["ue_thresholds_db"]]
    min_gap_s = float(scenario["blockage"].get("min_gap_s", 0.0))
    raw_ue_events = _ue_events(ue_rows, times_s, thresholds)
    ue_events = apply_hysteresis(raw_ue_events, min_gap_s, dt_s)
    range_m = float(scenario["blockage"].get("sensing_range_m", 40.0))
    blocker_ids = sorted({str(event["blocker_id"]) for event in ue_events if event.get("blocker_id")})
    if blocker_ids:
        distances = blocker_oru_distances(scenario, blocker_ids, n_snapshots, dt_s)
        annotate_los_events(ue_events, ue_rows, distances, dt_s, range_m)
    path_id_stability = _path_id_stability(ue_rows)
    elapsed_s = time.perf_counter() - clock
    steady = snapshot_s[1:] if len(snapshot_s) > 1 else snapshot_s
    ground_truth = {
        "seed": int(scenario["seed"]),
        "dt_s": dt_s,
        "n_snapshots": n_snapshots,
        "carrier_hz": float(scenario["carrier_hz"]),
        "numerology": int(scenario["numerology"]),
        "subcarrier_spacing_hz": subcarrier_spacing_hz(int(scenario["numerology"])),
        "n_subcarriers": int(scenario["n_subcarriers"]),
        "wavelength_m": wavelength_m,
        "mount": scenario.get("mount"),
        "oru_height_m": scenario.get("oru_height_m"),
        "density": scenario.get("density"),
        "min_gap_s": min_gap_s,
        "sensing_range_m": range_m,
        "runtime_s": elapsed_s,
        "seconds_per_snapshot": elapsed_s / n_snapshots,
        "seconds_per_snapshot_steady": (sum(steady) / len(steady)) if steady else elapsed_s / n_snapshots,
        "design": (
            "Comm paths are solved without sensing targets. "
            "Model B is applied to every segment of every comm path, LoS and reflected. "
            "The sensing scene keeps the targets, so PathSolver blockage there stays binary. "
            "A path id is the interaction-type sequence plus the Paths.objects id sequence. "
            "A blocker that wraps the periodic street starts a new ground-truth identity. "
            "UE metrics use the unblocked path powers and the model-B losses."
        ),
        "events": events,
        "ue_events": ue_events,
        "ue_events_raw": raw_ue_events,
        "ue_metrics": ue_rows if keep_snapshot_rows else [],
        "ue_trace": [
            {
                "ue": row["ue"],
                "oru": row["oru"],
                "snapshot": int(row["snapshot"]),
                "los_loss_db": row["los_loss_db"],
                "strongest_loss_db": row["strongest_loss_db"],
                "power_loss_db": row["power_loss_db"],
                "los_blocker_id": row["los_blocker_id"],
                "los_blocker_kind": row["los_blocker_kind"],
                "strongest_blocker_id": row["strongest_blocker_id"],
                "strongest_blocker_kind": row["strongest_blocker_kind"],
                "power_blocker_id": row["power_blocker_id"],
                "power_blocker_kind": row["power_blocker_kind"],
                "los_power": row["los_power"],
                "alt_power": row["alt_power"],
                "alt_available_power": row["alt_available_power"],
                "best_available_alt_power": row["best_available_alt_power"],
                "unblocked_power": row["unblocked_power"],
                "blocked_power": row["blocked_power"],
                "alt_blocked_by_los_blocker": row["alt_blocked_by_los_blocker"],
            }
            for row in ue_rows
        ],
        "path_id_stability": path_id_stability,
        "per_snapshot_blockage": rows if keep_snapshot_rows else [],
        "states": [
            {
                "t_s": item["t_s"],
                "actors": {
                    name: {
                        "position_m": state["position_m"].tolist(),
                        "velocity_mps": state["velocity_mps"].tolist(),
                    }
                    for name, state in item["states"].items()
                },
            }
            for item in packed
        ],
    }
    if write_outputs:
        gt_path = output_dir / "ground_truth.json"
        gt_path.write_text(
            json.dumps(_json_ready(ground_truth), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        channel_blob = []
        for item in packed:
            channel_blob.append(
                {
                    "t_s": item["t_s"],
                    "sensing": item["sensing"],
                    "background": item["background"],
                    "comm_geometric": item["comm_geometric"],
                    "comm_model_b": item["comm_model_b"],
                }
            )
        np.savez_compressed(
            output_dir / "channels.npz",
            snapshots=np.array(channel_blob, dtype=object),
        )
        write_offset_plot(output_dir / "blockage_lateral_offset.png", float(scenario["carrier_hz"]))
        if frame_paths:
            _write_gif(frame_paths, output_dir / "scenario.gif")
    return ground_truth
