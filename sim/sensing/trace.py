"""One sensing trace: solvers, model B, radar processing, and the cache.

Scene object counts stay constant. Path tensors are padded to ``max_paths``
before the OFDM step so that kernel sees one shape. Solver output sizes are
recorded; a change there is what makes Dr.Jit compile again.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim.comm.blockage import path_blocker_loss
from sim.comm.blockage_torch import blocker_path_loss_db
from sim.scenes.loop import _object_catalog, build_scenes, pack_paths
from sim.scenes.motion import move_radios, move_targets, states_at
from sim.sensing.cache import cache_dir, labels_from_scene, pack_paths as pack_radar_paths
from sim.sensing.cache import save_frame, save_meta, save_static
from sim.sensing.cfar import ca_cfar, local_maxima
from sim.sensing.channel import (
    add_thermal_noise,
    doppler_hz,
    frequency_response,
    localize,
    n_lags,
    power_map,
    radial_velocity_mps,
    range_m,
)
from sim.sensing.cluster import cluster_detections
from sim.sensing.radar import _cube, _radar_kwargs, build_radar, ground_truth
from sim.sensing.track import Tracker
from sim.sensing.waveform import Waveform, waveform_from_config


def compile_delta() -> int:
    """Number of Dr.Jit kernels compiled since the history was last cleared."""
    import drjit as dr

    history = dr.kernel_history()
    misses = sum(1 for entry in history if not int(entry.get("cache_hit", 1)))
    dr.kernel_history_clear()
    return int(misses)


def enable_kernel_history() -> None:
    """Turn on the Dr.Jit kernel log used for the compile count."""
    import drjit as dr

    dr.set_flag(dr.JitFlag.KernelHistory, True)
    dr.kernel_history_clear()


def trace_job(
    scenario: dict[str, Any],
    *,
    mount: str,
    density: str,
    cache_root: Path | None,
    max_frames: int | None = None,
    max_paths: int = 512,
    process: bool = True,
    record_blockage: bool = False,
) -> dict[str, Any]:
    """Trace one ``(seed, mount, density)`` job and optionally write the cache.

    Returns wall-time buckets in seconds and the compile count. Radar
    processing uses one CPI per snapshot at the scenario ``dt_s``.
    """
    enable_kernel_history()
    buckets = {
        "scene_edits": 0.0,
        "rcs": 0.0,
        "path_solver": 0.0,
        "model_b": 0.0,
        "ofdm_fft": 0.0,
        "cfar": 0.0,
        "dbscan": 0.0,
        "tracker": 0.0,
        "io": 0.0,
    }
    compiles = 0
    shapes: list[tuple[Any, ...]] = []
    path_counts: list[int] = []
    object_counts: set[int] = set()
    radar = build_radar(scenario)
    compiles += compile_delta()
    spec = scenario["sensing_radar"]
    waveform = waveform_from_config(scenario)
    noise = spec["noise"]
    n_frames = int(scenario["n_snapshots"] if max_frames is None else min(max_frames, scenario["n_snapshots"]))
    dt_s = float(scenario["dt_s"])
    directory = None
    if cache_root is not None:
        directory = cache_dir(cache_root, mount, density, int(scenario["seed"]))

    started = time.perf_counter()
    static_paths = radar["background"](radar["static"], **_radar_kwargs(scenario, "background"))
    buckets["path_solver"] += time.perf_counter() - started
    compiles += compile_delta()
    actor_positions = {
        actor["name"]: states_at(scenario, 0.0)[actor["name"]]["position_m"]
        for actor in list(scenario["vehicles"]) + list(scenario["pedestrians"])
    }
    static_labels = labels_from_scene(radar["static"], {})
    live_labels = labels_from_scene(radar["live"], actor_positions)
    static_packed = pack_radar_paths(static_paths, static_labels)
    shapes.append(("static",) + tuple(int(v) for v in np.asarray(static_packed["a"]).shape))
    path_counts.append(_path_count(static_packed["a"]))

    comm, _sensing, _targets = build_scenes(scenario)
    comm_solver_kwargs = _comm_kwargs(scenario)
    from sionna.rt import PathSolver

    comm_solver = PathSolver()
    ue_devices = [comm.get(ue["name"]) for ue in scenario["ues"]]
    catalog = _object_catalog(comm)
    wavelength = float(np.asarray(comm.wavelength.numpy()).reshape(-1)[0])
    tx_positions = {oru["name"]: np.asarray(oru["position_m"], dtype=np.float64) for oru in scenario["orus"]}
    comm_packed: dict[str, np.ndarray] | None = None
    target_sizes = {
        actor["name"]: (
            float(actor["length_m"]),
            float(actor["width_m"]),
            float(actor["height_m"]),
            str(actor["kind"]),
        )
        for actor in list(scenario["vehicles"]) + list(scenario["pedestrians"])
    }
    event_rows: list[dict[str, Any]] = []
    tracker = Tracker(
        dt_s=dt_s,
        association_gate_m=6.0,
        coast_frames=4,
        confirm_hits=int(spec["tracker"]["confirm_hits"]),
        radar_position_m=np.asarray(radar["radar_position_m"], dtype=np.float64),
    )
    if directory is not None:
        started = time.perf_counter()
        save_static(directory, static_packed)
        save_meta(
            directory,
            {
                "seed": int(scenario["seed"]),
                "mount": mount,
                "density": density,
                "dt_s": dt_s,
                "n_frames": n_frames,
                "max_paths": max_paths,
                "n_subcarriers": waveform.n_subcarriers,
            },
        )
        buckets["io"] += time.perf_counter() - started

    static_response = frequency_response(static_paths, waveform, float(noise["tx_power_dbm"]))
    for index in range(n_frames):
        t_s = index * dt_s
        states = states_at(scenario, t_s)
        rx_positions = {ue["name"]: states[ue["name"]]["position_m"] for ue in scenario["ues"]}
        started = time.perf_counter()
        move_targets(radar["live"], radar["targets"], states)
        move_radios(ue_devices, states)
        buckets["scene_edits"] += time.perf_counter() - started
        object_counts.add(len(radar["live"].objects))
        started = time.perf_counter()
        target_paths = radar["rcs"](radar["live"], **_radar_kwargs(scenario, "rcs"))
        buckets["rcs"] += time.perf_counter() - started
        compiles += compile_delta()
        started = time.perf_counter()
        live_paths = radar["background"](radar["live"], **_radar_kwargs(scenario, "background"))
        comm_paths = comm_solver(comm, **comm_solver_kwargs)
        buckets["path_solver"] += time.perf_counter() - started
        compiles += compile_delta()
        rcs_packed = pack_radar_paths(target_paths, live_labels)
        bg_packed = pack_radar_paths(live_paths, live_labels)
        comm_packed = pack_paths(comm_paths)
        comm_saved = pack_radar_paths(comm_paths, labels_from_scene(comm, {}))
        shapes.append(("rcs", index, *_shape(rcs_packed["a"])))
        shapes.append(("bg", index, *_shape(bg_packed["a"])))
        path_counts.append(_path_count(rcs_packed["a"]))
        path_counts.append(_path_count(bg_packed["a"]))
        truth = ground_truth(scenario, t_s, radar["radar_position_m"], float(spec["sensing_range_m"]))
        if directory is not None:
            started = time.perf_counter()
            save_frame(directory, index, rcs_packed, bg_packed, truth, comm_saved)
            buckets["io"] += time.perf_counter() - started
        segment_batch = _segment_batch(comm_packed, tx_positions, rx_positions, scenario, catalog)
        if record_blockage:
            event_rows.extend(
                _blockage_rows(
                    scenario,
                    comm_packed,
                    tx_positions,
                    rx_positions,
                    states,
                    target_sizes,
                    wavelength,
                    catalog,
                    index,
                )
            )
        else:
            _time_model_b(buckets, states, target_sizes, wavelength, segment_batch)
        if process:
            _process_frame(
                buckets,
                index,
                scenario,
                waveform,
                noise,
                spec,
                radar,
                target_paths,
                live_paths,
                static_response,
                tracker,
            )
    return {
        "buckets_s": buckets,
        "compiles": compiles,
        "object_counts": sorted(object_counts),
        "n_distinct_shapes": len({tuple(item) for item in shapes}),
        "path_count_min": min(path_counts) if path_counts else 0,
        "path_count_max": max(path_counts) if path_counts else 0,
        "n_frames": n_frames,
        "max_paths_seen": max(path_counts) if path_counts else 0,
        "blockage_rows": event_rows,
    }


def _time_model_b(
    buckets: dict[str, float],
    states: dict[str, dict[str, np.ndarray]],
    target_sizes: dict[str, tuple[float, float, float, str]],
    wavelength_m: float,
    segment_batch: dict[str, np.ndarray] | None,
) -> None:
    """Time the NumPy reference. The GPU batch is timed only when segments exist."""
    if segment_batch is None:
        return
    names = list(target_sizes)
    centers = np.stack([states[name]["position_m"] for name in names])
    length = np.asarray([target_sizes[name][0] for name in names])
    width = np.asarray([target_sizes[name][1] for name in names])
    height = np.asarray([target_sizes[name][2] for name in names])
    starts = segment_batch["starts"]
    ends = segment_batch["ends"]
    valid = segment_batch["valid"]
    started = time.perf_counter()
    for blocker in range(centers.shape[0]):
        for path in range(starts.shape[0]):
            segments = [
                (starts[path, seg], ends[path, seg])
                for seg in range(starts.shape[1])
                if bool(valid[path, seg])
            ]
            path_blocker_loss(
                segments,
                centers[blocker],
                float(length[blocker]),
                float(width[blocker]),
                float(height[blocker]),
                wavelength_m,
            )
    buckets["model_b"] += time.perf_counter() - started


def model_b_torch_seconds(
    states: dict[str, dict[str, np.ndarray]],
    target_sizes: dict[str, tuple[float, float, float, str]],
    wavelength_m: float,
    segment_batch: dict[str, np.ndarray],
) -> float:
    """Wall time [s] of one GPU model-B batch for the same segments."""
    names = list(target_sizes)
    centers = torch.as_tensor(np.stack([states[name]["position_m"] for name in names]), device="cuda")
    length = torch.as_tensor([target_sizes[name][0] for name in names], device="cuda")
    width = torch.as_tensor([target_sizes[name][1] for name in names], device="cuda")
    height = torch.as_tensor([target_sizes[name][2] for name in names], device="cuda")
    starts = torch.as_tensor(segment_batch["starts"], device="cuda")
    ends = torch.as_tensor(segment_batch["ends"], device="cuda")
    valid = torch.as_tensor(segment_batch["valid"], device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    blocker_path_loss_db(starts, ends, centers, length, width, height, wavelength_m, valid)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def _process_frame(
    buckets: dict[str, float],
    index: int,
    scenario: dict[str, Any],
    waveform: Waveform,
    noise: dict[str, Any],
    spec: dict[str, Any],
    radar: dict[str, Any],
    target_paths: object,
    live_paths: object,
    static_response: torch.Tensor,
    tracker: Tracker,
) -> None:
    started = time.perf_counter()
    measured = frequency_response(target_paths, waveform, float(noise["tx_power_dbm"]))
    measured = measured + frequency_response(live_paths, waveform, float(noise["tx_power_dbm"]))
    generator = torch.Generator(device=measured.device)
    generator.manual_seed(int(scenario["seed"]) * 100003 + index + waveform.n_subcarriers)
    measured = add_thermal_noise(
        measured,
        waveform,
        float(noise["noise_figure_db"]),
        float(noise["temperature_k"]),
        generator,
    )
    lags = n_lags(waveform, float(spec["max_range_m"]))
    cube = _cube(measured - static_response, waveform, lags)
    torch.cuda.synchronize()
    buckets["ofdm_fft"] += time.perf_counter() - started
    started = time.perf_counter()
    power = power_map(cube)
    mask, _alpha = ca_cfar(power, guard=int(spec["cfar"]["guard"]), train=4, pfa=1e-4, noise_applied=True)
    hits = []
    for doppler_index, lag_index, peak in local_maxima(mask, power):
        x_m, y_m = localize(
            cube,
            doppler_index,
            lag_index,
            waveform,
            radar["positions_m"],
            radar["orientation_rad"],
            radar["radar_position_m"],
        )
        hits.append(
            {
                "x_m": x_m,
                "y_m": y_m,
                "range_m": range_m(lag_index, waveform),
                "doppler_hz": doppler_hz(doppler_index, waveform),
                "radial_velocity_mps": radial_velocity_mps(doppler_index, waveform),
                "power": peak,
                "snapshot": index,
            }
        )
    buckets["cfar"] += time.perf_counter() - started
    started = time.perf_counter()
    clusters = cluster_detections(hits, eps_m=3.0, min_samples=int(spec["cluster"]["min_samples"]))
    buckets["dbscan"] += time.perf_counter() - started
    started = time.perf_counter()
    tracker.step(clusters)
    buckets["tracker"] += time.perf_counter() - started


def _segment_batch(
    comm_packed: dict[str, np.ndarray],
    tx_positions: dict[str, np.ndarray],
    rx_positions: dict[str, np.ndarray],
    scenario: dict[str, Any],
    catalog: dict[int, dict[str, str]],
) -> dict[str, np.ndarray] | None:
    """Pack comm segments once. The comm scene has no moving meshes."""
    from sim.scenes.loop import _path_segments

    vertices = comm_packed["vertices"]
    interactions = comm_packed["interactions"]
    objects = comm_packed["objects"]
    valid = comm_packed["valid"]
    if interactions.ndim != 4:
        return None
    oru_names = [oru["name"] for oru in scenario["orus"]]
    ue_names = [ue["name"] for ue in scenario["ues"]]
    n_rx, _rx_ant, n_tx, _tx_ant, n_paths, _time = comm_packed["a"].shape
    depth = int(vertices.shape[0])
    max_segments = depth + 1
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    for rx_index in range(n_rx):
        for tx_index in range(n_tx):
            for path_index in range(n_paths):
                parsed = _path_segments(
                    vertices,
                    interactions,
                    objects,
                    valid,
                    path_index,
                    rx_index,
                    tx_index,
                    tx_positions[oru_names[tx_index]],
                    rx_positions[ue_names[rx_index]],
                    catalog,
                )
                start = np.zeros((max_segments, 3), dtype=np.float64)
                end = np.zeros((max_segments, 3), dtype=np.float64)
                mask = np.zeros(max_segments, dtype=bool)
                if parsed is not None:
                    for seg_index, (a_m, b_m) in enumerate(parsed[3]):
                        start[seg_index] = a_m
                        end[seg_index] = b_m
                        mask[seg_index] = True
                starts.append(start)
                ends.append(end)
                valids.append(mask)
    if not starts:
        return None
    return {
        "starts": np.stack(starts),
        "ends": np.stack(ends),
        "valid": np.stack(valids),
    }


def _comm_kwargs(scenario: dict[str, Any]) -> dict[str, Any]:
    from sim.scenes.loop import _solver_kwargs

    kwargs = dict(_solver_kwargs(scenario, "background"))
    kwargs["los"] = True
    return kwargs


def _blockage_rows(
    scenario: dict[str, Any],
    comm_packed: dict[str, np.ndarray],
    tx_positions: dict[str, np.ndarray],
    rx_positions: dict[str, np.ndarray],
    states: dict[str, dict[str, np.ndarray]],
    target_sizes: dict[str, tuple[float, float, float, str]],
    wavelength_m: float,
    catalog: dict[int, dict[str, str]],
    index: int,
) -> list[dict[str, Any]]:
    """UE metric rows for one snapshot, including the LoS blocker identity."""
    from sim.scenes.loop import blockage_for_snapshot
    from sim.scenes.motion import track_identity

    identities = {
        actor["name"]: track_identity(scenario, actor, index * float(scenario["dt_s"]))
        for actor in list(scenario["vehicles"]) + list(scenario["pedestrians"])
    }
    _coefficients, _records, metrics = blockage_for_snapshot(
        scenario,
        comm_packed,
        tx_positions,
        rx_positions,
        {name: states[name] for name in target_sizes},
        target_sizes,
        wavelength_m,
        catalog,
        store_segments=False,
        identities=identities,
    )
    for row in metrics:
        row["snapshot"] = index
    return metrics


def _shape(value: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(value).shape)


def _path_count(coefficients: np.ndarray) -> int:
    array = np.asarray(coefficients)
    if array.ndim >= 5:
        return int(array.shape[-2] if array.shape[-1] <= 4 else array.shape[-1])
    return int(array.shape[-1])


def direct_frame_paths(scenario: dict[str, Any], index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Solve one snapshot without writing a cache. Used for the identity check."""
    radar = build_radar(scenario)
    t_s = index * float(scenario["dt_s"])
    states = states_at(scenario, t_s)
    move_targets(radar["live"], radar["targets"], states)
    target_paths = radar["rcs"](radar["live"], **_radar_kwargs(scenario, "rcs"))
    live_paths = radar["background"](radar["live"], **_radar_kwargs(scenario, "background"))
    positions = {
        actor["name"]: states[actor["name"]]["position_m"]
        for actor in list(scenario["vehicles"]) + list(scenario["pedestrians"])
    }
    labels = labels_from_scene(radar["live"], positions)
    return pack_radar_paths(target_paths, labels), pack_radar_paths(live_paths, labels)
