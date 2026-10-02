"""Build the urban scene and step it.

Comm paths are solved on a copy that has no sensing targets. Those targets
are perfect absorbers, so a comm solve in the sensing scene would replace
model B with a binary on/off loss. Model B is applied to the geometric
comm paths afterwards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.comm.blockage import (
    ScreenLoss,
    linear_amplitude_gain,
    screen_blockage,
    sum_blockage_db,
)
from sim.scenes.config import subcarrier_spacing_hz
from sim.scenes.motion import move_radios, move_targets, read_vec3, states_at

_NONE = 0  # sionna.rt.constants.InteractionType.NONE


def _scene_by_name(name: str) -> Any:
    import sionna.rt as rt

    try:
        return getattr(rt.scene, name)
    except AttributeError as exc:
        raise ValueError(f"unknown built-in scene {name}") from exc


def _add_radios(scene: Any, scenario: dict[str, Any], include_targets: bool) -> tuple[list[Any], list[Any]]:
    from sionna.rt import PlanarArray, Receiver, Transmitter
    from sionna.rt.rcs import TR38901SensingTarget

    array_cfg = scenario["array"]
    array = PlanarArray(
        num_rows=int(array_cfg["num_rows"]),
        num_cols=int(array_cfg["num_cols"]),
        polarization=str(array_cfg["polarization"]),
        pattern=str(array_cfg["pattern"]),
    )
    scene.frequency = float(scenario["carrier_hz"])
    scene.tx_array = array
    scene.rx_array = array

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
    if kind == "rcs":
        return {
            "max_depth": int(solvers["max_depth"]),
            "specular_reflection": bool(solvers["specular_reflection"]),
            "refraction": bool(solvers["refraction"]),
            "seed": int(scenario["seed"]),
            "samples_per_sp": int(solvers["samples_per_sp"]),
            "buffer_size_per_sp": int(solvers["buffer_size_per_sp"]),
            "los": False,
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
            valid[rx_index, tx_index, :] = valid[rx_index, tx_index, order]
    return {
        "a": coefficients,
        "tau": delays,
        "doppler": doppler,
        "vertices": vertices,
        "interactions": interactions,
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
        "valid": _to_numpy(paths.valid),
    }
    return _canonicalize_paths(packed)


def _device_positions(scene: Any, names: list[str]) -> dict[str, np.ndarray]:
    return {name: read_vec3(scene.get(name).position) for name in names}


def _incoming_segment(
    vertices: np.ndarray,
    interactions: np.ndarray,
    valid: np.ndarray,
    delays_s: np.ndarray,
    path_index: int,
    rx_index: int,
    tx_index: int,
    tx_m: np.ndarray,
    rx_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return ``(start, end, path_id)`` for one valid path, or ``None``.

    A reflected segment starts at the last interaction point. That point is
    placed on a 1 mm grid so a micrometre of ray-tracer jitter does not
    change the blocker identity. The LoS segment uses the radio positions.
    """
    if not bool(valid[rx_index, tx_index, path_index]):
        return None
    kinds = interactions[:, rx_index, tx_index, path_index]
    if np.all(kinds == _NONE):
        return tx_m, rx_m, "los"
    last = int(np.max(np.nonzero(kinds != _NONE)[0]))
    start = np.round(vertices[last, rx_index, tx_index, path_index].astype(np.float64), 3)
    signature = "-".join(str(int(kind)) for kind in kinds if int(kind) != _NONE)
    delay_s = round(float(delays_s[rx_index, tx_index, path_index]), 9)
    return start, rx_m, f"nlos-{signature}-{delay_s:.9f}"


def blockage_for_snapshot(
    scenario: dict[str, Any],
    comm_paths: dict[str, np.ndarray],
    tx_positions: dict[str, np.ndarray],
    rx_positions: dict[str, np.ndarray],
    target_states: dict[str, dict[str, np.ndarray]],
    target_sizes: dict[str, tuple[float, float, float]],
    wavelength_m: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply model B to every comm path.

    Returns the attenuated coefficients (same shape as ``comm_paths["a"]``)
    and one record per (UE, O-RU, path, blocker).
    """
    oru_names = [oru["name"] for oru in scenario["orus"]]
    ue_names = [ue["name"] for ue in scenario["ues"]]
    coefficients = np.array(comm_paths["a"], copy=True)
    interactions = comm_paths["interactions"]
    vertices = comm_paths["vertices"]
    valid = comm_paths["valid"]
    n_rx, _n_rx_ant, n_tx, _n_tx_ant, n_paths, _n_time = coefficients.shape
    if interactions.ndim != 4:
        raise RuntimeError(
            "expected synthetic-array interactions "
            "[depth, rx, tx, paths]; got shape "
            f"{interactions.shape}"
        )
    records: list[dict[str, Any]] = []
    threshold_db = float(scenario["blockage"]["event_loss_db"])

    for rx_index in range(n_rx):
        for tx_index in range(n_tx):
            ue_name = ue_names[rx_index]
            oru_name = oru_names[tx_index]
            for path_index in range(n_paths):
                segment = _incoming_segment(
                    vertices,
                    interactions,
                    valid,
                    comm_paths["tau"],
                    path_index,
                    rx_index,
                    tx_index,
                    tx_positions[oru_name],
                    rx_positions[ue_name],
                )
                if segment is None:
                    continue
                start, end, path_id = segment
                per_blocker: list[float] = []
                for blocker_name, state in target_states.items():
                    length, width, height = target_sizes[blocker_name]
                    result: ScreenLoss = screen_blockage(
                        start,
                        end,
                        state["position_m"],
                        length,
                        width,
                        height,
                        wavelength_m,
                    )
                    per_blocker.append(result.loss_db)
                    loss_db = result.loss_db
                    blocked = (not math.isfinite(loss_db)) or loss_db >= threshold_db
                    records.append(
                        {
                            "ue": ue_name,
                            "oru": oru_name,
                            "path": path_id,
                            "blocker": blocker_name,
                            "loss_db": loss_db,
                            "intersects": bool(result.intersects),
                            "blocked": blocked,
                        }
                    )
                gain = linear_amplitude_gain(sum_blockage_db(per_blocker))
                coefficients[rx_index, :, tx_index, :, path_index, :] *= gain
    return coefficients, records


def _events(snapshot_rows: list[dict[str, Any]], times_s: list[float]) -> list[dict[str, Any]]:
    """Contiguous blockage intervals with start, end, and blocker id."""
    grouped: dict[tuple[str, str, str, str], list[int]] = {}
    for row in snapshot_rows:
        if not row["blocked"]:
            continue
        key = (row["ue"], row["oru"], row["path"], row["blocker"])
        grouped.setdefault(key, []).append(int(row["snapshot"]))

    events = []
    for (ue, oru, path, blocker), snapshots in grouped.items():
        snapshots = sorted(set(snapshots))
        start = snapshots[0]
        previous = snapshots[0]
        for index in snapshots[1:] + [None]:
            if index is not None and index == previous + 1:
                previous = index
                continue
            losses = [
                row["loss_db"]
                for row in snapshot_rows
                if row["ue"] == ue
                and row["oru"] == oru
                and row["path"] == path
                and row["blocker"] == blocker
                and start <= int(row["snapshot"]) <= previous
                and row["blocked"]
            ]
            finite = [loss for loss in losses if math.isfinite(loss)]
            events.append(
                {
                    "ue": ue,
                    "oru": oru,
                    "path": path,
                    "blocker_id": blocker,
                    "start_s": times_s[start],
                    "end_s": times_s[previous],
                    "max_loss_db": None if not finite else max(finite),
                }
            )
            if index is None:
                break
            start = previous = index
    events.sort(key=lambda item: (item["ue"], item["oru"], item["path"], item["blocker_id"], item["start_s"]))
    return events


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


def run_scenario(scenario: dict[str, Any], output_dir: Path, render: bool | None = None) -> dict[str, Any]:
    """Step the scenario and write channels, ground truth, and optional frames."""
    from sionna.rt.rcs import RCSSolver
    from sionna.rt import PathSolver

    output_dir.mkdir(parents=True, exist_ok=True)
    do_render = scenario["render"]["enabled"] if render is None else render
    comm, sensing, targets = build_scenes(scenario)
    target_sizes = {
        target.name: (float(target.length), float(target.width), float(target.height))
        for target in targets
    }
    oru_names = [oru["name"] for oru in scenario["orus"]]
    ue_names = [ue["name"] for ue in scenario["ues"]]
    radios = [comm.get(name) for name in oru_names + ue_names]
    sensing_radios = [sensing.get(name) for name in oru_names + ue_names]

    rcs_solver = RCSSolver(deterministic=True)
    background_solver = PathSolver()
    comm_solver = PathSolver()
    wavelength_m = float(np.asarray(comm.wavelength.numpy()).reshape(-1)[0])

    dt_s = float(scenario["dt_s"])
    n_snapshots = int(scenario["n_snapshots"])
    packed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    times_s: list[float] = []
    frame_paths: list[Path] = []
    frame_dir = output_dir / "frames"

    for index in range(n_snapshots):
        t_s = index * dt_s
        times_s.append(t_s)
        states = states_at(scenario, t_s)
        move_targets(sensing, targets, states)
        move_radios(radios, states)
        move_radios(sensing_radios, states)

        sensing_paths = pack_paths(rcs_solver(sensing, **_solver_kwargs(scenario, "rcs")))
        background_paths = pack_paths(
            background_solver(sensing, **_solver_kwargs(scenario, "background"))
        )
        comm_solved = comm_solver(comm, **_solver_kwargs(scenario, "comm"))
        comm_geometric = pack_paths(comm_solved)
        readback = {
            name: {
                "position_m": read_vec3(sensing.get(name).position),
                "velocity_mps": read_vec3(sensing.get(name).velocity),
            }
            for name in states
        }
        attenuated, records = blockage_for_snapshot(
            scenario,
            comm_geometric,
            _device_positions(comm, oru_names),
            _device_positions(comm, ue_names),
            {name: states[name] for name in target_sizes},
            target_sizes,
            wavelength_m,
        )
        for record in records:
            record["snapshot"] = index
            record["t_s"] = t_s
            rows.append(record)
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

    events = _events(rows, times_s)
    ground_truth = {
        "seed": int(scenario["seed"]),
        "dt_s": dt_s,
        "n_snapshots": n_snapshots,
        "carrier_hz": float(scenario["carrier_hz"]),
        "numerology": int(scenario["numerology"]),
        "subcarrier_spacing_hz": subcarrier_spacing_hz(int(scenario["numerology"])),
        "n_subcarriers": int(scenario["n_subcarriers"]),
        "wavelength_m": wavelength_m,
        "design": (
            "Comm paths are solved without sensing targets. "
            "Model B scales those coefficients. "
            "The sensing scene keeps the targets, so PathSolver blockage there stays binary."
        ),
        "events": events,
        "per_snapshot_blockage": rows,
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
    gt_path = output_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(_json_ready(ground_truth), indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
