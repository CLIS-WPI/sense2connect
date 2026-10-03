"""Monostatic radar loop on the street scene.

The O-RU transmits on one element and receives on the 8×8 array. Ideal
full duplex is an assumption: self-interference is not simulated. The
static canyon, solved once, is the twin's clutter prediction. The live
background still contains the targets, so they can shadow that clutter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sim.scenes.loop import _make_array, _scene_by_name, _solver_kwargs
from sim.scenes.motion import move_targets, states_at, track_identity
from sim.sensing.cfar import ca_cfar, local_maxima
from sim.sensing.channel import (
    add_thermal_noise,
    antenna_positions_m,
    apply_window,
    delay_doppler,
    doppler_hz,
    frequency_response,
    localize,
    n_lags,
    power_map,
    radial_velocity_mps,
    range_m,
)
from sim.sensing.waveform import Waveform


def build_radar(scenario: dict[str, Any]) -> dict[str, Any]:
    """Live scene (targets) and static scene (twin clutter) at ``oru-0``."""
    from sionna.rt import Receiver, Transmitter, load_scene
    from sionna.rt.rcs import RCSSolver, TR38901SensingTarget
    from sionna.rt import PathSolver

    spec = scenario["sensing_radar"]
    arrays = spec["arrays"]
    oru = scenario["orus"][0]
    position = [float(v) for v in oru["position_m"]]
    look = [position[0], 0.0, 1.5]
    filename = _scene_by_name(str(scenario["scene"]))
    live = load_scene(filename)
    static = load_scene(filename)
    for scene in (live, static):
        scene.frequency = float(scenario["carrier_hz"])
        scene.tx_array = _make_array(arrays["tx"])
        scene.rx_array = _make_array(arrays["rx"])
        scene.add(Transmitter("radar-tx", position=position, look_at=look))
        scene.add(Receiver("radar-rx", position=position, look_at=look))
    states = states_at(scenario, 0.0)
    targets = []
    for actor in list(scenario["vehicles"]) + list(scenario["pedestrians"]):
        state = states[actor["name"]]
        targets.append(
            TR38901SensingTarget(
                actor["name"],
                object_type=str(actor["object_type"]),
                length=float(actor["length_m"]),
                width=float(actor["width_m"]),
                height=float(actor["height_m"]),
                position=state["position_m"].tolist(),
                velocity=state["velocity_mps"].tolist(),
                random_sigma_s=False,
                random_phases=False,
                random_xpr=False,
            )
        )
    live.add(targets)
    wavelength = float(np.asarray(live.wavelength.numpy()).reshape(-1)[0])
    receiver = live.get("radar-rx")
    raw_orientation = receiver.orientation
    orientation = np.array(
        raw_orientation.numpy() if hasattr(raw_orientation, "numpy") else raw_orientation,
        dtype=np.float64,
    ).reshape(-1)[:3]
    return {
        "live": live,
        "static": static,
        "targets": targets,
        "rcs": RCSSolver(deterministic=True),
        "background": PathSolver(deterministic=True),
        "positions_m": antenna_positions_m(live.rx_array, wavelength),
        "orientation_rad": orientation,
        "radar_position_m": np.asarray(position, dtype=np.float64),
        "n_rx_ant": int(live.rx_array.num_ant),
    }


def ground_truth(scenario: dict[str, Any], t_s: float, radar_position_m: np.ndarray, range_m_max: float) -> list[dict]:
    """Identities inside the sensing range at time ``t_s`` [s].

    A wrap starts a new id. The previous lap is not in this list.
    """
    states = states_at(scenario, t_s)
    rows = []
    radar = np.asarray(radar_position_m, dtype=np.float64)
    for actor in list(scenario["vehicles"]) + list(scenario["pedestrians"]):
        state = states[actor["name"]]
        position = state["position_m"]
        if float(np.linalg.norm(position - radar)) > range_m_max:
            continue
        velocity = state["velocity_mps"]
        rows.append(
            {
                "id": track_identity(scenario, actor, t_s),
                "kind": str(actor["kind"]),
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "z_m": float(position[2]),
                "vx_mps": float(velocity[0]),
                "vy_mps": float(velocity[1]),
            }
        )
    return rows


def scan_scenario(
    scenario: dict[str, Any],
    waveform: Waveform | list[Waveform],
    cfar_grid: list[dict[str, float]],
    *,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Range-Doppler frames and CFAR hits for every clutter mode and CFAR setting.

    ``waveform`` may be a list. Every bandwidth is formed from the same
    traced paths. ``cfar_grid`` entries carry ``guard``, ``train``, and
    ``pfa``. Hits are already localized. Thermal noise is always added.
    Detection keys are ``(n_subcarriers, clutter, train, pfa)``.
    """
    waveforms = [waveform] if isinstance(waveform, Waveform) else list(waveform)
    if not waveforms:
        raise ValueError("scan_scenario needs at least one waveform")
    radar = build_radar(scenario)
    spec = scenario["sensing_radar"]
    noise = spec["noise"]
    if not bool(noise["enabled"]):
        raise RuntimeError("sensing_radar.noise.enabled must be true")
    static_paths = radar["background"](radar["static"], **_radar_kwargs(scenario, "background"))
    static = {
        item.n_subcarriers: frequency_response(static_paths, item, float(noise["tx_power_dbm"]))
        for item in waveforms
    }
    n_frames = int(scenario["n_snapshots"] if max_frames is None else min(max_frames, scenario["n_snapshots"]))
    dt_s = float(scenario["dt_s"])
    frames: list[dict[str, Any]] = []
    for index in range(n_frames):
        t_s = index * dt_s
        states = states_at(scenario, t_s)
        move_targets(radar["live"], radar["targets"], states)
        target_paths = radar["rcs"](radar["live"], **_radar_kwargs(scenario, "rcs"))
        live_paths = radar["background"](radar["live"], **_radar_kwargs(scenario, "background"))
        detections: dict[tuple[int, str, int, float], list[dict]] = {}
        for item in waveforms:
            measured = frequency_response(target_paths, item, float(noise["tx_power_dbm"]))
            measured = measured + frequency_response(live_paths, item, float(noise["tx_power_dbm"]))
            generator = torch.Generator(device=measured.device)
            generator.manual_seed(int(scenario["seed"]) * 100003 + index + item.n_subcarriers)
            measured = add_thermal_noise(
                measured,
                item,
                float(noise["noise_figure_db"]),
                float(noise["temperature_k"]),
                generator,
            )
            lags = n_lags(item, float(spec["max_range_m"]))
            cubes = {
                "blind": _cube(measured - measured.mean(dim=-2, keepdim=True), item, lags),
                "twin": _cube(measured - static[item.n_subcarriers], item, lags),
            }
            for mode, cube in cubes.items():
                power = power_map(cube)
                for setting in cfar_grid:
                    mask, _alpha = ca_cfar(
                        power,
                        guard=int(setting["guard"]),
                        train=int(setting["train"]),
                        pfa=float(setting["pfa"]),
                        noise_applied=True,
                    )
                    hits = []
                    for doppler_index, lag_index, peak in local_maxima(mask, power):
                        x_m, y_m = localize(
                            cube,
                            doppler_index,
                            lag_index,
                            item,
                            radar["positions_m"],
                            radar["orientation_rad"],
                            radar["radar_position_m"],
                        )
                        hits.append(
                            {
                                "x_m": x_m,
                                "y_m": y_m,
                                "range_m": range_m(lag_index, item),
                                "doppler_hz": doppler_hz(doppler_index, item),
                                "radial_velocity_mps": radial_velocity_mps(doppler_index, item),
                                "power": peak,
                                "snapshot": index,
                            }
                        )
                    detections[(item.n_subcarriers, mode, int(setting["train"]), float(setting["pfa"]))] = hits
        frames.append(
            {
                "t_s": t_s,
                "ground_truth": ground_truth(scenario, t_s, radar["radar_position_m"], float(spec["sensing_range_m"])),
                "detections": detections,
            }
        )
    return frames


def _radar_kwargs(scenario: dict[str, Any], kind: str) -> dict[str, Any]:
    """Solver arguments for the monostatic radar.

    RCS ``los`` keeps the unobstructed legs from the array to a target.
    The background solver turns that flag off so the colocated transmitter
    and receiver do not form a self-link.
    """
    kwargs = dict(_solver_kwargs(scenario, kind))
    kwargs["los"] = kind == "rcs"
    return kwargs


def _cube(response: torch.Tensor, waveform: Waveform, lags: int) -> torch.Tensor:
    if waveform.window:
        response = apply_window(response)
    return delay_doppler(response, lags)


def closest_bin(
    target: dict[str, Any],
    radar_position_m: np.ndarray,
    waveform: Waveform,
) -> tuple[float, float]:
    """Expected range [m] and approaching radial speed [m/s] of one target."""
    delta = np.array(
        [float(target["x_m"]) - radar_position_m[0], float(target["y_m"]) - radar_position_m[1], float(target["z_m"])],
        dtype=np.float64,
    )
    # z of the target center is not in the GT row's radar comparison above; callers pass z.
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        return 0.0, 0.0
    radial = delta / distance
    velocity = np.array([float(target["vx_mps"]), float(target["vy_mps"]), 0.0], dtype=np.float64)
    approaching = -float(np.dot(velocity, radial))
    return distance, approaching
