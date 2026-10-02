"""Trajectories and a single batched pose update per snapshot.

Velocity is the vector stored on the actor. It is not estimated from the
change in position.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_vec(values: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(3)
    return vector.copy()


def _unit(direction: list[float]) -> np.ndarray:
    vector = _as_vec(direction)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("trajectory direction must be non-zero")
    return vector / norm


def actor_state(actor: dict[str, Any], t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Position [m] and velocity [m/s] of one actor at time ``t_s`` [s]."""
    if "velocity_mps" in actor and "position0_m" in actor:
        velocity = _as_vec(actor["velocity_mps"])
        position = _as_vec(actor["position0_m"]) + velocity * t_s
        return position, velocity

    raise KeyError("actor needs position0_m and velocity_mps")


def place_on_axis(
    origin_m: list[float],
    direction: list[float],
    s_m: float,
    height_m: float,
    speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Point at arc length ``s_m`` [m] and the matching velocity [m/s]."""
    forward = _unit(direction)
    origin = _as_vec(origin_m)
    position = origin + forward * s_m
    position[2] = height_m
    velocity = forward * speed_mps
    velocity[2] = 0.0
    return position, velocity


def states_at(scenario: dict[str, Any], t_s: float) -> dict[str, dict[str, np.ndarray]]:
    """Every named mover at time ``t_s`` [s].

    Keys are ``ue``, ``vehicle``, and ``pedestrian`` actor names. Each
    value has ``position_m`` and ``velocity_mps``.
    """
    lanes = {lane["name"]: lane for lane in scenario["lanes"]}
    sidewalks = {walk["name"]: walk for walk in scenario["sidewalks"]}
    states: dict[str, dict[str, np.ndarray]] = {}

    for ue in scenario["ues"]:
        walk = sidewalks[ue["sidewalk"]]
        position, velocity = place_on_axis(
            walk["origin_m"],
            walk["direction"],
            float(ue["s0_m"]) + float(ue["speed_mps"]) * t_s,
            float(ue["height_m"]),
            float(ue["speed_mps"]),
        )
        states[ue["name"]] = {"position_m": position, "velocity_mps": velocity}

    for vehicle in scenario["vehicles"]:
        lane = lanes[vehicle["lane"]]
        position, velocity = place_on_axis(
            lane["origin_m"],
            lane["direction"],
            float(vehicle["s0_m"]) + float(vehicle["speed_mps"]) * t_s,
            float(vehicle["center_height_m"]),
            float(vehicle["speed_mps"]),
        )
        states[vehicle["name"]] = {"position_m": position, "velocity_mps": velocity}

    for pedestrian in scenario["pedestrians"]:
        position, velocity = actor_state(pedestrian, t_s)
        states[pedestrian["name"]] = {
            "position_m": position,
            "velocity_mps": velocity,
        }
    return states


def read_vec3(value: Any) -> np.ndarray:
    """First three components of a Mitsuba or NumPy vector, in metres or m/s."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array[:3].copy()


def move_targets(scene: Any, targets: list[Any], states: dict[str, dict[str, np.ndarray]]) -> None:
    """Move every target, then apply one Mitsuba parameter update.

    Radio devices are not included: their positions are plain vectors and
    do not rebuild the scene.
    """
    import drjit as dr
    import mitsuba as mi

    params = scene.mi_scene_params
    for target in targets:
        state = states[target.name]
        current = read_vec3(target.position)
        delta = state["position_m"] - current
        key = target._mi_mesh.id() + ".vertex_positions"
        translation = mi.Vector3f(float(delta[0]), float(delta[1]), float(delta[2]))
        vertices = dr.unravel(mi.Point3f, params[key])
        params[key] = dr.ravel(vertices + translation)
        target.velocity = mi.Vector3f(
            float(state["velocity_mps"][0]),
            float(state["velocity_mps"][1]),
            float(state["velocity_mps"][2]),
        )
    params.update()
    scene.scene_geometry_updated()


def move_radios(devices: list[Any], states: dict[str, dict[str, np.ndarray]]) -> None:
    """Set radio positions [m] and velocities [m/s] from ``states``."""
    for device in devices:
        if device.name not in states:
            continue
        state = states[device.name]
        device.position = state["position_m"].tolist()
        device.velocity = state["velocity_mps"].tolist()
