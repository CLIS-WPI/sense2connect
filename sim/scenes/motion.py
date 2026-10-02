"""Trajectories and a single batched pose update per snapshot.

Velocity is the vector stored on the actor. It is not estimated from the
change in position.
"""

from __future__ import annotations

import math
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
    length_m: float | None = None,
    lateral_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Point at arc length ``s_m`` [m] and the matching velocity [m/s].

    When ``length_m`` is set, the arc length wraps on that period [m].
    The velocity stays ``speed_mps`` along ``direction``; the wrap does
    not change it. ``lateral_m`` is a cross-street shift [m] added to y.
    """
    forward = _unit(direction)
    origin = _as_vec(origin_m)
    distance = s_m
    if length_m is not None and length_m > 0.0:
        distance = float(distance % length_m)
    position = origin + forward * distance
    position[1] += lateral_m
    position[2] = height_m
    velocity = forward * speed_mps
    velocity[2] = 0.0
    return position, velocity


def track_identity(scenario: dict[str, Any], actor: dict[str, Any], t_s: float) -> str:
    """Ground-truth id of one blocker at time ``t_s`` [s].

    The body keeps circulating on the periodic street, but each time its
    along-street coordinate wraps, the old id ends and a new one starts.
    The suffix is the lap index.
    """
    length_m, unwrapped_m = _unwrapped_arc(scenario, actor, t_s)
    if length_m <= 0.0:
        lap = 0
    else:
        lap = math.floor(unwrapped_m / length_m)
    return f"{actor['name']}#{int(lap)}"


def _axis_length(scenario: dict[str, Any], actor: dict[str, Any]) -> float:
    if "lane" in actor:
        lane = next(item for item in scenario["lanes"] if item["name"] == actor["lane"])
        return float(lane["length_m"])
    if "sidewalk" in actor:
        walk = next(item for item in scenario["sidewalks"] if item["name"] == actor["sidewalk"])
        return float(walk["length_m"])
    return 0.0


def _unwrapped_arc(scenario: dict[str, Any], actor: dict[str, Any], t_s: float) -> tuple[float, float]:
    """Return ``(period [m], unwrapped along-street coordinate [m])``."""
    length_m = _axis_length(scenario, actor)
    if actor.get("motion") == "crossing":
        return length_m, _crossing_unwrapped_arc(scenario, actor, t_s, length_m)
    return length_m, float(actor["s0_m"]) + float(actor["speed_mps"]) * t_s


def _crossing_unwrapped_arc(
    scenario: dict[str, Any],
    actor: dict[str, Any],
    t_s: float,
    length_m: float,
) -> float:
    """Along-street coordinate [m] of a crossing walker, without the street modulo.

    Each full traversal of a sidewalk adds one period. A crossing itself
    holds the along-street coordinate fixed.
    """
    sidewalks = {walk["name"]: walk for walk in scenario["sidewalks"]}
    start = sidewalks[actor["sidewalk"]]
    other = sidewalks[actor["other_sidewalk"]]
    direction = float(actor["direction"])
    speed = float(actor["speed_mps"])
    s0 = float(actor["s0_m"]) % length_m
    s_cross = float(actor["s_cross_m"]) % length_m
    to_cross = (s_cross - s0) % length_m if direction > 0.0 else (s0 - s_cross) % length_m
    height_m = float(actor["center_height_m"])
    pos_a, _ = place_on_axis(start["origin_m"], start["direction"], s_cross, height_m, 0.0, length_m=length_m)
    pos_b, _ = place_on_axis(other["origin_m"], other["direction"], s_cross, height_m, 0.0, length_m=length_m)
    width_m = float(np.linalg.norm(pos_b - pos_a))
    dist = speed * t_s
    if dist <= to_cross:
        return s0 + direction * dist
    travel = dist - to_cross
    cycle_m = 2.0 * (width_m + length_m)
    n_cycles = math.floor(travel / cycle_m) if cycle_m > 0.0 else 0
    rem = travel - n_cycles * cycle_m
    arc = s_cross + direction * (2 * n_cycles) * length_m
    if rem < width_m:
        return arc
    rem -= width_m
    if rem < length_m:
        return arc + direction * rem
    rem -= length_m
    arc += direction * length_m
    if rem < width_m:
        return arc
    rem -= width_m
    return arc + direction * rem


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
            length_m=float(walk["length_m"]) if "length_m" in walk else None,
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
            length_m=float(lane["length_m"]) if "length_m" in lane else None,
            lateral_m=float(vehicle.get("lateral_m", 0.0)),
        )
        states[vehicle["name"]] = {"position_m": position, "velocity_mps": velocity}

    for pedestrian in scenario["pedestrians"]:
        if pedestrian.get("motion") == "crossing":
            position, velocity = crossing_state(pedestrian, sidewalks, t_s)
        elif "sidewalk" in pedestrian:
            walk = sidewalks[pedestrian["sidewalk"]]
            position, velocity = place_on_axis(
                walk["origin_m"],
                walk["direction"],
                float(pedestrian["s0_m"]) + float(pedestrian["speed_mps"]) * t_s,
                float(pedestrian["center_height_m"]),
                float(pedestrian["speed_mps"]),
                length_m=float(walk["length_m"]) if "length_m" in walk else None,
                lateral_m=float(pedestrian.get("lateral_m", 0.0)),
            )
        else:
            position, velocity = actor_state(pedestrian, t_s)
        states[pedestrian["name"]] = {
            "position_m": position,
            "velocity_mps": velocity,
        }
    return states


def crossing_state(
    pedestrian: dict[str, Any],
    sidewalks: dict[str, dict[str, Any]],
    t_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Position [m] and velocity [m/s] of a pedestrian who also crosses the street.

    The walker moves along the start sidewalk until ``s_cross_m``, crosses
    to the other sidewalk at that station, walks one full period, and
    crosses back. The pattern then repeats. Speed is the walker's own
    speed. The crossing station is not chosen from a radio position.
    """
    start = sidewalks[pedestrian["sidewalk"]]
    other = sidewalks[pedestrian["other_sidewalk"]]
    length_m = float(start["length_m"])
    speed = float(pedestrian["speed_mps"])
    direction = float(pedestrian["direction"])
    if speed <= 0.0 or direction == 0.0:
        raise ValueError("a crossing pedestrian needs a positive speed and a direction sign")
    s0 = float(pedestrian["s0_m"]) % length_m
    s_cross = float(pedestrian["s_cross_m"]) % length_m
    height_m = float(pedestrian["center_height_m"])
    to_cross = (s_cross - s0) % length_m if direction > 0.0 else (s0 - s_cross) % length_m
    pos_a, _ = place_on_axis(
        start["origin_m"], start["direction"], s_cross, height_m, 0.0, length_m=length_m
    )
    pos_b, _ = place_on_axis(
        other["origin_m"], other["direction"], s_cross, height_m, 0.0, length_m=length_m
    )
    span = pos_b - pos_a
    width_m = float(np.linalg.norm(span))
    if width_m < 1e-8:
        raise ValueError("the two sidewalks meet, so there is no crossing")

    def walk(sidewalk: dict[str, Any], s_from: float, walked_m: float) -> tuple[np.ndarray, np.ndarray]:
        return place_on_axis(
            sidewalk["origin_m"],
            sidewalk["direction"],
            s_from + direction * walked_m,
            height_m,
            direction * speed,
            length_m=length_m,
        )

    dist = speed * t_s
    if dist <= to_cross:
        return walk(start, s0, dist)
    dist -= to_cross
    cycle_m = 2.0 * (width_m + length_m)
    dist = dist % cycle_m
    if dist < width_m:
        fraction = dist / width_m
        return pos_a + span * fraction, span / width_m * speed
    dist -= width_m
    if dist < length_m:
        return walk(other, s_cross, dist)
    dist -= length_m
    if dist < width_m:
        fraction = dist / width_m
        return pos_b - span * fraction, -span / width_m * speed
    dist -= width_m
    return walk(start, s_cross, dist)


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
