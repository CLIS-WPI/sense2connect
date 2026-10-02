"""Turn the deployment YAML into radios, vehicles, and pedestrian groups.

Initial positions and speeds are drawn from the scenario seed. Nothing is
rejected or shifted because of a blockage result.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")


def prepare_scenario(
    raw: dict[str, Any],
    *,
    seed: int | None = None,
    mount: str | None = None,
    density: str | None = None,
    duration_s: float | None = None,
    dt_s: float | None = None,
) -> dict[str, Any]:
    """Return a scenario with concrete O-RU, vehicles, and pedestrians."""
    scenario = copy.deepcopy(raw)
    if seed is not None:
        scenario["seed"] = int(seed)
    if mount is not None:
        scenario["mount"] = mount
    if density is not None:
        scenario["density"] = density
    if duration_s is not None:
        scenario["duration_s"] = float(duration_s)
    if dt_s is not None:
        scenario["dt_s"] = float(dt_s)

    mount_name = str(scenario["mount"])
    if mount_name not in scenario["oru_mounts"]:
        raise ValueError(f"unknown mount {mount_name}")
    mount_spec = scenario["oru_mounts"][mount_name]
    height_m = float(mount_spec["height_m"])
    scenario["oru_height_m"] = height_m
    first = {
        "name": "oru-0",
        "mount": mount_name,
        "height_m": height_m,
        "position_m": [
            float(mount_spec["x_m"]),
            float(mount_spec["y_m"]),
            height_m,
        ],
    }
    scenario["orus"] = [first]
    second_cfg = scenario.get("second_oru") or {}
    if bool(second_cfg.get("enabled", False)):
        scenario["orus"].append(_second_oru(scenario, first, second_cfg))

    density_name = str(scenario["density"])
    counts = scenario["traffic"][density_name]
    kinds = scenario["blocker_kinds"]
    lanes = [lane["name"] for lane in scenario["lanes"]]
    rng = np.random.Generator(np.random.PCG64(int(scenario["seed"])))

    vehicles: list[dict[str, Any]] = []
    fleet = (
        [("car", int(counts["n_cars"]))]
        + [("bus", int(counts["n_buses"]))]
        + [("truck", int(counts["n_trucks"]))]
    )
    serial = 0
    for kind, count in fleet:
        spec = kinds[kind]
        speed_low, speed_high = (float(v) for v in spec["speed_mps"])
        for _ in range(count):
            lane_name = lanes[int(rng.integers(0, len(lanes)))]
            lane = next(item for item in scenario["lanes"] if item["name"] == lane_name)
            vehicles.append(
                _actor(
                    name=f"{kind}-{serial}",
                    kind=kind,
                    spec=spec,
                    axis_name=lane_name,
                    axis_key="lane",
                    s0_m=float(rng.uniform(0.0, float(lane["length_m"]))),
                    speed_mps=float(rng.uniform(speed_low, speed_high)),
                )
            )
            serial += 1
    scenario["vehicles"] = vehicles

    scenario["pedestrians"] = _pedestrians(scenario, kinds["pedestrian"], counts, rng)

    dt_s = float(scenario["dt_s"])
    scenario["n_snapshots"] = int(round(float(scenario["duration_s"]) / dt_s))
    if scenario["n_snapshots"] < 1:
        raise ValueError("duration_s must cover at least one snapshot")
    return scenario


def _second_oru(scenario: dict[str, Any], first: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Lamppost on the opposite sidewalk, shifted along the street from the first O-RU.

    The along-street offset wraps on the lane length so the lamp stays on the
    modeled block. The sidewalk is the one across the street from the first
    O-RU. Neither choice is taken from a blockage result.
    """
    sidewalks = list(scenario["sidewalks"])
    ys = [float(item["origin_m"][1]) for item in sidewalks]
    mid_y = 0.5 * (min(ys) + max(ys))
    first_y = float(first["position_m"][1])
    opposite_y = min(ys) if first_y >= mid_y else max(ys)
    lane = scenario["lanes"][0]
    origin_x = float(lane["origin_m"][0])
    length_m = float(lane["length_m"])
    offset_m = float(spec["x_offset_m"])
    raw_x = float(first["position_m"][0]) + offset_m
    x_m = origin_x + (raw_x - origin_x) % length_m
    height_m = float(spec.get("height_m", 5.0))
    return {
        "name": "oru-1",
        "mount": "lamppost",
        "height_m": height_m,
        "position_m": [x_m, opposite_y, height_m],
    }


def _pedestrians(
    scenario: dict[str, Any],
    spec: dict[str, Any],
    counts: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Independent sidewalk walkers and street crossings.

    Sidewalk, direction, speed, and starting arc length are drawn from the
    seed. A crossing uses a random along-street station. Nothing is placed
    from a UE position or from a blockage result.
    """
    sidewalks = list(scenario["sidewalks"])
    if len(sidewalks) < 2:
        raise ValueError("crossings need two sidewalks")
    speed_low, speed_high = (float(v) for v in spec["speed_mps"])
    pedestrians: list[dict[str, Any]] = []
    serial = 0
    for _ in range(int(counts["n_sidewalk_pedestrians"])):
        walk = sidewalks[int(rng.integers(0, len(sidewalks)))]
        sign = 1.0 if float(rng.random()) < 0.5 else -1.0
        pedestrians.append(
            _actor(
                name=f"ped-{serial}",
                kind="pedestrian",
                spec=spec,
                axis_name=walk["name"],
                axis_key="sidewalk",
                s0_m=float(rng.uniform(0.0, float(walk["length_m"]))),
                speed_mps=sign * float(rng.uniform(speed_low, speed_high)),
                motion="sidewalk",
            )
        )
        serial += 1
    for _ in range(int(counts["n_crossing_pedestrians"])):
        start = sidewalks[int(rng.integers(0, len(sidewalks)))]
        others = [item for item in sidewalks if item["name"] != start["name"]]
        other = others[int(rng.integers(0, len(others)))]
        sign = 1.0 if float(rng.random()) < 0.5 else -1.0
        length_m = float(start["length_m"])
        actor = _actor(
            name=f"ped-{serial}",
            kind="pedestrian",
            spec=spec,
            axis_name=start["name"],
            axis_key="sidewalk",
            s0_m=float(rng.uniform(0.0, length_m)),
            speed_mps=float(rng.uniform(speed_low, speed_high)),
            motion="crossing",
        )
        actor["direction"] = sign
        actor["other_sidewalk"] = other["name"]
        actor["s_cross_m"] = float(rng.uniform(0.0, length_m))
        pedestrians.append(actor)
        serial += 1
    return pedestrians


def _actor(
    *,
    name: str,
    kind: str,
    spec: dict[str, Any],
    axis_name: str,
    axis_key: str,
    s0_m: float,
    speed_mps: float,
    lateral_m: float = 0.0,
    motion: str = "axis",
) -> dict[str, Any]:
    height_m = float(spec["height_m"])
    return {
        "name": name,
        "kind": kind,
        "object_type": str(spec["object_type"]),
        "length_m": float(spec["length_m"]),
        "width_m": float(spec["width_m"]),
        "height_m": height_m,
        "center_height_m": 0.5 * height_m,
        axis_key: axis_name,
        "s0_m": float(s0_m),
        "speed_mps": float(speed_mps),
        "lateral_m": float(lateral_m),
        "motion": motion,
    }
