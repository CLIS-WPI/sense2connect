"""Image-method rejection of detections mirrored in a known wall."""

from __future__ import annotations

from typing import Any

import numpy as np


def mirror_y(point_m: np.ndarray, wall_y_m: float) -> np.ndarray:
    """Mirror a horizontal point across the vertical plane ``y = wall_y_m``."""
    mirrored = np.asarray(point_m, dtype=np.float64).copy()
    mirrored[1] = 2.0 * float(wall_y_m) - mirrored[1]
    return mirrored


def reject_ghosts(
    detections: list[dict[str, Any]],
    wall_y_m: list[float],
    gate_m: float,
) -> list[dict[str, Any]]:
    """Drop a detection that sits on the mirror of a stronger detection.

    The walls are the canyon facades known to the twin. A detection is kept
    when no stronger detection has a mirror within ``gate_m`` [m].
    """
    if gate_m <= 0.0:
        raise ValueError("ghost gate must be positive")
    kept: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        point = np.array([float(detection["x_m"]), float(detection["y_m"])], dtype=np.float64)
        power = float(detection["power"])
        ghost = False
        for other_index, other in enumerate(detections):
            if other_index == index or float(other["power"]) <= power:
                continue
            origin = np.array([float(other["x_m"]), float(other["y_m"])], dtype=np.float64)
            for wall in wall_y_m:
                image = mirror_y(origin, float(wall))
                if float(np.linalg.norm(image - point)) <= gate_m:
                    ghost = True
                    break
            if ghost:
                break
        if not ghost:
            kept.append(detection)
    return kept
