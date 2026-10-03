"""DBSCAN on horizontal detection coordinates [m]."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def dbscan(points_m: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    """Return a cluster id per point. Noise points are ``-1``.

    With ``min_samples == 1`` every point is a core point, so a cluster is
    a connected component under distance ``eps_m``.
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_m must have shape [n, 2]")
    n = points.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    if eps_m <= 0.0 or min_samples < 1:
        raise ValueError("eps_m must be positive and min_samples at least 1")
    neighbors = [np.flatnonzero(np.linalg.norm(points - points[index], axis=1) <= eps_m) for index in range(n)]
    cluster = 0
    visited = np.zeros(n, dtype=bool)
    for index in range(n):
        if visited[index]:
            continue
        visited[index] = True
        if neighbors[index].size < min_samples:
            continue
        labels[index] = cluster
        queue = list(neighbors[index].tolist())
        while queue:
            point = queue.pop()
            if not visited[point]:
                visited[point] = True
                if neighbors[point].size >= min_samples:
                    queue.extend(int(item) for item in neighbors[point] if not visited[int(item)])
            if labels[point] < 0:
                labels[point] = cluster
        cluster += 1
    return labels


def cluster_detections(detections: list[dict[str, Any]], eps_m: float, min_samples: int) -> list[dict[str, Any]]:
    """One detection per DBSCAN cluster, at the power-weighted centroid [m]."""
    if not detections:
        return []
    points = np.array([[float(item["x_m"]), float(item["y_m"])] for item in detections], dtype=np.float64)
    labels = dbscan(points, eps_m, min_samples)
    clustered: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        members = [detections[index] for index in range(len(detections)) if int(labels[index]) == label]
        weights = np.array([max(float(item["power"]), 0.0) for item in members], dtype=np.float64)
        if not math.isfinite(weights.sum()) or weights.sum() <= 0.0:
            weights = np.ones(len(members), dtype=np.float64)
        weights = weights / weights.sum()
        centroid = np.sum(points[labels == label] * weights[:, None], axis=0)
        strongest = max(members, key=lambda item: float(item["power"]))
        clustered.append(
            {
                "x_m": float(centroid[0]),
                "y_m": float(centroid[1]),
                "z_m": float(strongest.get("z_m", 0.0)),
                "power": float(sum(float(item["power"]) for item in members)),
                "radial_velocity_mps": float(strongest["radial_velocity_mps"]),
                "range_m": float(strongest["range_m"]),
                "n_points": len(members),
                "snapshot": int(strongest.get("snapshot", -1)),
            }
        )
    return clustered
