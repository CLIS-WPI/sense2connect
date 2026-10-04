"""Replay the M2 map tracker from cached detections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.sensing.cluster import cluster_detections
from sim.sensing.ghost import reject_ghosts
from sim.sensing.track_map import MapTracker


def load_detections(directory: Path) -> dict[str, Any]:
    """Load ``detections_1024.json``. Provenance is recorded, not refused.

    The M2 detections were stamped before the M3 tree. Replaying the
    tracker does not re-trace.
    """
    payload = json.loads((directory / "detections_1024.json").read_text(encoding="utf-8"))
    events_payload = json.loads((directory / "events.json").read_text(encoding="utf-8"))
    payload["events"] = events_payload["events"] if isinstance(events_payload, dict) else events_payload
    payload["trace_provenance"] = payload.get("provenance")
    return payload


def replay_map(
    payload: dict[str, Any],
    *,
    train: int,
    pfa: float,
    eps_m: float,
    ghost_association_m: float,
    walls: list[float],
    spec: dict[str, Any],
    lanes: list[dict],
    sidewalks: list[dict],
    tuned: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    """Confirmed-track rows for every 0.1 s snapshot."""
    tracker = MapTracker(
        dt_s=0.1,
        association_gate_m=float(tuned["association_m"]),
        coast_frames=int(tuned["coast"]),
        confirm_hits=int(spec["tracker"]["confirm_hits"]),
        radar_position_m=np.asarray(payload["radar_position_m"], dtype=np.float64),
        process_q=float(tuned["process_q"]),
        lanes=lanes,
        sidewalks=sidewalks,
        lane_gate_m=float(tuned["lane_gate_m"]),
        sidewalk_gate_m=float(tuned["sidewalk_gate_m"]),
        leave_m=float(tuned["leave_m"]),
    )
    key = f"blind:{int(train)}:{float(pfa)}"
    frames = []
    for frame in payload["frames"]:
        detections = reject_ghosts(list(frame["detections"][key]), walls, float(ghost_association_m))
        clusters = cluster_detections(detections, float(eps_m), int(spec["cluster"]["min_samples"]))
        alive = tracker.step(clusters)
        frames.append(
            [
                {
                    "id": int(track.identifier),
                    "x_m": float(track.state[0]),
                    "y_m": float(track.state[1]),
                    "z_m": float(track.state[2]),
                    "vx_mps": float(track.state[3]),
                    "vy_mps": float(track.state[4]),
                    "confirmed": bool(track.confirmed),
                    "mode": str(getattr(track, "mode", "free")),
                    "line_y_m": getattr(track, "line_y_m", None),
                    "age_s": float(track.age_s),
                }
                for track in alive
            ]
        )
    return frames
