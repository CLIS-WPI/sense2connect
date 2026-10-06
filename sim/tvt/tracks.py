"""Blocker-track posteriors for the TVT milestones.

``replay_with_covariance`` replays the frozen paper-1 map tracker (sim/sensing/track_map.py,
via the same steps as xapp/tracks.replay_map) and additionally exports each track's EKF
covariance (state x, y, z, vx, vy). ``pack_posteriors`` turns confirmed tracks into arrays
with the paper-1 size rule (lane -> bus, sidewalk -> pedestrian, free -> bus; class-agnostic,
as the paper-1 predictor). ``truth_posteriors`` gives the perfect tracks (true states, true
sizes and classes, zero covariance). All functions are read-only users of the frozen code.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def replay_with_covariance(payload: dict[str, Any], *, train: int, pfa: float, eps_m: float, ghost_association_m: float, walls: list[float],
                           spec: dict[str, Any], lanes: list[dict], sidewalks: list[dict], tuned: dict[str, Any]) -> list[list[dict[str, Any]]]:
    from sim.sensing.cluster import cluster_detections
    from sim.sensing.ghost import reject_ghosts
    from sim.sensing.track_map import MapTracker

    tracker = MapTracker(
        dt_s=0.1, association_gate_m=float(tuned["association_m"]), coast_frames=int(tuned["coast"]), confirm_hits=int(spec["tracker"]["confirm_hits"]),
        radar_position_m=np.asarray(payload["radar_position_m"], dtype=np.float64), process_q=float(tuned["process_q"]), lanes=lanes, sidewalks=sidewalks,
        lane_gate_m=float(tuned["lane_gate_m"]), sidewalk_gate_m=float(tuned["sidewalk_gate_m"]), leave_m=float(tuned["leave_m"]))
    key = f"blind:{int(train)}:{float(pfa)}"
    frames = []
    for frame in payload["frames"]:
        detections = reject_ghosts(list(frame["detections"][key]), walls, float(ghost_association_m))
        clusters = cluster_detections(detections, float(eps_m), int(spec["cluster"]["min_samples"]))
        alive = tracker.step(clusters)
        frames.append([{"id": int(t.identifier), "state": np.array(t.state, dtype=np.float64), "cov": np.array(t.covariance, dtype=np.float64),
                        "confirmed": bool(t.confirmed), "mode": str(getattr(t, "mode", "free")), "line_y_m": getattr(t, "line_y_m", None)} for t in alive])
    return frames


def pack_posteriors(frames: list[list[dict[str, Any]]], sizes: dict[str, tuple[float, float, float]]) -> dict[str, np.ndarray]:
    """Confirmed tracks per epoch padded to K: mean [R, K, 5] (x, y, z, vx, vy), cov [R, K, 5, 5], size [R, K, 3], valid [R, K], kind [R, K] (0 vehicle-lane/free, 1 sidewalk)."""
    conf = [[row for row in rows if row["confirmed"]] for rows in frames]
    k = max(1, max((len(r) for r in conf), default=1))
    n = len(frames)
    mean = np.zeros((n, k, 5))
    cov = np.zeros((n, k, 5, 5))
    size = np.ones((n, k, 3))
    valid = np.zeros((n, k), dtype=bool)
    kind = np.zeros((n, k), dtype=np.int64)
    for r, rows in enumerate(conf):
        for i, row in enumerate(rows):
            st = row["state"][:5].copy()  # tracker state [x, y, z, vx, vy, vz] -> (x, y, z, vx, vy)
            if row["mode"] in ("lane", "sidewalk") and row.get("line_y_m") is not None:
                st[1], st[4] = float(row["line_y_m"]), 0.0
            mean[r, i] = st
            cv = row["cov"][:5, :5].copy()
            if row["mode"] in ("lane", "sidewalk") and row.get("line_y_m") is not None:
                cv[1, :], cv[:, 1], cv[4, :], cv[:, 4] = 0.0, 0.0, 0.0, 0.0  # pinned to the line
            cov[r, i] = cv
            sw = row["mode"] == "sidewalk"
            size[r, i] = sizes["pedestrian"] if sw else sizes["bus"]
            kind[r, i] = 1 if sw else 0
            valid[r, i] = True
    return {"mean": mean, "cov": cov, "size": size, "valid": valid, "kind": kind}


def truth_posteriors(scenario: dict, times_s: np.ndarray) -> dict[str, Any]:
    """Perfect tracks: true blocker states [R, B, 5], zero covariance, true sizes, classes; plus UE truth."""
    from sim.scenes.motion import states_at

    bl = list(scenario["vehicles"]) + list(scenario["pedestrians"])
    ues = [u["name"] for u in scenario["ues"]]
    R, B = len(times_s), len(bl)
    mean = np.zeros((R, B, 5))
    ue = np.zeros((R, len(ues), 3))
    uev = np.zeros((R, len(ues), 3))
    for r, t in enumerate(times_s):
        s = states_at(scenario, float(t))
        for b, spec in enumerate(bl):
            p, v = s[spec["name"]]["position_m"], s[spec["name"]]["velocity_mps"]
            mean[r, b] = (p[0], p[1], p[2], v[0], v[1])
        for u, name in enumerate(ues):
            ue[r, u] = s[name]["position_m"]
            uev[r, u] = s[name]["velocity_mps"]
    size = np.array([[float(b["length_m"]), float(b["width_m"]), float(b["height_m"])] for b in bl])
    return {"mean": mean, "cov": np.zeros((R, B, 5, 5)), "size": np.broadcast_to(size, (R, B, 3)).copy(), "valid": np.ones((R, B), dtype=bool),
            "cls": [str(b["kind"]) for b in bl], "ue": ue, "ue_vel": uev}
