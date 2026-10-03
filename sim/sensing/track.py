"""Extended Kalman filter on range, azimuth, elevation, and radial velocity.

The state is Cartesian position and velocity. The first detection takes its
radial velocity from Doppler. The second detection replaces the tangential
part with two-point differencing and keeps the Doppler radial component.
Velocity is never initialised at rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    """One track. ``state`` is ``[x, y, z, vx, vy, vz]`` in metres and m/s."""

    identifier: int
    state: np.ndarray
    covariance: np.ndarray
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    birth_s: float = 0.0
    age_s: float = 0.0
    measured_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_s: float = 0.0


@dataclass
class Tracker:
    """Gated EKF. ``dt_s`` is the snapshot interval [s], not the CPI length."""

    dt_s: float
    association_gate_m: float
    coast_frames: int
    confirm_hits: int = 2
    radar_position_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    range_sigma_m: float = 1.5
    angle_sigma_rad: float = np.deg2rad(2.0)
    radial_sigma_mps: float = 0.5
    process_q: float = 1.0
    _next_id: int = 1
    tracks: list[Track] = field(default_factory=list)
    _time_s: float = 0.0
    _started: bool = False

    def step(self, detections: list[dict]) -> list[Track]:
        """Predict, assign, and return the tracks that are still alive."""
        dt = float(self.dt_s)
        if self._started:
            self._time_s += dt
        self._started = True
        radar = np.asarray(self.radar_position_m, dtype=np.float64)
        for track in self.tracks:
            track.state, track.covariance = _predict(track.state, track.covariance, dt, self.process_q)
            track.age_s = self._time_s - track.birth_s
        pairs = _assign(self.tracks, detections, self.association_gate_m)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for track_index, det_index in pairs:
            track = self.tracks[track_index]
            measurement, position = _measurement(detections[det_index], radar)
            if track.hits == 1:
                _apply_two_point(track, position, measurement, radar, self._time_s)
            track.state, track.covariance = _update(track.state, track.covariance, measurement, radar, self)
            track.measured_m = position
            track.measured_s = self._time_s
            track.hits += 1
            track.misses = 0
            track.confirmed = track.hits >= self.confirm_hits
            used_tracks.add(track_index)
            used_dets.add(det_index)
        for index, track in enumerate(self.tracks):
            if index not in used_tracks:
                track.misses += 1
        for index, detection in enumerate(detections):
            if index in used_dets:
                continue
            measurement, position = _measurement(detection, radar)
            velocity = -float(measurement[3]) * _unit(position - radar)
            variance_p = self.range_sigma_m ** 2
            variance_v = 4.0
            state = np.concatenate([position, velocity])
            self.tracks.append(
                Track(
                    identifier=self._next_id,
                    state=state,
                    covariance=np.diag([variance_p, variance_p, variance_p, variance_v, variance_v, variance_v]),
                    hits=1,
                    confirmed=self.confirm_hits <= 1,
                    birth_s=self._time_s,
                    age_s=0.0,
                    measured_m=position.copy(),
                    measured_s=self._time_s,
                )
            )
            self._next_id += 1
        self.tracks = [track for track in self.tracks if track.misses <= self.coast_frames]
        return list(self.tracks)


def _measurement(detection: dict, radar_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(
        [float(detection["x_m"]), float(detection["y_m"]), float(detection.get("z_m", radar_m[2]))],
        dtype=np.float64,
    )
    delta = position - radar_m
    distance = float(np.linalg.norm(delta))
    horizontal = float(np.hypot(delta[0], delta[1]))
    azimuth = float(np.atan2(delta[1], delta[0]))
    elevation = float(np.atan2(delta[2], horizontal if horizontal > 1e-9 else 1e-9))
    radial = float(detection["radial_velocity_mps"])
    return np.array([distance, azimuth, elevation, radial], dtype=np.float64), position


def _apply_two_point(
    track: Track,
    position: np.ndarray,
    measurement: np.ndarray,
    radar_m: np.ndarray,
    time_s: float,
) -> None:
    """Tangential velocity from the two positions, radial velocity from Doppler."""
    dt_s = time_s - track.measured_s
    if dt_s <= 0.0:
        return
    finite_difference = (position - track.measured_m) / dt_s
    radial_hat = _unit(position - radar_m)
    tangential = finite_difference - np.dot(finite_difference, radial_hat) * radial_hat
    track.state[3:] = tangential - float(measurement[3]) * radial_hat


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _predict(state: np.ndarray, covariance: np.ndarray, dt_s: float, process_q: float) -> tuple[np.ndarray, np.ndarray]:
    transition = np.eye(6)
    transition[0, 3] = dt_s
    transition[1, 4] = dt_s
    transition[2, 5] = dt_s
    q = float(process_q)
    process = np.zeros((6, 6))
    for axis in range(3):
        process[axis, axis] = q * dt_s**3 / 3.0
        process[axis, axis + 3] = q * dt_s**2 / 2.0
        process[axis + 3, axis] = q * dt_s**2 / 2.0
        process[axis + 3, axis + 3] = q * dt_s
    return transition @ state, transition @ covariance @ transition.T + process


def _update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    radar_m: np.ndarray,
    tracker: Tracker,
) -> tuple[np.ndarray, np.ndarray]:
    predicted, jacobian = _observe(state, radar_m)
    noise = np.diag(
        [
            tracker.range_sigma_m**2,
            tracker.angle_sigma_rad**2,
            tracker.angle_sigma_rad**2,
            tracker.radial_sigma_mps**2,
        ]
    )
    innovation = measurement - predicted
    innovation[1] = _wrap(innovation[1])
    residual = jacobian @ covariance @ jacobian.T + noise
    gain = covariance @ jacobian.T @ np.linalg.inv(residual)
    updated = state + gain @ innovation
    identity = np.eye(6)
    covariance = (identity - gain @ jacobian) @ covariance @ (identity - gain @ jacobian).T + gain @ noise @ gain.T
    return updated, covariance


def _observe(state: np.ndarray, radar_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = state[:3] - radar_m
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        distance = 1e-6
    horizontal = float(np.hypot(delta[0], delta[1]))
    if horizontal < 1e-6:
        horizontal = 1e-6
    unit = delta / distance
    azimuth = float(np.atan2(delta[1], delta[0]))
    elevation = float(np.atan2(delta[2], horizontal))
    radial = -float(np.dot(state[3:], unit))
    jacobian = np.zeros((4, 6))
    jacobian[0, :3] = unit
    jacobian[1, 0] = -delta[1] / horizontal**2
    jacobian[1, 1] = delta[0] / horizontal**2
    jacobian[2, 0] = -delta[2] * delta[0] / (distance**2 * horizontal)
    jacobian[2, 1] = -delta[2] * delta[1] / (distance**2 * horizontal)
    jacobian[2, 2] = horizontal / distance**2
    jacobian[3, :3] = state[3:] / distance - np.dot(state[3:], delta) * delta / distance**3
    jacobian[3, :3] *= -1.0
    jacobian[3, 3:] = -unit
    return np.array([distance, azimuth, elevation, radial]), jacobian


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _assign(tracks: list[Track], detections: list[dict], gate_m: float) -> list[tuple[int, int]]:
    if not tracks or not detections:
        return []
    cost = np.full((len(tracks), len(detections)), 1e6, dtype=np.float64)
    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            distance = float(
                np.hypot(track.state[0] - float(detection["x_m"]), track.state[1] - float(detection["y_m"]))
            )
            if distance <= gate_m:
                cost[i, j] = distance
    rows, cols = linear_sum_assignment(cost)
    return [(int(row), int(col)) for row, col in zip(rows, cols, strict=True) if cost[row, col] <= gate_m]
