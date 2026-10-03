"""Map-constrained tracker. The detector is unchanged.

A vehicle birth within ``lane_gate_m`` of a lane centre line uses an
along-lane constant-velocity model: the cross-lane coordinate is the line
and the speed keeps the lane direction. A pedestrian birth within
``sidewalk_gate_m`` of a sidewalk centre does the same, except the
along-sidewalk sign is free. When a later measurement leaves that sidewalk
by ``leave_m``, the track switches to the free 2D model and stays there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.sensing.track import Track, Tracker, _apply_two_point, _assign, _measurement, _update


@dataclass
class MapTrack(Track):
    """A track plus the map mode that constrains it."""

    mode: str = "free"
    line_y_m: float | None = None
    direction_sign: float = 1.0


@dataclass
class MapTracker(Tracker):
    """Same EKF measurements as ``Tracker``, with digital-twin motion models."""

    lanes: list[dict] = field(default_factory=list)
    sidewalks: list[dict] = field(default_factory=list)
    lane_gate_m: float = 2.0
    sidewalk_gate_m: float = 2.0
    leave_m: float = 2.5
    cross_q: float = 1e-3

    def step(self, detections: list[dict]) -> list[MapTrack]:
        """Predict, assign, and return the tracks that are still alive."""
        dt = float(self.dt_s)
        if self._started:
            self._time_s += dt
        self._started = True
        radar = np.asarray(self.radar_position_m, dtype=np.float64)
        for track in self.tracks:
            track.state, track.covariance = _predict_mode(track, dt, self.process_q, self.cross_q)
            _pin(track)
            track.age_s = self._time_s - track.birth_s
        pairs = _assign(self.tracks, detections, self.association_gate_m)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for track_index, det_index in pairs:
            track = self.tracks[track_index]
            measurement, position = _measurement(detections[det_index], radar)
            if track.mode == "sidewalk" and track.line_y_m is not None:
                if abs(float(position[1]) - float(track.line_y_m)) > float(self.leave_m):
                    track.mode = "free"
                    track.line_y_m = None
            if track.hits == 1:
                _apply_two_point(track, position, measurement, radar, self._time_s)
                _pin(track)
            track.state, track.covariance = _update(track.state, track.covariance, measurement, radar, self)
            _pin(track)
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
            mode, line_y, sign = _classify(float(position[1]), self)
            state = np.concatenate([position, velocity])
            track = MapTrack(
                identifier=self._next_id,
                state=state,
                covariance=np.diag(
                    [
                        self.range_sigma_m**2,
                        self.range_sigma_m**2,
                        self.range_sigma_m**2,
                        4.0,
                        4.0,
                        4.0,
                    ]
                ),
                hits=1,
                confirmed=self.confirm_hits <= 1,
                birth_s=self._time_s,
                age_s=0.0,
                measured_m=position.copy(),
                measured_s=self._time_s,
                mode=mode,
                line_y_m=line_y,
                direction_sign=sign,
            )
            _pin(track)
            self.tracks.append(track)
            self._next_id += 1
        self.tracks = [track for track in self.tracks if track.misses <= self.coast_frames]
        return list(self.tracks)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _classify(y_m: float, tracker: MapTracker) -> tuple[str, float | None, float]:
    lane = _nearest(y_m, tracker.lanes)
    if lane is not None and abs(y_m - lane[0]) <= float(tracker.lane_gate_m):
        return "lane", lane[0], lane[1]
    walk = _nearest(y_m, tracker.sidewalks)
    if walk is not None and abs(y_m - walk[0]) <= float(tracker.sidewalk_gate_m):
        return "sidewalk", walk[0], 1.0
    return "free", None, 1.0


def _nearest(y_m: float, lines: list[dict]) -> tuple[float, float] | None:
    best = None
    best_distance = None
    for line in lines:
        center = float(line["origin_m"][1])
        sign = 1.0 if float(line["direction"][0]) >= 0.0 else -1.0
        distance = abs(y_m - center)
        if best_distance is None or distance < best_distance:
            best = (center, sign)
            best_distance = distance
    return best


def _pin(track: MapTrack) -> None:
    """Hard constraint for lane and sidewalk modes. Free tracks are unchanged."""
    if track.mode == "free" or track.line_y_m is None:
        return
    track.state[1] = float(track.line_y_m)
    track.state[4] = 0.0
    if track.mode == "lane":
        track.state[3] = float(track.direction_sign) * abs(float(track.state[3]))
    for index in (1, 4):
        track.covariance[index, :] = 0.0
        track.covariance[:, index] = 0.0
        track.covariance[index, index] = 1e-4


def _predict_mode(track: MapTrack, dt_s: float, along_q: float, cross_q: float) -> tuple[np.ndarray, np.ndarray]:
    transition = np.eye(6)
    transition[0, 3] = dt_s
    transition[1, 4] = dt_s
    transition[2, 5] = dt_s
    process = np.zeros((6, 6))
    if track.mode == "free":
        axes = (along_q, along_q, along_q)
    else:
        axes = (along_q, cross_q, along_q)
    for axis, q in enumerate(axes):
        process[axis, axis] = q * dt_s**3 / 3.0
        process[axis, axis + 3] = q * dt_s**2 / 2.0
        process[axis + 3, axis] = q * dt_s**2 / 2.0
        process[axis + 3, axis + 3] = q * dt_s
    return transition @ track.state, transition @ track.covariance @ transition.T + process
