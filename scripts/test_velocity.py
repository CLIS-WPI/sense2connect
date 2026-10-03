"""Velocity sanity: Doppler sign/scale, EKF measurement, extended target.

Empty scene, one constant-velocity target. Approaching, receding, and
crossing. With and without noise. Point target and a 5-point vehicle.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.sensing.channel import (
    add_thermal_noise,
    delay_doppler,
    doppler_hz,
    frequency_response,
    n_lags,
    power_map,
    radial_velocity_mps,
    range_m,
)
from sim.sensing.track import Tracker, _observe
from sim.sensing.waveform import Waveform, waveform_from_config
from sim.scenes.config import load_yaml

C = 299792458.0
CARRIER_HZ = 2.8e10


def _wave() -> Waveform:
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    return waveform_from_config(raw)


def _true_receding(position_m: np.ndarray, velocity_mps: np.ndarray, radar_m: np.ndarray) -> float:
    delta = position_m - radar_m
    distance = float(np.linalg.norm(delta))
    return float(np.dot(velocity_mps, delta / distance))


def _true_approaching(position_m: np.ndarray, velocity_mps: np.ndarray, radar_m: np.ndarray) -> float:
    return -_true_receding(position_m, velocity_mps, radar_m)


def test_observe_jacobian() -> list[str]:
    """Finite-difference Jacobian of the EKF measurement model."""
    lines = ["## EKF measurement model", ""]
    radar = np.array([12.0, 6.5, 5.0])
    state = np.array([20.0, -2.5, 1.5, 15.0, 0.0, 0.0], dtype=np.float64)
    predicted, jacobian = _observe(state, radar)
    unit = (state[:3] - radar) / float(np.linalg.norm(state[:3] - radar))
    approaching = -float(np.dot(state[3:], unit))
    ok_sign = abs(predicted[3] - approaching) < 1e-9
    lines.append(
        f"Predicted approaching radial {predicted[3]:.4f} m/s vs -v·r_hat {approaching:.4f} m/s: "
        f"{'PASS' if ok_sign else 'FAIL'}"
    )
    numeric = np.zeros((4, 6))
    step = 1e-5
    for col in range(6):
        plus = state.copy()
        minus = state.copy()
        plus[col] += step
        minus[col] -= step
        numeric[:, col] = (_observe(plus, radar)[0] - _observe(minus, radar)[0]) / (2.0 * step)
    numeric[1, :] = [((d + math.pi) % (2.0 * math.pi) - math.pi) / (2.0 * step) * 2.0 * step for d in numeric[1, :]]
    err = float(np.max(np.abs(jacobian - numeric)))
    ok_j = err < 2e-4
    lines.append(f"Jacobian max abs error vs finite difference {err:.2e}: {'PASS' if ok_j else 'FAIL'}")
    lines.append(f"H[radial, velocity] = {-unit}  (must be -r_hat for approaching measurement)")
    birth = Tracker(dt_s=0.1, association_gate_m=8.0, coast_frames=4, radar_position_m=radar)
    detection = {
        "x_m": float(state[0]),
        "y_m": float(state[1]),
        "z_m": float(state[2]),
        "radial_velocity_mps": approaching,
    }
    birth.step([detection])
    born = birth.tracks[0].state[3:]
    expected_v = -approaching * unit
    birth_err = float(np.linalg.norm(born - expected_v))
    lines.append(
        f"Birth velocity {born} vs -v_app r_hat {expected_v}: err {birth_err:.3e} "
        f"{'PASS' if birth_err < 1e-9 else 'FAIL'}"
    )
    if not (ok_sign and ok_j and birth_err < 1e-9):
        raise SystemExit("EKF measurement model failed")
    return lines


def _monostatic(frequency_hz: float):
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene()
    scene.frequency = frequency_hz
    scene.add(Transmitter("tx", position=[0.0, 0.0, 0.0]))
    scene.add(Receiver("rx", position=[0.0, 0.0, 0.0]))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization="V", pattern="iso")
    scene.rx_array = PlanarArray(num_rows=8, num_cols=8, polarization="V", pattern="iso")
    return scene


def _peak_radial(paths, waveform: Waveform, noise: bool, seed: int) -> tuple[float, float, float]:
    response = frequency_response(paths, waveform, 30.0)
    if noise:
        generator = __import__("torch").Generator(device=response.device)
        generator.manual_seed(seed)
        response = add_thermal_noise(response, waveform, 7.0, 290.0, generator)
    lags = n_lags(waveform, 80.0)
    if waveform.window:
        from sim.sensing.channel import apply_window

        response = apply_window(response)
    cube = delay_doppler(response, lags)
    power = power_map(cube)
    # Ignore the DC Doppler bin; a noiseless static residual can win there.
    masked = power.copy()
    dc = waveform.cpi_slots // 2
    masked[dc, :] = 0.0
    peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
    doppler_index = int(peak[0])
    lag = int(peak[1])
    return radial_velocity_mps(doppler_index, waveform), doppler_hz(doppler_index, waveform), range_m(lag, waveform)


def _solve_point(position, velocity):
    from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver

    scene = _monostatic(CARRIER_HZ)
    scene.add(
        ConstantRCSSensingTarget(
            "st",
            sigma=10.0,
            position=[float(x) for x in position],
            velocity=[float(x) for x in velocity],
        )
    )
    return RCSSolver(deterministic=True)(scene, max_depth=1, los=True, seed=1), scene


def _solve_extended(position, velocity):
    from sionna.rt.rcs import RCSSolver, TR38901SensingTarget

    scene = _monostatic(CARRIER_HZ)
    scene.add(
        TR38901SensingTarget(
            "bus",
            object_type="vehicle-multi-sp",
            length=12.0,
            width=2.5,
            height=3.0,
            position=[float(x) for x in position],
            velocity=[float(x) for x in velocity],
            random_sigma_s=False,
            random_phases=False,
            random_xpr=False,
        )
    )
    return RCSSolver(deterministic=True)(scene, max_depth=1, los=True, seed=1), scene


def test_doppler_and_map() -> list[str]:
    waveform = _wave()
    radar = np.zeros(3)
    cases = [
        ("approaching", np.array([30.0, 0.0, 0.0]), np.array([-12.0, 0.0, 0.0])),
        ("receding", np.array([30.0, 0.0, 0.0]), np.array([12.0, 0.0, 0.0])),
        ("crossing", np.array([0.0, 25.0, 0.0]), np.array([15.0, 0.0, 0.0])),
    ]
    lines = [
        "## Doppler sign and scale (empty scene, ConstantRCS)",
        "",
        "Sionna: f_D = -2 v_receding / lambda, v_receding > 0 leaving the radar. "
        "Positive Doppler is approaching. Conversion: v_app = f_D * lambda / 2.",
        "",
        "| Geometry | Noise | True v_app | Path v_app | Map v_app | Path f_D | Formula f_D | Range |",
        "|---|---|---|---|---|---|---|---|",
    ]
    failed = False
    for name, position, velocity in cases:
        paths, scene = _solve_point(position, velocity)
        lam = float(np.asarray(scene.wavelength.numpy()).reshape(-1)[0])
        v_rec = _true_receding(position, velocity, radar)
        v_app = -v_rec
        fd_exp = -2.0 * v_rec / lam
        fd_path = float(np.squeeze(np.asarray(paths.doppler.numpy())))
        path_app = fd_path * lam / 2.0
        for noise in (False, True):
            map_app, map_fd, rng = _peak_radial(paths, waveform, noise, seed=11)
            path_ok = abs(fd_path - fd_exp) / max(abs(fd_exp), 1.0) < 1e-3
            # Crossing true radial is ~0; allow one Doppler bin.
            bin_tol = waveform.v_res_mps * 2.5
            map_ok = abs(map_app - v_app) < max(1.0, bin_tol if abs(v_app) < 1.0 else 1.5)
            if not path_ok or not map_ok:
                failed = True
            lines.append(
                f"| {name} | {noise} | {v_app:.3f} | {path_app:.3f} | {map_app:.3f} | "
                f"{fd_path:.1f} | {fd_exp:.1f} | {rng:.1f} |"
            )
            print(
                f"point {name} noise={noise} true={v_app:.3f} path={path_app:.3f} "
                f"map={map_app:.3f} fd {fd_path:.1f}/{fd_exp:.1f} range {rng:.1f}",
                flush=True,
            )
    lines.extend(["", "## Extended target (5 scattering points, vehicle-multi-sp)", ""])
    lines.append("| Geometry | Noise | Center v_app | Map v_app | Path v_app (max |a|) |")
    lines.append("|---|---|---|---|---|")
    for name, position, velocity in cases:
        paths, scene = _solve_extended(position, velocity)
        lam = float(np.asarray(scene.wavelength.numpy()).reshape(-1)[0])
        v_app = _true_approaching(position, velocity, radar)
        doppler = np.asarray(paths.doppler.numpy()).reshape(-1)
        amp, _tau = paths.cir(out_type="numpy", normalize_delays=False)
        power = np.abs(np.squeeze(amp)) ** 2
        if power.ndim == 0:
            path_app = float(doppler.reshape(-1)[0]) * lam / 2.0
        else:
            strongest = int(np.argmax(np.asarray(power).reshape(-1)))
            path_app = float(np.asarray(doppler).reshape(-1)[strongest]) * lam / 2.0
        for noise in (False, True):
            map_app, _fd, _rng = _peak_radial(paths, waveform, noise, seed=17)
            # Extended: strongest scatterer radial can differ from the mesh origin.
            lines.append(f"| {name} | {noise} | {v_app:.3f} | {map_app:.3f} | {path_app:.3f} |")
            print(
                f"ext {name} noise={noise} center={v_app:.3f} map={map_app:.3f} strongest={path_app:.3f}",
                flush=True,
            )
    if failed:
        raise SystemExit("Doppler sign or scale failed on the point target")
    return lines


def test_ekf_tracks_constant_velocity() -> list[str]:
    radar = np.zeros(3)
    lines = ["", "## EKF on noiseless point detections", "", "| Geometry | True v_app | Track v_app after 1 s | Cartesian vel RMSE |", "|---|---|---|---|"]
    geometries = [
        ("approaching", np.array([30.0, 0.0, 0.0]), np.array([-12.0, 0.0, 0.0])),
        ("receding", np.array([30.0, 0.0, 0.0]), np.array([12.0, 0.0, 0.0])),
        ("crossing", np.array([0.0, 25.0, 0.0]), np.array([15.0, 0.0, 0.0])),
    ]
    failed = False
    for name, p0, vel in geometries:
        tracker = Tracker(dt_s=0.1, association_gate_m=8.0, coast_frames=4, radar_position_m=radar, process_q=0.25)
        for k in range(12):
            position = p0 + vel * (k * 0.1)
            v_app = _true_approaching(position, vel, radar)
            tracker.step(
                [
                    {
                        "x_m": float(position[0]),
                        "y_m": float(position[1]),
                        "z_m": float(position[2]),
                        "radial_velocity_mps": v_app,
                    }
                ]
            )
        track = tracker.tracks[0]
        unit = (track.state[:3] - radar) / float(np.linalg.norm(track.state[:3] - radar))
        est_app = -float(np.dot(track.state[3:], unit))
        true_app = _true_approaching(p0 + vel * 1.1, vel, radar)
        rmse = float(np.linalg.norm(track.state[3:] - vel))
        ok = abs(est_app - true_app) < 0.2 and rmse < 0.5
        if not ok:
            failed = True
        lines.append(f"| {name} | {true_app:.3f} | {est_app:.3f} | {rmse:.3f} |")
        print(f"ekf {name} true_app={true_app:.3f} est_app={est_app:.3f} vel_rmse={rmse:.3f}", flush=True)
    if failed:
        raise SystemExit("EKF failed to recover constant-velocity radial speed")
    return lines


def main() -> None:
    lines = [
        "# M2 velocity sanity",
        "",
        "Empty scene. One constant-velocity target. Radar at the origin.",
        "",
    ]
    lines.extend(test_observe_jacobian())
    lines.append("")
    lines.extend(test_doppler_and_map())
    lines.extend(test_ekf_tracks_constant_velocity())
    text = "\n".join(lines) + "\n"
    out = ROOT / "results" / "M2" / "velocity_sanity.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
