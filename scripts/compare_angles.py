"""Compare batched angles with the per-hit search and the Sionna spectrum."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml
from sim.scenes.traffic import prepare_scenario
from sim.sensing.cache import cache_dir, load_frame, load_static, split_frame
from sim.sensing.cfar import ca_cfar, local_maxima
from sim.sensing.channel import localize, localize_cells, n_lags, power_map, spectrum_peaks
from sim.sensing.process_torch import cpi_from_paths, delay_doppler_batch, frequency_responses
from sim.sensing.radar import build_radar
from sim.sensing.waveform import waveform_from_config


def main() -> None:
    scenario = prepare_scenario(
        load_yaml(ROOT / "configs" / "m2_scenario.yaml"),
        seed=101,
        mount="lamppost",
        density="low",
        duration_s=60.0,
        dt_s=0.1,
    )
    waveform = waveform_from_config(scenario)
    spec = scenario["sensing_radar"]
    frame = load_frame(cache_dir(ROOT / "results" / "cache", "lamppost", "low", 101), 20)
    static = load_static(cache_dir(ROOT / "results" / "cache", "lamppost", "low", 101))
    radar = build_radar(scenario)
    lags = n_lags(waveform, float(spec["max_range_m"]))
    power_dbm = float(spec["noise"]["tx_power_dbm"])
    rcs_c, rcs_d = cpi_from_paths(split_frame(frame, "rcs"), waveform, power_dbm, max_paths=256)
    bg_c, bg_d = cpi_from_paths(split_frame(frame, "bg"), waveform, power_dbm, max_paths=256)
    st_c, st_d = cpi_from_paths(static, waveform, power_dbm, max_paths=256)
    measured = frequency_responses(rcs_c, rcs_d, waveform) + frequency_responses(bg_c, bg_d, waveform)
    measured = measured - frequency_responses(st_c, st_d, waveform)
    cube = delay_doppler_batch(measured, lags, waveform.window)[0]
    power = power_map(cube)
    mask, _alpha = ca_cfar(power, guard=2, train=4, pfa=1e-3, noise_applied=True)
    hits = local_maxima(mask, power)[:32]
    if not hits:
        raise SystemExit("no detections on the cached frame")
    doppler = torch.tensor([item[0] for item in hits], device=cube.device)
    lag = torch.tensor([item[1] for item in hits], device=cube.device)
    started = time.perf_counter()
    per_hit = [
        localize(cube, int(d), int(g), waveform, radar["positions_m"], radar["orientation_rad"], radar["radar_position_m"])
        for d, g, _peak in hits
    ]
    per_hit_s = time.perf_counter() - started
    started = time.perf_counter()
    batch_x, batch_y, _batch_z, batch_t, batch_p = localize_cells(
        cube, doppler, lag, waveform, radar["positions_m"], radar["orientation_rad"], radar["radar_position_m"]
    )
    batch_s = time.perf_counter() - started
    spec_t, spec_p = spectrum_peaks(cube, doppler, lag, waveform, radar["positions_m"])
    hit_t = []
    hit_p = []
    for d, g, _peak in hits:
        spatial = cube[:, int(d), int(g)].detach().cpu().numpy()
        from sim.sensing.channel import _peak_direction

        theta, phi = _peak_direction(spatial, radar["positions_m"], waveform.wavelength_m)
        hit_t.append(theta)
        hit_p.append(phi)
    hit_t = np.asarray(hit_t)
    hit_p = np.asarray(hit_p)
    print(f"hits {len(hits)} per_hit_s {per_hit_s:.3f} batch_s {batch_s:.3f}")
    print("batch vs per-hit theta deg", _deg(batch_t, hit_t), "phi deg", _deg(batch_p, hit_p))
    print("batch vs per-hit xy m", float(np.max(np.hypot(batch_x - np.array([p[0] for p in per_hit]), batch_y - np.array([p[1] for p in per_hit])))))
    print("spectrum vs per-hit theta deg", _deg(spec_t, hit_t), "phi deg", _deg(spec_p, hit_p))
    _fft_report(cube, doppler, lag, radar["positions_m"], waveform.wavelength_m, hit_t, hit_p)


def _deg(left: np.ndarray, right: np.ndarray) -> str:
    err = np.abs(np.asarray(left) - np.asarray(right))
    return f"max {np.rad2deg(err.max()):.3f} median {np.rad2deg(np.median(err)):.3f}"


def _fft_report(cube, doppler, lag, positions_m, wavelength_m, hit_t, hit_p) -> None:
    positions = np.asarray(positions_m)
    if positions.shape != (64, 3):
        print("fft skipped, antenna count", positions.shape)
        return
    span_y = np.unique(np.round(positions[:, 1], 5))
    span_z = np.unique(np.round(positions[:, 2], 5))
    if len(span_y) != 8 or len(span_z) != 8:
        print("fft skipped, grid", len(span_y), len(span_z))
        return
    grid = np.zeros((8, 8), dtype=np.complex64)
    order = np.lexsort((positions[:, 2], positions[:, 1]))
    spatial = cube[:, doppler, lag].detach().cpu().numpy().T
    errors = []
    for index in range(spatial.shape[0]):
        grid.reshape(-1)[np.argsort(order)] = spatial[index]
        spectrum = np.fft.fftshift(np.fft.fft2(grid.reshape(-1)[np.argsort(order)].reshape(8, 8)))
        peak = np.unravel_index(int(np.argmax(np.abs(spectrum) ** 2)), spectrum.shape)
        # Bin centers are spatial frequencies, not the search angles. Report the bin only.
        errors.append(peak)
    print("fft peaks (row, col)", errors[:4], "n", len(errors))


if __name__ == "__main__":
    main()
