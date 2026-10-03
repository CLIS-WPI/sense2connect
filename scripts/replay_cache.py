"""Time radar processing from the channel cache, with no ray tracing."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml
from sim.scenes.traffic import prepare_scenario
from sim.sensing.cache import cache_dir, load_frame, load_static, split_frame
from sim.sensing.channel import n_lags
from sim.sensing.process_torch import (
    add_noise_batch,
    ca_cfar_torch,
    cpi_from_paths,
    delay_doppler_batch,
    frequency_responses,
    power_maps,
)
from sim.sensing.waveform import waveform_from_config


def main() -> None:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    directory = cache_dir(ROOT / "results" / "cache", "lamppost", "low", 101)
    scenario = prepare_scenario(
        load_yaml(ROOT / "configs" / "m2_scenario.yaml"),
        seed=101,
        mount="lamppost",
        density="low",
        duration_s=0.1 * frames,
        dt_s=0.1,
    )
    waveform = waveform_from_config(scenario)
    spec = scenario["sensing_radar"]
    noise = spec["noise"]
    lags = n_lags(waveform, float(spec["max_range_m"]))
    max_paths = 256
    started = time.perf_counter()
    static_paths = load_static(directory)
    io_s = time.perf_counter() - started
    ofdm_s = 0.0
    cfar_s = 0.0
    index = 0
    torch.cuda.reset_peak_memory_stats()
    while index < frames:
        count = min(batch, frames - index)
        started = time.perf_counter()
        rows = [load_frame(directory, index + offset) for offset in range(count)]
        io_s += time.perf_counter() - started
        started = time.perf_counter()
        rcs_c, rcs_d, bg_c, bg_d = [], [], [], []
        for frame in rows:
            coeff, delay = cpi_from_paths(split_frame(frame, "rcs"), waveform, float(noise["tx_power_dbm"]), max_paths=max_paths)
            rcs_c.append(coeff)
            rcs_d.append(delay)
            coeff, delay = cpi_from_paths(split_frame(frame, "bg"), waveform, float(noise["tx_power_dbm"]), max_paths=max_paths)
            bg_c.append(coeff)
            bg_d.append(delay)
        st_c, st_d = cpi_from_paths(static_paths, waveform, float(noise["tx_power_dbm"]), max_paths=max_paths)
        measured = frequency_responses(torch.cat(rcs_c, dim=0), torch.cat(rcs_d, dim=0), waveform)
        measured = measured + frequency_responses(torch.cat(bg_c, dim=0), torch.cat(bg_d, dim=0), waveform)
        seeds = [
            int(scenario["seed"]) * 100003 + index + offset + waveform.n_subcarriers
            for offset in range(count)
        ]
        measured = add_noise_batch(
            measured,
            waveform,
            float(noise["noise_figure_db"]),
            float(noise["temperature_k"]),
            seeds,
        )
        twin = measured - frequency_responses(st_c.expand(count, -1, -1, -1, -1, -1, -1), st_d.expand(count, -1, -1, -1), waveform)
        cubes = delay_doppler_batch(twin, lags, waveform.window)
        torch.cuda.synchronize()
        ofdm_s += time.perf_counter() - started
        started = time.perf_counter()
        power = power_maps(cubes)
        ca_cfar_torch(power, guard=int(spec["cfar"]["guard"]), train=4, pfa=1e-4, noise_applied=True)
        torch.cuda.synchronize()
        cfar_s += time.perf_counter() - started
        index += count
    summary = {
        "n_frames": frames,
        "batch": batch,
        "max_paths": max_paths,
        "io_s": io_s,
        "ofdm_fft_s": ofdm_s,
        "cfar_s": cfar_s,
        "seconds_per_snapshot": (io_s + ofdm_s + cfar_s) / frames,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }
    out = ROOT / "results" / "M2" / "profile_after.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
