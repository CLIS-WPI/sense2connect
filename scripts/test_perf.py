"""Equality tests for the GPU path and the channel cache."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sim.comm.blockage import screen_blockage
from sim.comm.blockage_torch import blocker_path_loss_db, screen_loss_db
from sim.scenes.traffic import prepare_scenario
from sim.sensing.cache import load_frame, split_frame
from sim.sensing.cfar import ca_cfar
from sim.sensing.channel import frequency_response
from sim.sensing.process_torch import ca_cfar_torch, cpi_from_paths, frequency_responses
from sim.sensing.radar import _radar_kwargs, build_radar
from sim.sensing.waveform import waveform_from_config
from sim.sensing.cache import labels_from_scene, pack_paths
from sim.scenes.motion import move_targets, states_at
from sim.sensing.trace import direct_frame_paths, trace_job
from sim.scenes.config import load_yaml

_ATOL = 1e-5


def test_model_b_matches_numpy() -> None:
    rng = np.random.default_rng(7)
    n = 64
    tx = rng.normal(size=(n, 3)) * 20.0
    rx = rng.normal(size=(n, 3)) * 20.0
    center = rng.normal(size=(n, 3))
    length = rng.uniform(0.5, 12.0, size=n)
    width = rng.uniform(0.5, 3.5, size=n)
    height = rng.uniform(1.5, 3.5, size=n)
    wavelength = 0.0107
    reference = np.array(
        [
            screen_blockage(tx[i], rx[i], center[i], float(length[i]), float(width[i]), float(height[i]), wavelength).loss_db
            for i in range(n)
        ]
    )
    got = screen_loss_db(
        torch.as_tensor(tx, device="cuda"),
        torch.as_tensor(rx, device="cuda"),
        torch.as_tensor(center, device="cuda"),
        torch.as_tensor(length, device="cuda"),
        torch.as_tensor(width, device="cuda"),
        torch.as_tensor(height, device="cuda"),
        wavelength,
    ).detach().cpu().numpy()
    _close(reference, got, "screen loss")
    starts = rng.normal(size=(5, 3, 3)) * 10.0
    ends = rng.normal(size=(5, 3, 3)) * 10.0
    valid = np.ones((5, 3), dtype=bool)
    valid[:, -1] = False
    centers = rng.normal(size=(4, 3))
    batched = blocker_path_loss_db(
        torch.as_tensor(starts, device="cuda"),
        torch.as_tensor(ends, device="cuda"),
        torch.as_tensor(centers, device="cuda"),
        torch.as_tensor(np.full(4, 4.0), device="cuda"),
        torch.as_tensor(np.full(4, 2.0), device="cuda"),
        torch.as_tensor(np.full(4, 1.7), device="cuda"),
        wavelength,
        torch.as_tensor(valid, device="cuda"),
    ).detach().cpu().numpy()
    manual = np.zeros((4, 5))
    for blocker in range(4):
        for path in range(5):
            total = 0.0
            for seg in range(3):
                if not valid[path, seg]:
                    continue
                loss = screen_blockage(
                    starts[path, seg], ends[path, seg], centers[blocker], 4.0, 2.0, 1.7, wavelength
                ).loss_db
                if not np.isfinite(loss):
                    total = np.inf
                    break
                total += loss
            manual[blocker, path] = total
    _close(manual, batched, "batched model B")


def test_cfar_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    power = rng.random((2, 32, 40))
    power[0, 10, 12] = 50.0
    reference = []
    for index in range(2):
        mask, alpha = ca_cfar(power[index], guard=2, train=4, pfa=1e-3, noise_applied=True)
        reference.append(mask)
    masks, alpha_t = ca_cfar_torch(
        torch.as_tensor(power, device="cuda"),
        guard=2,
        train=4,
        pfa=1e-3,
        noise_applied=True,
    )
    got = masks.detach().cpu().numpy()
    if not np.array_equal(np.stack(reference), got):
        raise AssertionError("torch CFAR mask differs from the NumPy reference")
    if abs(alpha_t - alpha) > 0.0:
        raise AssertionError("CFAR alpha differs")


def test_cached_cpi_matches_solver() -> None:
    scenario = _scenario(101, "lamppost", "low", n_frames=1)
    waveform = waveform_from_config(scenario)
    radar = build_radar(scenario)
    states = states_at(scenario, 0.0)
    move_targets(radar["live"], radar["targets"], states)
    paths = radar["rcs"](radar["live"], **_radar_kwargs(scenario, "rcs"))
    direct = frequency_response(paths, waveform, 30.0).detach().cpu().numpy()
    positions = {
        actor["name"]: states[actor["name"]]["position_m"]
        for actor in list(scenario["vehicles"]) + list(scenario["pedestrians"])
    }
    packed = pack_paths(paths, labels_from_scene(radar["live"], positions))
    n_paths = int(np.asarray(packed["a"]).shape[-2])
    coefficients, delays = cpi_from_paths(packed, waveform, 30.0, max_paths=n_paths)
    rebuilt = frequency_responses(coefficients, delays, waveform)[0].detach().cpu().numpy()
    _close(direct, rebuilt, "cached CPI")


def test_cache_matches_direct_solve() -> None:
    scenario = _scenario(101, "lamppost", "low", n_frames=1)
    directory = Path(tempfile.mkdtemp(prefix="cache-check-"))
    try:
        trace_job(
            scenario,
            mount="lamppost",
            density="low",
            cache_root=directory,
            max_frames=1,
            process=False,
        )
        cached = load_frame(directory / "lamppost" / "low" / "seed_101", 0)
        direct_rcs, direct_bg = direct_frame_paths(scenario, 0)
        for prefix, direct in (("rcs", direct_rcs), ("bg", direct_bg)):
            saved = split_frame(cached, prefix)
            for name in saved:
                if name == "object_names":
                    continue
                left = np.asarray(saved[name])
                right = np.asarray(direct[name])
                if left.shape != right.shape or not np.array_equal(left, right):
                    if left.dtype.kind in "fc":
                        difference = float(np.max(np.abs(left - right)))
                    else:
                        difference = int(np.sum(left != right))
                    raise AssertionError(f"{prefix}.{name} cache differs from a direct solve ({difference})")
    finally:
        shutil.rmtree(directory)


def _scenario(seed: int, mount: str, density: str, n_frames: int) -> dict:
    return prepare_scenario(
        load_yaml(ROOT / "configs" / "m2_scenario.yaml"),
        seed=seed,
        mount=mount,
        density=density,
        duration_s=0.1 * n_frames,
        dt_s=0.1,
    )


def _close(reference: np.ndarray, got: np.ndarray, label: str) -> None:
    reference = np.asarray(reference)
    got = np.asarray(got)
    if np.iscomplexobj(reference) or np.iscomplexobj(got):
        error = float(np.max(np.abs(reference - got)))
        if error > _ATOL:
            raise AssertionError(f"{label} max abs error {error}")
        return
    reference = reference.astype(np.float64)
    got = got.astype(np.float64)
    both_inf = np.isinf(reference) & np.isinf(got)
    if np.any(np.sign(reference[both_inf]) != np.sign(got[both_inf])):
        raise AssertionError(f"{label} infinite sign differs")
    finite = np.isfinite(reference) & np.isfinite(got)
    if not np.allclose(reference[finite], got[finite], rtol=_ATOL, atol=_ATOL):
        error = np.max(np.abs(reference[finite] - got[finite]))
        raise AssertionError(f"{label} max abs error {error}")


def main() -> None:
    test_model_b_matches_numpy()
    test_cfar_matches_numpy()
    test_cached_cpi_matches_solver()
    test_cache_matches_direct_solve()
    print("perf tests passed")


if __name__ == "__main__":
    main()
