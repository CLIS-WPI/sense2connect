"""Radar re-trace for the back-to-back panels (configs/tvt.yaml panels: method).

The paper-1 radar at oru-0 was tilted (look_at the street centre). The solver is polarized, so its trace cannot be
turned into the untilted TR 38.901 panels exactly (scripts/test_tvt_panels.py). This module re-traces the radar
ONCE with the paper-1 code path unchanged - build_radar (scene, sensing targets, isotropic 1-element TX and 8x8
RX, RCSSolver / PathSolver with the paper-1 settings, deterministic), move_targets, pack_paths, ground_truth -
except that the radar's TX and RX orientation is set to 0 (untilted, boresight +x). Both panels follow exactly
from this trace (sim/tvt/panels.radar_panel_paths). The communication paths are not traced (the TVT pipeline
uses the comm-geometry and paper-2 path caches).

Cache: results/TVT/radar/trace/<mount>/<density>/seed_<s>/{static.npz, frame_<i>.npz, meta.json}, the paper-1
field layout (rcs_*, bg_*, gt_*), stage-scoped provenance (sources below + the scenario config); a mismatch
is refused. The paper-1 caches in results/cache are never touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOT = ROOT / "results" / "TVT" / "radar" / "trace"
N_FRAMES = 600  # 60 s at 0.1 s, as the paper-1 sensing caches


def sources(scenario_config: Path) -> list[Path]:
    return [ROOT / n for n in ("sim/tvt/radar_trace.py", "sim/sensing/radar.py", "sim/sensing/cache.py", "sim/scenes/motion.py",
                               "sim/scenes/traffic.py", "sim/scenes/loop.py", "sim/scenes/config.py")] + [Path(scenario_config)]


def trace_dir(mount: str, density: str, seed: int) -> Path:
    return TRACE_ROOT / mount / density / f"seed_{seed}"


def fresh(mount: str, density: str, seed: int, scenario_config: Path) -> bool:
    from sim.sensing.provenance import CacheRefused, require_scoped

    m = trace_dir(mount, density, seed) / "meta.json"
    if not m.exists():
        return False
    try:
        meta = json.loads(m.read_text())
        require_scoped(meta, kind="tvt_radar_trace", sources=sources(scenario_config))
        return int(meta["n_frames"]) >= N_FRAMES  # a shortened (test) trace is never reused
    except (CacheRefused, KeyError):
        return False


def load_meta(mount: str, density: str, seed: int, scenario_config: Path) -> dict[str, Any]:
    from sim.sensing.provenance import require_scoped

    meta = json.loads((trace_dir(mount, density, seed) / "meta.json").read_text())
    require_scoped(meta, kind="tvt_radar_trace", sources=sources(scenario_config))
    return meta


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)


def load_frame(mount: str, density: str, seed: int, index: int) -> dict[str, np.ndarray]:
    with np.load(trace_dir(mount, density, seed) / f"frame_{index:04d}.npz", allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def load_static(mount: str, density: str, seed: int) -> dict[str, np.ndarray]:
    with np.load(trace_dir(mount, density, seed) / "static.npz", allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def untilted_radar(scenario: dict[str, Any]) -> dict[str, Any]:
    """paper-1 build_radar with the radar TX / RX orientation set to 0 in the live and the static scene."""
    from sim.sensing.radar import build_radar

    radar = build_radar(scenario)
    for scene in (radar["live"], radar["static"]):
        for name in ("radar-tx", "radar-rx"):
            scene.get(name).orientation = [0.0, 0.0, 0.0]
    radar["orientation_rad"] = np.zeros(3)
    return radar


def trace(scenario: dict[str, Any], mount: str, density: str, scenario_config: Path, n_frames: int = N_FRAMES) -> dict[str, Any]:
    """Trace one job (untilted isotropic radar) and write the cache; returns timing."""
    from sim.scenes.motion import move_targets, states_at
    from sim.sensing.cache import labels_from_scene, pack_paths
    from sim.sensing.provenance import scoped_version
    from sim.sensing.radar import _radar_kwargs, ground_truth

    seed = int(scenario["seed"])
    d = trace_dir(mount, density, seed)
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    if meta_p.exists():
        meta_p.unlink()  # an interrupted / stale job is rewritten completely
    clock = time.perf_counter()
    radar = untilted_radar(scenario)
    spec = scenario["sensing_radar"]
    static = pack_paths(radar["background"](radar["static"], **_radar_kwargs(scenario, "background")), labels_from_scene(radar["static"], {}))
    _save(d / "static.npz", static)
    actors = list(scenario["vehicles"]) + list(scenario["pedestrians"])
    pos0 = {a["name"]: states_at(scenario, 0.0)[a["name"]]["position_m"] for a in actors}
    labels = labels_from_scene(radar["live"], pos0)
    n = min(int(n_frames), int(scenario["n_snapshots"]))
    for index in range(n):
        t_s = index * float(scenario["dt_s"])
        states = states_at(scenario, t_s)
        move_targets(radar["live"], radar["targets"], states)
        rcs = pack_paths(radar["rcs"](radar["live"], **_radar_kwargs(scenario, "rcs")), labels)
        bg = pack_paths(radar["background"](radar["live"], **_radar_kwargs(scenario, "background")), labels)
        truth = ground_truth(scenario, t_s, radar["radar_position_m"], float(spec["sensing_range_m"]))
        payload = {f"rcs_{k}": v for k, v in rcs.items()} | {f"bg_{k}": v for k, v in bg.items()}
        payload |= {"gt_x": np.array([r["x_m"] for r in truth]), "gt_y": np.array([r["y_m"] for r in truth]),
                    "gt_z": np.array([r["z_m"] for r in truth]), "gt_vx": np.array([r["vx_mps"] for r in truth]),
                    "gt_vy": np.array([r["vy_mps"] for r in truth]), "gt_id": np.array([str(r["id"]) for r in truth]),
                    "gt_kind": np.array([str(r["kind"]) for r in truth])}
        _save(d / f"frame_{index:04d}.npz", payload)
    wl = 299_792_458.0 / float(scenario["carrier_hz"])
    meta = {"seed": seed, "mount": mount, "density": density, "dt_s": float(scenario["dt_s"]), "n_frames": n,
            "radar_position_m": [float(v) for v in radar["radar_position_m"]], "positions_m": np.asarray(radar["positions_m"]).tolist(),
            "orientation_rad": [0.0, 0.0, 0.0], "wavelength_m": wl, "scenario_config": str(Path(scenario_config).relative_to(ROOT)),
            "trace_s": time.perf_counter() - clock, "provenance": {"kind": "tvt_radar_trace", **scoped_version(sources(scenario_config))}}
    meta_p.write_text(json.dumps(meta) + "\n")
    return meta
