"""TVT T0: run an unchanged paper-1 / paper-2 pipeline script under a chosen service model.

The frozen pipelines read the cell powers from run_m3.build() (cached 10 ms timelines,
power-sum model). This wrapper imports run_m3, replaces build() by a version that
overwrites the four power fields (unblocked_power, blocked_power, los_blocked_power,
best_alt_power) with those of the chosen model (sim/tvt/service.py), then calls the
target script's main(). ``power_sum`` leaves the frozen fields untouched (the
reference); ``best_beam`` / ``mrt`` are computed from the same paths, model-B losses
and timeline, and cached per job under results/TVT/cache/service/ with stage-scoped
provenance (refused on mismatch). The frozen caches are read only.

Output isolation: the paper-1 scripts write to results/M3, results/M5 and the paper-2
closing to results/P2/closing_<set>.json. Run them in the container with those paths
bind-mounted to a sandbox (scripts/tvt_t0_service.sh), never on the real results.

Run: python scripts/tvt_service.py --model best_beam scripts/run_m5_planner.py [args...]
     python scripts/tvt_service.py --model best_beam --precompute   (fields of all tuning + development jobs)
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CACHE = ROOT / "results" / "TVT" / "cache" / "service"
FIELDS = ("unblocked_power", "blocked_power", "los_blocked_power", "best_alt_power")
DT_COMM, DT_SENSE = 0.01, 0.1


def sources() -> list[Path]:
    return [ROOT / n for n in ("sim/tvt/service.py", "scripts/tvt_service.py", "xapp/timeline.py", "sim/comm/blockage_torch.py",
                               "sim/scenes/motion.py", "sim/scenes/traffic.py", "sim/positioning/array.py", "sim/tvt/panels.py",
                               "sim/tvt/array_model.py")]


def _tag(model: str) -> str:
    """Cache tag of the fields: model (+ codebook oversampling) + array model (_b2b = back-to-back panels)."""
    from sim.tvt.panels import enabled

    sc = service_cfg()
    tag = f"{model}_os{int(sc['codebook_oversampling'])}" if model == "best_beam" else model
    return tag + ("_b2b" if enabled() else "")


def service_cfg() -> dict[str, Any]:
    from sim.scenes.config import load_yaml

    return load_yaml(ROOT / "configs" / "tvt.yaml")["service"]


def _job_dir(job: tuple) -> Path:
    seed, mount, density = job
    return CACHE / mount / density / f"seed_{seed}"


def _actors(item: tuple) -> tuple:
    """CPU: 10 ms actor poses of one job (as run_m3._cpu_part)."""
    raw, job = item
    import torch

    torch.set_num_threads(2)
    from sim.scenes.traffic import prepare_scenario
    from xapp.timeline import actor_tracks, comm_times, load_geometry

    seed, mount, density = job
    geom, meta = load_geometry(ROOT / "results" / "cache" / mount / density / f"seed_{seed}")
    scenario = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=DT_SENSE)
    times = comm_times(int(meta["n_snapshots"]), DT_SENSE, DT_COMM)
    return job, actor_tracks(scenario, times)


def path_loss(job: tuple, actors: dict) -> np.ndarray:
    """Per-path model-B loss [T,U,C,P] dB of one job (model independent), cached."""
    from sim.sensing.provenance import CacheRefused, require_scoped, scoped_version
    from sim.tvt.service import per_path_loss_timeline
    from xapp.timeline import held_segments, load_geometry

    d = _job_dir(job)
    meta_p = d / "path_loss_meta.json"
    if meta_p.exists():
        try:
            require_scoped(json.loads(meta_p.read_text()), kind="tvt_path_loss", sources=sources())
            return np.load(d / "path_loss.npz")["loss_db"]
        except CacheRefused as e:
            print(f"  refuse path_loss {job}: {e}", flush=True)
    seed, mount, density = job
    geom, meta = load_geometry(ROOT / "results" / "cache" / mount / density / f"seed_{seed}")
    times = np.arange(actors["ue_position_m"].shape[0]) * DT_COMM
    seg = held_segments(geom, times, actors["ue_position_m"], DT_SENSE)
    loss = per_path_loss_timeline(seg, actors["blocker_position_m"], actors["blocker_size_m"], float(meta["wavelength_m"]))
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "path_loss.npz", loss_db=loss)
    meta_p.write_text(json.dumps({"job": list(job), "provenance": {"kind": "tvt_path_loss", **scoped_version(sources())}}) + "\n")
    return loss


def fields(job: tuple, model: str, raw: dict, actors: dict | None = None) -> dict[str, np.ndarray]:
    """The four power fields of ``job`` under ``model`` (cached per model)."""
    from sim.sensing.provenance import CacheRefused, require_scoped, scoped_version
    from sim.tvt.service import timeline_fields
    from xapp.timeline import held_segments, load_geometry

    from sim.tvt.panels import enabled

    sc = service_cfg()
    d = _job_dir(job)
    tag = _tag(model)
    meta_p = d / f"fields_{tag}_meta.json"
    if meta_p.exists():
        try:
            m = json.loads(meta_p.read_text())
            require_scoped(m, kind="tvt_service_fields", sources=sources())
            with np.load(d / f"fields_{tag}.npz") as f:
                return {k: f[k] for k in FIELDS}
        except CacheRefused as e:
            print(f"  refuse fields {job} {tag}: {e}", flush=True)
    if actors is None:
        actors = _actors((raw, job))[1]
    seed, mount, density = job
    geom, meta = load_geometry(ROOT / "results" / "cache" / mount / density / f"seed_{seed}")
    times = np.arange(actors["ue_position_m"].shape[0]) * DT_COMM
    seg = held_segments(geom, times, actors["ue_position_m"], DT_SENSE)
    loss = path_loss(job, actors)
    a_center = np.load(ROOT / "results" / "P2" / "cache" / mount / density / f"seed_{seed}" / "p2_paths.npz")["a_center"]
    n_sc = int(raw["n_subcarriers"])
    df = 15e3 * 2 ** int(raw["numerology"])
    f_off = (np.arange(n_sc) - (n_sc - 1) / 2.0) * df
    pn = None
    if enabled():
        from sim.scenes.traffic import prepare_scenario

        sc0 = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=0.1, dt_s=DT_SENSE)
        pn = {"ue": np.asarray(actors["ue_position_m"], dtype=np.float64), "oru": np.array([o["position_m"] for o in sc0["orus"]], dtype=np.float64)}
    out = timeline_fields(model, seg, loss, a_center, float(meta["wavelength_m"]), f_off, int(round(DT_SENSE / DT_COMM)),
                          oversampling=int(sc["codebook_oversampling"]), panels=pn)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"fields_{tag}.npz", **out)
    meta_p.write_text(json.dumps({"job": list(job), "model": model, "oversampling": int(sc["codebook_oversampling"]), "array": "back_to_back" if enabled() else "single_iso",
                                  "provenance": {"kind": "tvt_service_fields", **scoped_version(sources())}}) + "\n")
    return out


def precompute(jobs: list[tuple], model: str, raw: dict, workers: int = 4) -> None:
    todo = []
    for job in jobs:
        try:
            fields_cached = (_job_dir(job) / f"fields_{_tag(model)}_meta.json").exists()
        except Exception:  # noqa: BLE001
            fields_cached = False
        if not fields_cached:
            todo.append(job)
    print(f"service fields {model}: {len(jobs) - len(todo)} cached, {len(todo)} to compute", flush=True)
    if not todo:
        return
    clock = time.perf_counter()
    with mp.get_context("spawn").Pool(max(1, min(4, workers))) as pool:
        for job, actors in pool.imap(_actors, [(raw, j) for j in todo]):
            fields(job, model, raw, actors)
            print(f"  {job} done ({time.perf_counter() - clock:.0f} s)", flush=True)


def install(model: str) -> None:
    """Patch run_m3.build so that every pipeline sees the fields of ``model``."""
    import run_m3 as R

    from sim.tvt.panels import enabled

    if model == "power_sum" and not enabled():
        return  # the frozen paper-1 fields ARE the isotropic power sum
    orig = R.build
    if getattr(orig, "_tvt_model", None):
        return

    def build(jobs, raw, cfg, workers, rebuild=False):
        out = orig(jobs, raw, cfg, workers, rebuild)
        precompute(list(jobs), model, raw, workers)
        for job in jobs:
            f = fields(job, model, raw)
            data = dict(out[job]["data"])
            for k in FIELDS:
                if data[k].shape != f[k].shape:
                    raise SystemExit(f"{job} {k}: shape {f[k].shape} != frozen {data[k].shape}")
                data[k] = f[k]
            out[job] = {**out[job], "data": data}
        return out

    build._tvt_model = model
    R.build = build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["power_sum", "best_beam", "mrt"])
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("script", nargs="?")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    if a.precompute:
        s = load()
        jobs = [(x, m, dd) for x in check(s["tuning"] + s["development"]) for m in ("lamppost", "facade") for dd in ("low", "high")]
        precompute(jobs, a.model, raw)
        return
    install(a.model)
    name = Path(a.script).stem
    sys.argv = [a.script, *a.args]
    mod = importlib.import_module(name)
    if name != "run_m3":
        install(a.model)  # no-op if already patched
    mod.main()


if __name__ == "__main__":
    main()
