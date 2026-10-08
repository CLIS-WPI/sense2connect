"""TVT panels: radar re-trace (untilted, sim/tvt/radar_trace.py) and per-panel radar detections.

trace   one untilted isotropic radar trace per job (paper-1 scene, targets, solvers, seeds; orientation 0).
detect  per panel (configs/tvt.yaml panels.sensing): the cached paths are turned into the panel's per-element
        coefficients (sim/tvt/panels.radar_panel_paths: TR 38.901 pattern at TX and RX, element phases of the
        panel), then the UNCHANGED paper-1 detection chain of scripts/run_m2_review._detections runs on them
        (CPI synthesis, thermal noise, blind and twin clutter removal, delay-Doppler cube, CA-CFAR, local maxima,
        localisation with the panel's local element positions and orientation); a panel keeps the detections in its
        half-space (sign(x - x_O-RU) = panel sign) and the frame's detections are the union over both panels.
        Thermal noise: the paper-1 seeds (seed * 100000 + frame) for the +x panel, + 50000 for the -x panel.
        --inr k: residual self-interference as in T6 (noise figure + 10 log10(1 + 10^(INR/10)), every panel).
Output: results/TVT/radar/det[_inr<k>]/<mount>/<density>/seed_<s>/detections_1024.json (the paper-1 payload layout,
read by xapp.tracks.load_detections; events.json linked from the paper-1 cache) + per-panel counts.
Run: python scripts/tvt_radar.py trace|detect --sets tuning development [--scenario cfg] [--mounts ...] [--inr 0 10] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DET_ROOT = ROOT / "results" / "TVT" / "radar"
NOISE_OFFSET = {1: 0, -1: 50000}


def det_dir(mount: str, density: str, seed: int, inr: float | None = None) -> Path:
    tag = "det" if not inr else f"det_inr{inr:g}"
    return DET_ROOT / tag / mount / density / f"seed_{seed}"


def _paper_cache(mount: str, density: str, seed: int) -> Path:
    return ROOT / "results" / "cache" / mount / density / f"seed_{seed}"


def _trace_one(item) -> str:
    cfg_path, mount, density, seed = item
    import torch

    torch.set_num_threads(2)
    from sim.scenes.config import load_yaml
    from sim.scenes.traffic import prepare_scenario
    from sim.tvt import radar_trace as RT
    from sim.tvt.scene_register import register

    register()  # custom TVT scenes (intersection) by name in every worker
    cfg_path = Path(cfg_path)
    if RT.fresh(mount, density, seed, cfg_path):
        return f"trace {mount} {density} {seed}: cached"
    raw = load_yaml(cfg_path)
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    meta = RT.trace(sc, mount, density, cfg_path)
    return f"trace {mount} {density} {seed}: {meta['n_frames']} frames in {meta['trace_s'] / 60:.1f} min"


def _cfar_grid(raw: dict) -> list[dict]:
    """The paper-1 CFAR settings: the tracker budgets of configs/m3.yaml and the paper-1 tuning grid (as the paper-1 caches)."""
    import run_m2_review as M
    from sim.scenes.config import load_yaml

    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    grid = [{"guard": int(raw["sensing_radar"]["cfar"]["guard"]), "train": int(b["train"]), "pfa": float(b["pfa"])} for b in cfg["sensing"]["budgets"].values()]
    grid += [dict(g) for g in M.CFAR_GRID]
    return [dict(t) for t in sorted({tuple(sorted(g.items())) for g in grid})]


def detect_job(cfg_path: Path, mount: str, density: str, seed: int, inr: float | None) -> dict:
    import torch

    import run_m2_review as M
    from sim.scenes.config import load_yaml
    from sim.sensing.provenance import scoped_version
    from sim.sensing.waveform import waveform_from_config
    from sim.tvt import panels as PN
    from sim.tvt import radar_trace as RT

    raw = load_yaml(cfg_path)
    if inr:
        raw["sensing_radar"]["noise"]["noise_figure_db"] = float(raw["sensing_radar"]["noise"]["noise_figure_db"]) + 10 * math.log10(1 + 10 ** (inr / 10))
    meta = RT.load_meta(mount, density, seed, cfg_path)
    waveform = waveform_from_config(raw)
    spec = raw["sensing_radar"]
    noise = spec["noise"]
    grid = _cfar_grid(raw)
    lags = M.n_lags(waveform, float(spec["max_range_m"]))
    pos = np.asarray(meta["positions_m"], dtype=np.float64)
    wl = float(meta["wavelength_m"])
    radar_xyz = np.asarray(meta["radar_position_m"], dtype=np.float64)
    eye = np.eye(3)
    radars = {sx: {"positions_m": pos, "orientation_rad": PN.orientation(sx), "radar_position_m": radar_xyz} for sx in PN.SIGNS}
    static = RT.load_static(mount, density, seed)
    static_h = {}
    resid_max = 0.0
    for sx in PN.SIGNS:
        st, r = PN.radar_panel_paths(static, sx, pos, eye, wl)
        resid_max = max(resid_max, r)
        c, dl = M.cpi_from_paths(st, waveform, float(noise["tx_power_dbm"]), max_paths=1024)
        static_h[sx] = M.frequency_responses(c, dl, waveform)
    n_frames = int(meta["n_frames"])
    frames = []
    counts = {str(sx): 0 for sx in PN.SIGNS}
    batch = 4
    index = 0
    while index < n_frames:
        count = min(batch, n_frames - index)
        rows = [RT.load_frame(mount, density, seed, index + o) for o in range(count)]
        dets = [dict() for _ in range(count)]
        for sx in PN.SIGNS:
            measured = []
            for fr in rows:
                h = 0
                for kind in ("rcs", "bg"):
                    part = {k[len(kind) + 1:]: v for k, v in fr.items() if k.startswith(kind + "_")}
                    pp, r = PN.radar_panel_paths(part, sx, pos, eye, wl)
                    resid_max = max(resid_max, r)
                    c, dl = M.cpi_from_paths(pp, waveform, float(noise["tx_power_dbm"]), max_paths=1024)
                    h = h + M.frequency_responses(c, dl, waveform)
                measured.append(h)
            batch_h = torch.cat(measured, dim=0)
            seeds = [int(seed) * 100000 + NOISE_OFFSET[sx] + index + o for o in range(count)]
            batch_h = M.add_noise_batch(batch_h, waveform, float(noise["noise_figure_db"]), float(noise["temperature_k"]), seeds)
            blind = batch_h - batch_h.mean(dim=-2, keepdim=True)
            twin = batch_h - static_h[sx]
            for o in range(count):
                for mode, response in (("blind", blind[o]), ("twin", twin[o])):
                    cube = M.delay_doppler_batch(response[None], lags, waveform.window)[0]
                    power = M.power_map(cube)
                    for s in grid:
                        mask, _a = M.ca_cfar(power, guard=int(s["guard"]), train=int(s["train"]), pfa=float(s["pfa"]), noise_applied=True)
                        hits = M.local_maxima(mask, power)
                        kept = PN.keep_facing(M._locate(cube, hits, waveform, radars[sx]), sx, float(radar_xyz[0]))
                        for d in kept:
                            d["panel"] = int(sx)
                        key = f"{mode}:{int(s['train'])}:{float(s['pfa'])}"
                        dets[o].setdefault(key, []).extend(kept)
                        if mode == "blind" and int(s["train"]) == 4:
                            counts[str(sx)] += len(kept)
        for o, fr in enumerate(rows):
            frames.append({"t_s": (index + o) * 0.1, "ground_truth": M._truth(fr), "detections": dets[o]})
        index += count
    out_d = det_dir(mount, density, seed, inr)
    out_d.mkdir(parents=True, exist_ok=True)
    src = _paper_cache(mount, density, seed)
    for name in ("events.json",):
        t = out_d / name
        if (src / name).exists() and not t.exists():
            os.symlink(src / name, t)
    sources = [ROOT / "sim/tvt/panels.py", ROOT / "scripts/tvt_radar.py", ROOT / "scripts/run_m2_review.py", ROOT / "sim/sensing/process_torch.py",
               ROOT / "sim/sensing/channel.py", ROOT / "sim/sensing/cfar.py", ROOT / "sim/sensing/waveform.py"]
    payload = {"mount": mount, "density": density, "seed": seed, "dt_s": 0.1, "n_frames": n_frames, "radar_position_m": radar_xyz.tolist(),
               "array": "back_to_back panels (configs/tvt.yaml panels)", "inr_db": inr or 0.0, "detections_per_panel_blind_train4": counts,
               "fit_residual_max": resid_max, "frames": frames,
               "provenance": {"kind": "tvt_panel_detections", **scoped_version([p for p in sources if p.exists()])}}
    (out_d / "detections_1024.json").write_text(json.dumps(payload, default=M._json))
    return {"counts": counts, "resid": resid_max}


def _detect_one(item) -> str:
    cfg_path, mount, density, seed, inr = item
    import torch

    torch.set_num_threads(2)
    clock = time.perf_counter()
    out = det_dir(mount, density, seed, inr) / "detections_1024.json"
    if out.exists():
        return f"detect {mount} {density} {seed} inr {inr}: exists"
    r = detect_job(Path(cfg_path), mount, density, seed, inr)
    return f"detect {mount} {density} {seed} inr {inr}: panels {r['counts']} resid {r['resid']:.1e} ({(time.perf_counter() - clock) / 60:.1f} min)"


def main() -> None:
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["trace", "detect"])
    ap.add_argument("--sets", nargs="*", default=["tuning", "development"])
    ap.add_argument("--scenario", default="configs/m2_scenario.yaml")
    ap.add_argument("--mounts", nargs="*", default=["lamppost", "facade"])
    ap.add_argument("--densities", nargs="*", default=["low", "high"])
    ap.add_argument("--inr", type=float, nargs="*", default=[0.0])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    s = load()
    seeds = check([x for name in a.sets for x in s[name]])
    cfg = str(ROOT / a.scenario)
    jobs = [(m, d, x) for x in seeds for m in a.mounts for d in a.densities]
    if a.stage == "trace":
        items = [(cfg, m, d, x) for m, d, x in jobs]
        fn = _trace_one
    else:
        items = [(cfg, m, d, x, inr) for inr in a.inr for m, d, x in jobs]
        fn = _detect_one
    clock = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(fn, items), 1):
            print(f"[{k}/{len(items)} {(time.perf_counter() - clock) / 60:.0f} min] {msg}", flush=True)


if __name__ == "__main__":
    main()
