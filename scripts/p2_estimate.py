"""Paper 2 (P2-M2): practical positioning estimator on synthesised SRS channels.

Stages (run in order; GPU 1):
  measure  -> per job and configuration: synthesise the uplink SRS channel of
              both O-RUs at every 0.1 s epoch (paths from the paper-1 geometry,
              complex coefficients from results/P2/cache, model-B per path,
              hardware errors) and extract the dominant-path measurement per
              O-RU (sim/positioning/estimator.py). results/P2/est/<set>/<cfg>/<job>.npz
  tune     -> tuning seeds only: grid over (gate_db, floor_tau_ns, floor_u, q,
              chi2) per (bandwidth, timing) at the main hardware; objective:
              median EKF position error, then p90 (pooled over the tuning runs).
              results/P2/est_tuned.json
  evaluate -> development or held-out seeds with the tuned parameters:
              per-epoch EKF / fix errors, saved per job, for every configuration.
Configurations: main (sigma_sync 1 ns constant per run, sigma_phi 2 deg,
blocked LoS "diffracted" = variant b) for bandwidth {100, 200, 400} MHz x
timing {toa, tdoa, aoa}; hardware sweep at 400 MHz TDoA (sigma_sync {0, 0.3,
1, 3} ns x sigma_phi {0, 2, 5} deg); sync drawn per epoch (sensitivity) at
{0.3, 1, 3} ns; blocked LoS "kept" (variant a) at the main 400 MHz TDoA.
Variant b in the estimator: a LoS with model-B loss >= 10 dB arrives via the
shortest detour over or around its dominant blocker (sim/positioning/
blockage.diffracted_los), keeping the model-B attenuation; variant a keeps the
exact LoS geometry. Random numbers (sync offsets, element phases, UE clock
bias U(-50, 50) ns per epoch for TDoA, noise) are common across
configurations of one job (paired comparisons).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

K_B, T0, C0 = 1.380649e-23, 290.0, 299_792_458.0
Z_UE = 1.5
GRID = {"gate_db": [0.0, 6.0, 12.0], "floor_tau_ns": [0.1, 0.3, 1.0], "floor_u": [0.002, 0.01], "res_gate": [30.0, 1e9], "q": [0.1, 1.0],
        "chi2": [9.21, 1e9]}


def configs() -> dict[str, dict]:
    out = {}
    for bw in ("100", "200", "400"):
        for tm in ("toa", "tdoa", "aoa"):
            out[f"bw{bw}_{tm}_s1_p2_b"] = {"bw": bw, "timing": tm, "sync": 1.0, "per_epoch": False, "phi": 2.0, "blocked": "biased"}
    for sy in (0.0, 0.3, 1.0, 3.0):
        for ph in (0.0, 2.0, 5.0):
            out.setdefault(f"bw400_tdoa_s{sy:g}_p{ph:g}_b", {"bw": "400", "timing": "tdoa", "sync": sy, "per_epoch": False, "phi": ph, "blocked": "biased"})
    for sy in (0.3, 1.0, 3.0):
        out[f"bw400_tdoa_s{sy:g}e_p2_b"] = {"bw": "400", "timing": "tdoa", "sync": sy, "per_epoch": True, "phi": 2.0, "blocked": "biased"}
    out["bw400_tdoa_s1_p2_a"] = {"bw": "400", "timing": "tdoa", "sync": 1.0, "per_epoch": False, "phi": 2.0, "blocked": "kept"}
    return out


def main_cfg_name(bw: str, timing: str) -> str:
    return f"bw{bw}_{timing}_s1_p2_b"


def _job_seed(job: tuple, *extra) -> list[int]:
    h = hashlib.sha256(("|".join(map(str, job + tuple(extra)))).encode()).digest()
    return [int.from_bytes(h[i:i + 4], "little") for i in range(0, 16, 4)]


def job_paths(job: tuple, raw: dict, p2cfg: dict) -> dict:
    """Per-path inputs [T, U, C, P] shared by all configurations of one job."""
    from sim.positioning.blockage import diffracted_los, path_losses_db
    from sim.scenes.traffic import prepare_scenario

    seed, mount, density = job
    gdir = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    with np.load(gdir / "comm_geometry.npz") as g:
        geom = {k: g[k] for k in g.files}
    wl = float(json.loads((gdir / "comm_geometry_meta.json").read_text())["wavelength_m"])
    with np.load(ROOT / "results" / "P2" / "cache" / mount / density / f"seed_{seed}" / "p2_paths.npz") as q:
        a_center = q["a_center"]
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    bl = path_losses_db(geom, sc, wl)
    valid = geom["path_class"] >= 0
    loss = np.where(valid, bl["loss_db"], np.inf)
    amp = np.where(np.isfinite(loss), 10.0 ** (-np.nan_to_num(loss, posinf=0.0) / 20.0), 0.0)
    P = geom["points_m"]
    n = geom["n_points"].astype(np.int64)
    # path length and arrival direction straight from the cached polyline (exact geometry)
    pts = np.nan_to_num(P)
    seg = np.linalg.norm(pts[..., 1:, :] - pts[..., :-1, :], axis=-1)
    length = np.where(np.arange(3) < (n[..., None] - 1), seg, 0.0).sum(-1)
    first = pts[..., 1, :] - pts[..., 0, :]
    u = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    cls = geom["path_class"].astype(np.int64)
    los_loss = np.where(cls == 0, loss, -1.0).max(-1)
    blocked = los_loss >= float(p2cfg["blockage"]["blocked_los_db"])  # [T, U, C]
    # diffracted LoS (variant b) for blocked LoS paths
    length_b, u_b = length.copy(), u.copy()
    T, U, C, Pn = cls.shape
    oru = geom["oru_position_m"]  # [T, C, 3]
    ue = geom["ue_position_m"]  # [T, U, 3]
    for t in range(T):
        for uu in range(U):
            for c in range(C):
                if not blocked[t, uu, c]:
                    continue
                for p in np.flatnonzero(cls[t, uu, c] == 0):
                    b = int(bl["dominant_blocker"][t, uu, c, p])
                    qd = diffracted_los(oru[t, c], ue[t, uu], bl["blocker_position_m"][t, b], bl["blocker_size_m"][b])
                    length_b[t, uu, c, p] = np.linalg.norm(qd - oru[t, c]) + np.linalg.norm(ue[t, uu] - qd)
                    d = qd - oru[t, c]
                    u_b[t, uu, c, p] = d / np.linalg.norm(d)
    return {"wl": wl, "a": np.where(valid, a_center, 0.0) * amp, "tau_ns": length / C0 * 1e9, "u": u, "tau_b_ns": length_b / C0 * 1e9, "u_b": u_b,
            "ue": ue, "oru": oru, "blocked": blocked, "los_loss": los_loss}


def measure_job(job: tuple, jp: dict, cfg: dict, p2cfg: dict, device: str = "cuda") -> dict:
    import torch

    from sim.positioning.array import element_positions
    from sim.positioning.estimator import measure_torch, synth_torch

    T, U, C, P = jp["a"].shape
    spec = p2cfg["bandwidths"][cfg["bw"]]
    n_sc = int(spec["n_sc"])
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    f = (np.arange(n_sc) - (n_sc - 1) / 2.0) * df
    link = p2cfg["link"]
    noise = K_B * T0 * 10 ** (float(link["oru_noise_figure_db"]) / 10.0) * n_sc * df
    p_tx = 10 ** ((float(link["ue_tx_power_dbm"]) - 30.0) / 10.0)
    beta = jp["a"] * math.sqrt(p_tx / noise)
    tau = jp["tau_b_ns"] if cfg["blocked"] == "biased" else jp["tau_ns"]
    u = jp["u_b"] if cfg["blocked"] == "biased" else jp["u"]
    rng = np.random.default_rng(_job_seed(job, "hw"))
    z_run = rng.standard_normal(C)
    z_ep = rng.standard_normal((T, C))
    z_psi = rng.standard_normal((C, 64))
    b_clk = rng.uniform(-50.0, 50.0, T)
    delta = cfg["sync"] * (z_ep if cfg["per_epoch"] else np.broadcast_to(z_run, (T, C)))  # [T, C] ns
    shift = delta[:, None, :] + (b_clk[:, None, None] if cfg["timing"] == "tdoa" else 0.0)  # [T, 1|U, C]
    shift = np.broadcast_to(shift, (T, U, C))
    psi = math.radians(cfg["phi"]) * z_psi  # [C, 64]
    r = element_positions(jp["wl"])
    items = [(t, uu, c) for t in range(T) for uu in range(U) for c in range(C)]
    gen = torch.Generator(device=device)
    gen.manual_seed(_job_seed(job, "noise", cfg["bw"])[0])
    res = {k: np.zeros((T, U, C)) for k in ("tau_ns", "uy", "uz", "snr", "ratio_db")}
    chunk = 48
    for lo in range(0, len(items), chunk):
        it = items[lo:lo + chunk]
        ti, ui, ci = (np.array(x) for x in zip(*it))
        bt = torch.as_tensor(beta[ti, ui, ci], device=device)
        tt = torch.as_tensor(tau[ti, ui, ci] + shift[ti, ui, ci][:, None], device=device)
        ut = torch.as_tensor(u[ti, ui, ci], device=device)
        ps = torch.as_tensor(psi[ci], device=device)
        Y = synth_torch(bt, tt, ut, f, r, jp["wl"], psi=ps, gen=gen)
        m = measure_torch(Y, f, jp["wl"])
        for k in res:
            res[k][ti, ui, ci] = m[k]
    res["tau_ns"] = np.where(res["tau_ns"] > 0.5e9 / df, res["tau_ns"] - 1e9 / df, res["tau_ns"])  # unwrap to (-1/(2 df), 1/(2 df)]
    return res


def track(meas: dict, jp: dict, timing: str, params: dict, n_sc: int, df: float, memo: dict | None = None) -> dict:
    """Gate + WLS candidates + EKF per UE; ``memo`` caches the WLS candidates per (UE, gate, floors)."""
    from sim.positioning.estimator import noise_model, run_tracker, wls_candidates

    T, U, C = meas["uy"].shape
    xy_ekf = np.zeros((T, U, 2))
    xy_fix = np.zeros((T, U, 2))
    for uu in range(U):
        key = (uu, params["gate_db"], params["floor_tau_ns"], params["floor_u"])
        if memo is not None and key in memo:
            cand = memo[key]
        else:
            m = {k: v[:, uu, :] for k, v in meas.items()}
            use, st, su = noise_model(m, params, df, n_sc)
            cand = {"all": wls_candidates(m, jp["oru"], use, timing, st, su, Z_UE)}
            for c in range(C):
                only = np.zeros_like(use)
                only[:, c] = use[:, c]
                cand[c] = wls_candidates(m, jp["oru"], only, timing, st, su, Z_UE)
            if memo is not None:
                memo[key] = cand
        tr = run_tracker(cand, meas["snr"][:, uu, :], params)
        xy_ekf[:, uu] = tr["xy_ekf"]
        xy_fix[:, uu] = tr["xy_fix"]
    return {"xy_ekf": xy_ekf, "xy_fix": xy_fix}


def jobs_of(which: str) -> list[tuple]:
    from seedsets import load_seeds

    seeds = load_seeds()
    return [(int(s), m, d) for s in seeds[which] for m in ("lamppost", "facade") for d in ("low", "high")]


def main() -> None:
    import torch

    from seedsets import eval_set
    from sim.scenes.config import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "tune", "evaluate"])
    ap.add_argument("--set", default="evaluation", help="tuning | evaluation")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    tag = "tuning" if args.set == "tuning" else ("heldout" if eval_set() == "heldout" else "dev")
    jobs = jobs_of(args.set)[: args.limit or None]
    cfgs = configs()
    names = args.configs or (list(cfgs) if args.set != "tuning" else [main_cfg_name(bw, tm) for bw in ("100", "200", "400") for tm in ("toa", "tdoa", "aoa")])
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    base = ROOT / "results" / "P2" / "est" / tag
    clock = time.perf_counter()
    if args.stage == "measure":
        if torch.cuda.device_count() != 1:
            raise SystemExit("expected exactly one visible GPU (GPU 1)")
        for job in jobs:
            t0 = time.perf_counter()
            jp = None
            for name in names:
                dest = base / name / ("%s_%s_%d.npz" % (job[1], job[2], job[0]))
                if dest.exists():
                    continue
                jp = jp or job_paths(job, raw, p2cfg)
                dest.parent.mkdir(parents=True, exist_ok=True)
                m = measure_job(job, jp, cfgs[name], p2cfg)
                np.savez(dest, **m, ue=jp["ue"], oru=jp["oru"], blocked=jp["blocked"], los_loss=jp["los_loss"])
            print(f"measured {job} ({len(names)} configs) in {time.perf_counter() - t0:.0f} s", flush=True)
        return
    if args.stage == "tune":
        if args.set != "tuning":
            raise SystemExit("tuning runs on the tuning seeds only")
        tuned = {}
        for bw in ("100", "200", "400"):
            for tm in ("toa", "tdoa", "aoa"):
                name = main_cfg_name(bw, tm)
                data = [dict(np.load(base / name / ("%s_%s_%d.npz" % (j[1], j[2], j[0])))) for j in jobs]
                memos = [{} for _ in data]
                n_sc = int(p2cfg["bandwidths"][bw]["n_sc"])
                scores = {}
                for combo in itertools.product(*GRID.values()):
                    params = dict(zip(GRID, combo))
                    errs = []
                    for d, memo in zip(data, memos):
                        meas = {k: d[k] for k in ("tau_ns", "uy", "uz", "snr", "ratio_db")}
                        out = track(meas, {"oru": d["oru"]}, tm, params, n_sc, df, memo)
                        e = np.linalg.norm(out["xy_ekf"] - d["ue"][..., :2], axis=-1)
                        errs.append(np.where(np.isfinite(e), e, 1e3).ravel())
                    e = np.concatenate(errs)
                    scores[combo] = (float(np.median(e)), float(np.percentile(e, 90)))
                best = min(scores, key=lambda c: (scores[c][0], scores[c][1], c))
                tuned[f"{bw}|{tm}"] = {"params": dict(zip(GRID, best)), "median_m": scores[best][0], "p90_m": scores[best][1],
                                       "grid_points": len(scores)}
                print(f"tuned {bw} MHz {tm}: {tuned[f'{bw}|{tm}']}", flush=True)
        (ROOT / "results" / "P2" / "est_tuned.json").write_text(json.dumps({"grid": GRID, "tuned": tuned, "jobs": [list(j) for j in jobs],
                                                                            "objective": "median EKF error, then p90, pooled over tuning runs"}, indent=1) + "\n")
        return
    tuned = json.loads((ROOT / "results" / "P2" / "est_tuned.json").read_text())["tuned"]
    for name in names:
        c = cfgs[name]
        params = tuned[f"{c['bw']}|{c['timing']}"]["params"]
        n_sc = int(p2cfg["bandwidths"][c["bw"]]["n_sc"])
        for job in jobs:
            src = base / name / ("%s_%s_%d.npz" % (job[1], job[2], job[0]))
            d = np.load(src)
            meas = {k: d[k] for k in ("tau_ns", "uy", "uz", "snr", "ratio_db")}
            out = track(meas, {"oru": d["oru"]}, c["timing"], params, n_sc, df)
            np.savez(base / name / ("%s_%s_%d_track.npz" % (job[1], job[2], job[0])), **out)
        print(f"evaluated {name} on {len(jobs)} jobs ({time.perf_counter() - clock:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
