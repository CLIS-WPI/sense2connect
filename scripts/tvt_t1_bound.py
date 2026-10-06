"""TVT T1: position error bound with an uncertain map and the physical array model, development seeds.

Per development job (10 seeds x 2 mounts x 2 densities, 600 epochs x 2 UEs), main
configuration of configs/tvt.yaml (bound: 400 MHz, TDoA, sigma_sync 1 ns, blocked LoS biased):
1. map sweep: LoS-only PEB and map-aided PEB for sigma_map in configs/tvt.yaml map.sigma_map_m
   (paper-2 array: iso, sigma_phi 2 deg; uncertain surfaces = facades), plus free offsets on all
   known planes including the ground (mapsweep_xyz);
2. array sweep: LoS-only and map-aided (sigma_map 0) PEB for every (pattern, sigma_g, sigma_r) of
   array.sweep, with sigma_phi 2 deg.
Seed level: per seed the median PEB over its 4 runs x 2 UEs x epochs (all / both-LoS / blocked
epochs; blocked = LoS model-B loss >= 10 dB at one O-RU at least), mean over seeds with a
bootstrap CI; exact Wilcoxon for paired configurations. Writes results/TVT/T1/bound/<job>.npz and
results/TVT/T1/bound_summary.json. Run: python scripts/tvt_t1_bound.py [--limit N]
"""

from __future__ import annotations

import argparse
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

K_B, T0 = 1.380649e-23, 290.0
OUT = ROOT / "results" / "TVT" / "T1"


def link(p2cfg: dict, bw: str) -> tuple[np.ndarray, float]:
    spec = p2cfg["bandwidths"][bw]
    n_sc = int(spec["n_sc"])
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    f = (np.arange(n_sc) - (n_sc - 1) / 2.0) * df
    lk = p2cfg["link"]
    noise = K_B * T0 * 10 ** (float(lk["oru_noise_figure_db"]) / 10.0) * n_sc * df
    p_tx = 10 ** ((float(lk["ue_tx_power_dbm"]) - 30.0) / 10.0)
    return f, math.sqrt(p_tx / noise)


def run_job(job, raw, p2cfg, tcfg) -> dict:
    from sim.tvt.peb import job_setup, pebs

    b = tcfg["bound"]
    f, scale = link(p2cfg, str(b["bw"]))
    inp = job_setup(job, raw, p2cfg)
    sm = [float(x) for x in tcfg["map"]["sigma_map_m"]]
    out = {}
    res = pebs(inp, f, scale, sync_ns=float(b["sigma_sync_ns"]), sigma_phi_deg=float(tcfg["array"]["sigma_phi_deg"]), sigma_g_db=0.0, sigma_r_m=0.0,
               pattern="iso", sigma_maps=sm, timing=str(b["timing"]), blocked_db=float(b["blocked_db"]))
    for k, v in res.items():
        out[f"mapsweep|{k}"] = v
    for pat, sg, sr in tcfg["array"]["sweep"]:
        res = pebs(inp, f, scale, sync_ns=float(b["sigma_sync_ns"]), sigma_phi_deg=float(tcfg["array"]["sigma_phi_deg"]), sigma_g_db=float(sg),
                   sigma_r_m=float(sr), pattern=str(pat), sigma_maps=[0.0], timing=str(b["timing"]), blocked_db=float(b["blocked_db"]))
        for k, v in res.items():
            out[f"array|{pat}|g{float(sg):g}|r{float(sr):g}|{k}"] = v
    # all known planes uncertain (facades and ground), offsets free
    inp_xyz = job_setup(job, raw, p2cfg, axes="xyz")
    res = pebs(inp_xyz, f, scale, sync_ns=float(b["sigma_sync_ns"]), sigma_phi_deg=float(tcfg["array"]["sigma_phi_deg"]), sigma_g_db=0.0, sigma_r_m=0.0,
               pattern="iso", sigma_maps=[float("inf")], timing=str(b["timing"]), blocked_db=float(b["blocked_db"]))
    out["mapsweep_xyz|map_inf"] = res["map_inf"]
    blocked = (inp["los_loss"] >= float(b["blocked_db"])).any(-1)  # [T, U]
    return {"peb": out, "blocked": blocked, "ue": inp["ue"]}


def summarize(jobs, store) -> dict:
    from sim.tvt.stats import paired, seed_summary

    seeds = sorted({j[0] for j in jobs})
    keys = list(next(iter(store.values()))["peb"].keys())
    res = {"seeds": seeds, "n_jobs": len(jobs), "configs": {}}
    per_seed = {}
    for k in keys:
        cell = {}
        for state in ("all", "both_los", "blocked"):
            vals = []
            for s in seeds:
                v = []
                for j in jobs:
                    if j[0] != s:
                        continue
                    pe = store[j]["peb"][k][20:]  # first 2 s excluded as in paper 2
                    bl = store[j]["blocked"][20:]
                    sel = np.ones_like(bl) if state == "all" else (bl if state == "blocked" else ~bl)
                    v.append(pe[sel])
                v = np.concatenate(v)
                vals.append(float(np.median(v)) if v.size else math.nan)
            cell[state] = seed_summary(vals)
            per_seed[(k, state)] = vals
            vv = np.concatenate([store[j]["peb"][k][20:].ravel() for j in jobs])
            cell["share_below_0.1m_all"] = float(np.mean(vv < 0.1))
        res["configs"][k] = cell
    # paired tests: map sigma vs perfect map and vs LoS-only; array configs vs the main array
    tests = {}
    for state in ("all", "blocked"):
        base_los = per_seed[("mapsweep|los", state)]
        base_map = per_seed[("mapsweep|map_0", state)]
        for k in keys:
            if k.startswith("mapsweep|map_"):
                tests[f"{k} vs map_0 | {state}"] = paired(per_seed[(k, state)], base_map)
                tests[f"{k} vs los | {state}"] = paired(per_seed[(k, state)], base_los)
            if k.startswith("array|") and not k.startswith("array|iso|g0|r0|"):
                ref = "array|iso|g0|r0|" + k.split("|")[-1]
                tests[f"{k} vs {ref} | {state}"] = paired(per_seed[(k, state)], per_seed[(ref, state)])
    res["paired"] = tests
    return res


def main() -> None:
    import torch

    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
    seeds = check(load()["development"])
    jobs = [(s, m, d) for s in seeds for m in ("lamppost", "facade") for d in ("low", "high")]
    if a.limit:
        jobs = jobs[: a.limit]
    (OUT / "bound").mkdir(parents=True, exist_ok=True)
    store = {}
    clock = time.perf_counter()
    for job in jobs:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        r = run_job(job, raw, p2cfg, tcfg)
        store[job] = r
        np.savez(OUT / "bound" / f"{job[1]}_{job[2]}_{job[0]}.npz", blocked=r["blocked"], **{k.replace("|", "__"): v for k, v in r["peb"].items()})
        print(f"{job}: {time.perf_counter() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
              f"median los {np.median(r['peb']['mapsweep|los']):.3f} map0 {np.median(r['peb']['mapsweep|map_0']):.4f} "
              f"mapinf {np.median(r['peb']['mapsweep|map_inf']):.4f} m", flush=True)
    summ = summarize(jobs, store)
    summ["wall_s"] = time.perf_counter() - clock
    summ["definition"] = __doc__
    (OUT / "bound_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    for k, c in summ["configs"].items():
        print(f"{k:45s} all {c['all']['mean']:.4f} [{c['all']['ci95_boot'][0]:.4f}, {c['all']['ci95_boot'][1]:.4f}]  blocked {c['blocked']['mean']:.4f}  "
              f"both-LoS {c['both_los']['mean']:.4f}  <0.1 m {c['share_below_0.1m_all']:.2f}")


if __name__ == "__main__":
    main()
