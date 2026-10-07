"""TVT T4 stage 1: multipath components per job, epoch, UE and O-RU (tuning and development seeds).

The SRS snapshots are synthesised EXACTLY as the paper-2 estimator does (scripts/p2_estimate.py
measure_job, main configuration bw400_tdoa_s1_p2_b: same paths, model-B amplitudes, diffracted
blocked LoS, run-constant O-RU offsets sigma_sync 1 ns, element phase errors 2 deg, UE clock bias
U(-50, 50) ns per epoch, same random streams and batching), so the tracker sees the same noise
realisations as the paper-2 estimator. On each snapshot both the frozen paper-2 dominant-path
measurement (regression: equal to results/P2/est/dev/<cfg>/<job>.npz on development jobs) and the
T2 extraction (sim/tvt/extract.py, configs/tvt.yaml extract) are run. Delays are unwrapped to
(-1/(2 df), 1/(2 df)]. Writes results/TVT/T4/meas/<set>/<mount>_<density>_<seed>.npz.
Run: python scripts/tvt_t4_measure.py --set tuning|development [--cfg bw400_tdoa_s1_p2_b] [--limit N]
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

OUT = ROOT / "results" / "TVT" / "T4" / "meas"


def measure(job, raw, p2cfg, ecfg, cfg_name: str, device: str = "cuda") -> dict:
    import torch

    import p2_estimate as P
    from sim.positioning.array import element_positions
    from sim.positioning.estimator import measure_torch, synth_torch
    from sim.tvt.extract import extract

    cfg = P.configs()[cfg_name]
    jp = P.job_paths(job, raw, p2cfg)
    T, U, C, Pn = jp["a"].shape
    spec = p2cfg["bandwidths"][cfg["bw"]]
    n_sc = int(spec["n_sc"])
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    f = (np.arange(n_sc) - (n_sc - 1) / 2.0) * df
    link = p2cfg["link"]
    noise = P.K_B * P.T0 * 10 ** (float(link["oru_noise_figure_db"]) / 10.0) * n_sc * df
    p_tx = 10 ** ((float(link["ue_tx_power_dbm"]) - 30.0) / 10.0)
    beta = jp["a"] * math.sqrt(p_tx / noise)
    tau = jp["tau_b_ns"] if cfg["blocked"] == "biased" else jp["tau_ns"]
    u = jp["u_b"] if cfg["blocked"] == "biased" else jp["u"]
    # --- identical to p2_estimate.measure_job ---
    rng = np.random.default_rng(P._job_seed(job, "hw"))
    z_run = rng.standard_normal(C)
    z_ep = rng.standard_normal((T, C))
    z_psi = rng.standard_normal((C, 64))
    b_clk = rng.uniform(-50.0, 50.0, T)
    delta = cfg["sync"] * (z_ep if cfg["per_epoch"] else np.broadcast_to(z_run, (T, C)))
    shift = delta[:, None, :] + (b_clk[:, None, None] if cfg["timing"] == "tdoa" else 0.0)
    shift = np.broadcast_to(shift, (T, U, C))
    psi = math.radians(cfg["phi"]) * z_psi
    r = element_positions(jp["wl"])
    items = [(t, uu, c) for t in range(T) for uu in range(U) for c in range(C)]
    gen = torch.Generator(device=device)
    gen.manual_seed(P._job_seed(job, "noise", cfg["bw"])[0])
    K = int(ecfg["k_max"])
    from sim.tvt import panels as PN

    use_panels = PN.enabled()
    NP = 2 if use_panels else 1  # back-to-back panels: components of both panels side by side (K per panel)
    dom = {k: np.zeros((T, U, C)) for k in ("tau_ns", "uy", "uz", "snr", "ratio_db")}
    comp = {k: np.zeros((T, U, C, NP * K)) for k in ("tau_ns", "uy", "uz", "var_tau", "var_uy", "var_uz", "amp2")}
    comp["valid"] = np.zeros((T, U, C, NP * K), dtype=bool)
    if use_panels:
        # panel sign of every component / of the dominant-path measurement; -x panel: own element phase errors and noise
        comp["sx"] = np.repeat(np.array(PN.SIGNS), K)[None, None, None, :].repeat(T, 0).repeat(U, 1).repeat(C, 2)
        dom["sx"] = np.ones((T, U, C))
        dom["energy_ratio_db"] = np.zeros((T, U, C))
        psi_m = math.radians(cfg["phi"]) * np.random.default_rng(P._job_seed(job, "hw_panel_minus")).standard_normal((C, 64))
        gen_m = torch.Generator(device=device)
        gen_m.manual_seed(P._job_seed(job, "noise_panel_minus", cfg["bw"])[0])
        amp_pn = PN.snapshot_amplitude(u)  # [2, T, U, C, P]
    dup_removed = [0, 0]  # back-lobe copies removed, components before
    chunk = 48
    for lo in range(0, len(items), chunk):
        it = items[lo:lo + chunk]
        ti, ui, ci = (np.array(x) for x in zip(*it))
        bt = torch.as_tensor(beta[ti, ui, ci], device=device)
        tt = torch.as_tensor(tau[ti, ui, ci] + shift[ti, ui, ci][:, None], device=device)
        ut = torch.as_tensor(u[ti, ui, ci], device=device)
        ps = torch.as_tensor(psi[ci], device=device)
        if not use_panels:
            Y = synth_torch(bt, tt, ut, f, r, jp["wl"], psi=ps, gen=gen)
            m = measure_torch(Y, f, jp["wl"])
            for k in dom:
                dom[k][ti, ui, ci] = m[k]
            ex = extract(Y, f, jp["wl"], k_max=K, pfa=float(ecfg["pfa"]), dyn_range_db=float(ecfg["dyn_range_db"]), q=int(ecfg["q"]))
            for k in ("tau_ns", "uy", "uz", "var_tau", "var_uy", "var_uz"):
                comp[k][ti, ui, ci] = ex[k]
            comp["amp2"][ti, ui, ci] = np.abs(ex["beta"]) ** 2
            comp["valid"][ti, ui, ci] = ex["valid"]
            continue
        per = []
        for pi, sx in enumerate(PN.SIGNS):
            btp = bt * torch.as_tensor(amp_pn[pi][ti, ui, ci], device=device)
            rr = r @ PN.rotation(sx).T  # world element positions of the panel
            Y = synth_torch(btp, tt, ut, f, rr, jp["wl"], psi=ps if sx == 1 else torch.as_tensor(psi_m[ci], device=device),
                            gen=gen if sx == 1 else gen_m)
            energy = (Y.abs() ** 2).sum((1, 2)).double().cpu().numpy()
            m = measure_torch(Y, f, jp["wl"])  # local frame of the panel: world u_y = sx * local u_y
            m["uy"] = sx * np.asarray(m["uy"])
            ex = extract(Y, f, jp["wl"], k_max=K, pfa=float(ecfg["pfa"]), dyn_range_db=float(ecfg["dyn_range_db"]), q=int(ecfg["q"]))
            ex["uy"] = sx * np.asarray(ex["uy"])
            sl = slice(pi * K, (pi + 1) * K)
            for k in ("tau_ns", "uy", "uz", "var_tau", "var_uy", "var_uz"):
                comp[k][ti, ui, ci, sl] = ex[k]
            comp["amp2"][ti, ui, ci, sl] = np.abs(ex["beta"]) ** 2
            comp["valid"][ti, ui, ci, sl] = ex["valid"]
            per.append((energy, m))
        # the same path on both panels (back-lobe copy): one resolution cell apart -> keep the stronger component only
        n_sc_b = float(f.size) * float(f[1] - f[0])  # occupied bandwidth [Hz]
        per_ns = 1e9 / float(f[1] - f[0])
        c0, c1 = (slice(0, K), slice(K, 2 * K))
        g_ = lambda k, sl: comp[k][ti, ui, ci, sl]  # noqa: E731
        dt = g_("tau_ns", c0)[:, :, None] - g_("tau_ns", c1)[:, None, :]
        dt = (dt + 0.5 * per_ns) % per_ns - 0.5 * per_ns
        d2 = (dt * 1e-9 * n_sc_b) ** 2 + ((g_("uy", c0)[:, :, None] - g_("uy", c1)[:, None, :]) * 4) ** 2 + ((g_("uz", c0)[:, :, None] - g_("uz", c1)[:, None, :]) * 4) ** 2
        same = (d2 < 1.0) & g_("valid", c0)[:, :, None] & g_("valid", c1)[:, None, :]
        a0, a1 = g_("amp2", c0)[:, :, None], g_("amp2", c1)[:, None, :]
        drop0 = (same & (a0 < a1)).any(2)
        drop1 = (same & (a1 <= a0)).any(1)
        dup_removed[0] += int(drop0.sum() + drop1.sum())
        dup_removed[1] += int(g_("valid", c0).sum() + g_("valid", c1).sum())
        comp["valid"][ti, ui, ci, c0] = g_("valid", c0) & ~drop0
        comp["valid"][ti, ui, ci, c1] = g_("valid", c1) & ~drop1
        sel = per[0][0] >= per[1][0]  # dominant-path measurement from the panel with the larger snapshot energy
        for k in ("tau_ns", "uy", "uz", "snr", "ratio_db"):
            dom[k][ti, ui, ci] = np.where(sel, per[0][1][k], per[1][1][k])
        dom["sx"][ti, ui, ci] = np.where(sel, 1.0, -1.0)
        dom["energy_ratio_db"][ti, ui, ci] = 10 * np.log10(np.maximum(per[0][0], 1e-30) / np.maximum(per[1][0], 1e-30))
    period = 1e9 / df
    dom["tau_ns"] = np.where(dom["tau_ns"] > 0.5 * period, dom["tau_ns"] - period, dom["tau_ns"])
    comp["tau_ns"] = np.mod(comp["tau_ns"], period)
    comp["tau_ns"] = np.where(comp["tau_ns"] > 0.5 * period, comp["tau_ns"] - period, comp["tau_ns"])
    if use_panels:
        print(f"  {job}: cross-panel duplicates removed {dup_removed[0]} of {dup_removed[1]} components", flush=True)
    return {"dom": dom, "comp": comp, "ue": jp["ue"], "oru": jp["oru"], "blocked": jp["blocked"], "los_loss": jp["los_loss"],
            "truth": {"delta_ns": delta, "b_clk_ns": b_clk, "psi": psi}, "period_ns": period, "n_sc": n_sc, "df": df}


def main() -> None:
    import torch

    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="development", choices=["tuning", "development"])
    ap.add_argument("--cfg", default="bw400_tdoa_s1_p2_b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenario", default="configs/m2_scenario.yaml")
    ap.add_argument("--p2cfg", default="configs/p2.yaml")
    ap.add_argument("--mounts", nargs="*", default=["lamppost", "facade"])
    a = ap.parse_args()
    raw = load_yaml(ROOT / a.scenario)
    p2cfg = load_yaml(ROOT / a.p2cfg)
    ecfg = load_yaml(ROOT / "configs" / "tvt.yaml")["extract"]
    seeds = check(load()[a.set])
    jobs = [(s, m, d) for s in seeds for m in a.mounts for d in ("low", "high")]
    if a.limit:
        jobs = jobs[: a.limit]
    out = OUT / a.set / a.cfg
    out.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    clock = time.perf_counter()
    for job in jobs:
        t0 = time.perf_counter()
        res = measure(job, raw, p2cfg, ecfg, a.cfg)
        ref = ROOT / "results" / "P2" / "est" / "dev" / a.cfg / f"{job[1]}_{job[2]}_{job[0]}.npz"
        if a.set == "development" and ref.exists() and not __import__("sim.tvt.panels", fromlist=["enabled"]).enabled():
            with np.load(ref) as g:
                for k in ("tau_ns", "uy", "uz"):
                    worst = max(worst, float(np.max(np.abs(g[k] - res["dom"][k]))))
        np.savez(out / f"{job[1]}_{job[2]}_{job[0]}.npz", **{f"dom_{k}": v for k, v in res["dom"].items()}, **{f"comp_{k}": v for k, v in res["comp"].items()},
                 ue=res["ue"], oru=res["oru"], blocked=res["blocked"], los_loss=res["los_loss"], delta_ns=res["truth"]["delta_ns"],
                 b_clk_ns=res["truth"]["b_clk_ns"], period_ns=res["period_ns"])
        nv = res["comp"]["valid"].sum(-1)
        print(f"{job}: {time.perf_counter() - t0:.0f} s, components per snapshot {nv.mean():.2f} (max {nv.max()}), "
              f"paper-2 regression max |diff| {worst:.2e}", flush=True)
    panels_on = __import__("sim.tvt.panels", fromlist=["enabled"]).enabled()
    meta = {"set": a.set, "cfg": a.cfg, "jobs": [list(j) for j in jobs], "extract": ecfg, "array": "back_to_back" if panels_on else "single_iso",
            "paper2_measurement_max_abs_diff": None if panels_on else worst,
            "wall_s": time.perf_counter() - clock}
    (out / f"meta_{'_'.join(a.mounts)}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
