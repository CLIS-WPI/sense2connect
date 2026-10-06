"""TVT T2: multipath extraction (sim/tvt/extract.py) on fixed geometries, development seed.

Geometries: as the paper-2 validation table but from DEVELOPMENT job 1001 (low density, UE 0):
per mount the both-LoS epochs whose UE x is closest to -20 m and +20 m and the blocked epoch at
the median index of the blocked epochs. Per geometry, bandwidth (100, 400 MHz) and hardware
(ideal: no sync / phase error; main: sigma_sync 1 ns, sigma_phi 2 deg): N_DRAWS independent
noise, O-RU time offset, element-phase and UE clock-bias (U(-50, 50) ns) draws of the SRS
snapshot of each O-RU (frozen synthesis, blocked LoS as the diffracted detour = paper-2 main
variant b). True paths: all traced paths with nonzero amplitude (per-path SNR
|beta|^2 N M). Matching: an extracted component matches the nearest unmatched true path if the
normalised distance (dtau B)^2 + (du_y 8/2)^2 + (du_z 8/2)^2 < 1 (one resolution cell; delays
compared modulo 1/df after removing nothing: the common clock offset is part of every true
delay). Metrics: detection rate of true paths per SNR bin, false components per snapshot
(unmatched), normalised errors e / sqrt(CRB) of matched components (single-path CRB with the
true gain; RMS ~ 1 for an efficient estimator, > 1 with interference / model mismatch), the
covariance consistency (RMS of e / sqrt(reported var)), run time per snapshot (GPU, batched).
Writes results/TVT/T2/extract_<tag>.json. Run: python scripts/tvt_t2_extract.py [--draws 200]
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

OUT = ROOT / "results" / "TVT" / "T2"
ISO_CELLS = 2.0  # a true path is "isolated" if no other true path lies within 2 resolution cells
BINS = [(-np.inf, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, np.inf)]


def crb(beta_abs2, N, M, f, k, yz):
    fz = ((2 * np.pi * f * 1e-9) ** 2).sum()
    yc = yz[:, 0] - yz[:, 0].mean()
    zc = yz[:, 1] - yz[:, 1].mean()
    return (1 / (2 * beta_abs2 * M * fz), 1 / (2 * beta_abs2 * N * k ** 2 * (yc ** 2).sum()), 1 / (2 * beta_abs2 * N * k ** 2 * (zc ** 2).sum()))


def main() -> None:
    import torch

    import p2_estimate as P
    from sim.positioning.array import element_positions
    from sim.positioning.estimator import synth_torch
    from sim.scenes.config import load_yaml
    from sim.tvt.extract import extract
    from sim.tvt.seeds import check

    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dyn-range-db", type=float, default=None)
    ap.add_argument("--pfa", type=float, default=None)
    ap.add_argument("--sweeps", type=int, default=2)
    a = ap.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    ecfg = load_yaml(ROOT / "configs" / "tvt.yaml")["extract"]
    dyn = float(a.dyn_range_db if a.dyn_range_db is not None else ecfg["dyn_range_db"])
    pfa = float(a.pfa if a.pfa is not None else ecfg["pfa"])
    seed = check([1001])[0]
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    lk = p2cfg["link"]
    rows = []
    timing = []
    for mount in ("lamppost", "facade"):
        jp = P.job_paths((seed, mount, "low"), raw, p2cfg)
        wl = jp["wl"]
        r = element_positions(wl)
        yz = r[:, 1:]
        k = 2 * np.pi / wl
        u = 0
        nb = jp["blocked"][:, u, :].any(-1)
        x = jp["ue"][:, u, 0]
        li, bi = np.flatnonzero(~nb), np.flatnonzero(nb)
        picks = [("both LoS, x near -20 m", int(li[np.argmin(np.abs(x[li] + 20))])), ("both LoS, x near +20 m", int(li[np.argmin(np.abs(x[li] - 20))]))]
        if bi.size:
            picks.append(("blocked", int(bi[len(bi) // 2])))
        for gname, t in picks:
            for bw in ("100", "400"):
                n_sc = int(p2cfg["bandwidths"][bw]["n_sc"])
                f = (np.arange(n_sc) - (n_sc - 1) / 2.0) * df
                noise = P.K_B * P.T0 * 10 ** (float(lk["oru_noise_figure_db"]) / 10.0) * n_sc * df
                p_tx = 10 ** ((float(lk["ue_tx_power_dbm"]) - 30.0) / 10.0)
                beta = jp["a"][t, u] * math.sqrt(p_tx / noise)  # [C, P]
                tau = jp["tau_b_ns"][t, u]
                uu = jp["u_b"][t, u]
                for hw, (ss, ph) in (("ideal", (0.0, 0.0)), ("main", (1.0, 2.0))):
                    rng = np.random.default_rng([seed, ["lamppost", "facade"].index(mount), t, int(bw), int(ss * 10), int(ph), 77])
                    gen = torch.Generator(device="cuda")
                    gen.manual_seed(int(rng.integers(1 << 31)))
                    C = beta.shape[0]
                    stats = {"det": [], "snr": [], "nerr": [], "cons": [], "false": [], "n_true": [], "iso": []}
                    for c in range(C):
                        present = np.abs(beta[c]) > 0
                        if not present.any():
                            continue
                        nd = a.draws
                        delta = ss * rng.standard_normal(nd)
                        bclk = rng.uniform(-50.0, 50.0, nd)
                        psi = math.radians(ph) * rng.standard_normal((nd, 64))
                        bt = torch.as_tensor(np.broadcast_to(beta[c], (nd, beta.shape[1])).copy(), device="cuda")
                        tt_np = tau[c][None, :] + (delta + bclk)[:, None]
                        tt = torch.as_tensor(tt_np, device="cuda")
                        ut = torch.as_tensor(np.broadcast_to(uu[c], (nd,) + uu[c].shape).copy(), device="cuda")
                        Y = synth_torch(bt, tt, ut, f, r, wl, psi=torch.as_tensor(psi, device="cuda"), gen=gen)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        ex = extract(Y, f, wl, k_max=int(ecfg["k_max"]), pfa=pfa, dyn_range_db=dyn, q=int(ecfg["q"]), sweeps=a.sweeps)
                        torch.cuda.synchronize()
                        timing.append((time.perf_counter() - t0) / nd)
                        period = 1e9 / df
                        snr_db = 10 * np.log10(np.abs(beta[c][present]) ** 2 * n_sc * 64)
                        vt, vy, vz = crb(np.abs(beta[c][present]) ** 2, n_sc, 64, f, k, yz)
                        # separation of every true path from its nearest other true path, in resolution cells
                        tp, up = tau[c][present], uu[c][present][:, 1:]
                        sep = np.sqrt(((tp[:, None] - tp[None, :]) * n_sc * df * 1e-9) ** 2 + ((up[:, None, 0] - up[None, :, 0]) * 4) ** 2
                                      + ((up[:, None, 1] - up[None, :, 1]) * 4) ** 2)
                        np.fill_diagonal(sep, np.inf)
                        isolated = sep.min(1) >= ISO_CELLS
                        for d in range(nd):
                            tr_tau = np.mod(tt_np[d][present], period)
                            tr_u = uu[c][present][:, 1:]
                            v = ex["valid"][d]
                            et, ey, ez = np.mod(ex["tau_ns"][d][v], period), ex["uy"][d][v], ex["uz"][d][v]
                            used = np.zeros(present.sum(), bool)
                            det = np.zeros(present.sum(), bool)
                            nfalse = 0
                            order = np.argsort(-np.abs(ex["beta"][d][v]))
                            for j in order:
                                dt_ = np.abs(et[j] - tr_tau)
                                dt_ = np.minimum(dt_, period - dt_)
                                dist = (dt_ * n_sc * df * 1e-9) ** 2 + ((ey[j] - tr_u[:, 0]) * 4) ** 2 + ((ez[j] - tr_u[:, 1]) * 4) ** 2
                                dist = np.where(used, np.inf, dist)
                                i = int(np.argmin(dist))
                                if dist[i] < 1.0:
                                    used[i] = det[i] = True
                                    e = np.array([(et[j] - tr_tau[i] + period / 2) % period - period / 2, ey[j] - tr_u[i, 0], ez[j] - tr_u[i, 1]])
                                    stats["nerr"].append((snr_db[i], e[0] / math.sqrt(vt[i]), e[1] / math.sqrt(vy[i]), e[2] / math.sqrt(vz[i]), float(isolated[i])))
                                    vj = np.flatnonzero(v)[j]
                                    stats["cons"].append((snr_db[i], e[0] / math.sqrt(ex["var_tau"][d][vj]), e[1] / math.sqrt(ex["var_uy"][d][vj]),
                                                          e[2] / math.sqrt(ex["var_uz"][d][vj])))
                                else:
                                    nfalse += 1
                            stats["det"].append(det)
                            stats["iso"].append(isolated)
                            stats["snr"].append(snr_db)
                            stats["false"].append(nfalse)
                            stats["n_true"].append(int(present.sum()))
                    det = np.concatenate(stats["det"])
                    snr = np.concatenate(stats["snr"])
                    ne = np.array(stats["nerr"]).reshape(-1, 5)
                    iso = np.concatenate(stats["iso"])
                    co = np.array(stats["cons"]).reshape(-1, 4)
                    row = {"mount": mount, "geometry": gname, "epoch": t, "bw": bw, "hardware": hw, "draws": a.draws,
                           "false_per_snapshot": float(np.mean(stats["false"])), "true_paths_per_snapshot": float(np.mean(stats["n_true"])), "bins": []}
                    for lo, hi in BINS:
                        sel = (snr >= lo) & (snr < hi)
                        nsel = (ne[:, 0] >= lo) & (ne[:, 0] < hi) if ne.size else np.zeros(0, bool)
                        csel = (co[:, 0] >= lo) & (co[:, 0] < hi) if co.size else np.zeros(0, bool)
                        row["bins"].append({"snr_db": [lo, hi], "n_true": int(sel.sum()), "detection_rate": float(det[sel].mean()) if sel.any() else None,
                                            "rms_err_over_crb": [float(np.sqrt(np.mean(ne[nsel, i] ** 2))) if nsel.any() else None for i in (1, 2, 3)],
                                            "rms_err_over_reported_sd": [float(np.sqrt(np.mean(co[csel, i] ** 2))) if csel.any() else None for i in (1, 2, 3)]})
                    for name, m_t, m_e in (("isolated", iso, ne[:, 4] > 0.5), ("clustered", ~iso, ne[:, 4] < 0.5)):
                        strong_t = m_t & (snr >= 30.0)
                        strong_e = m_e & (ne[:, 0] >= 30.0)
                        row[name] = {"n_true_ge30dB": int(strong_t.sum()), "detection_rate_ge30dB": float(det[strong_t].mean()) if strong_t.any() else None,
                                     "rms_err_over_crb_ge30dB": [float(np.sqrt(np.mean(ne[strong_e, i] ** 2))) if strong_e.any() else None for i in (1, 2, 3)],
                                     "median_abs_err_over_crb_ge30dB": [float(np.median(np.abs(ne[strong_e, i]))) if strong_e.any() else None for i in (1, 2, 3)]}
                    rows.append(row)
                    print(f"{mount:8s} {gname:24s} {bw} {hw:5s}: false/snap {row['false_per_snapshot']:.2f}, true/snap {row['true_paths_per_snapshot']:.1f}, det "
                          + " ".join(f"{b['snr_db'][0]:g}:{b['detection_rate'] if b['detection_rate'] is None else round(b['detection_rate'], 2)}" for b in row["bins"])
                          + " | err/CRB " + " ".join(f"{b['snr_db'][0]:g}:{[None if v is None else round(v, 1) for v in b['rms_err_over_crb']]}" for b in row["bins"]), flush=True)
    # pooled summary
    pooled = {}
    for bw in ("100", "400"):
        for hw in ("ideal", "main"):
            rs = [r_ for r_ in rows if r_["bw"] == bw and r_["hardware"] == hw]
            cell = {"false_per_snapshot": float(np.mean([r_["false_per_snapshot"] for r_ in rs])), "bins": []}
            for i, (lo, hi) in enumerate(BINS):
                n = sum(r_["bins"][i]["n_true"] for r_ in rs)
                dr = sum((r_["bins"][i]["detection_rate"] or 0.0) * r_["bins"][i]["n_true"] for r_ in rs) / n if n else None
                cell["bins"].append({"snr_db": [lo, hi], "n_true": n, "detection_rate": dr})
            for name in ("isolated", "clustered"):
                n = sum(r_[name]["n_true_ge30dB"] for r_ in rs)
                cell[name] = {"n_true_ge30dB": n,
                              "detection_rate_ge30dB": sum((r_[name]["detection_rate_ge30dB"] or 0) * r_[name]["n_true_ge30dB"] for r_ in rs) / n if n else None,
                              "median_over_geometries_rms_err_over_crb": [float(np.median([r_[name]["rms_err_over_crb_ge30dB"][i] for r_ in rs
                                                                                        if r_[name]["rms_err_over_crb_ge30dB"][i] is not None] or [np.nan])) for i in range(3)],
                              "median_over_geometries_median_abs_err_over_crb": [float(np.median([r_[name]["median_abs_err_over_crb_ge30dB"][i] for r_ in rs
                                                                                               if r_[name]["median_abs_err_over_crb_ge30dB"][i] is not None] or [np.nan])) for i in range(3)]}
            pooled[f"{bw}|{hw}"] = cell
    out = {"definition": __doc__, "seed": seed, "dyn_range_db": dyn, "pfa": pfa, "config": ecfg, "rows": rows, "pooled": pooled,
           "runtime_s_per_snapshot": {"median": float(np.median(timing)), "max": float(np.max(timing))}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"extract_{a.tag}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    for kk, c in pooled.items():
        print(kk, f"false/snap {c['false_per_snapshot']:.2f}", {n: (c[n]["n_true_ge30dB"], c[n]["detection_rate_ge30dB"], c[n]["median_over_geometries_rms_err_over_crb"],
                                                                   c[n]["median_over_geometries_median_abs_err_over_crb"]) for n in ("isolated", "clustered")})
    print(f"runtime per snapshot: {out['runtime_s_per_snapshot']}")


if __name__ == "__main__":
    main()
