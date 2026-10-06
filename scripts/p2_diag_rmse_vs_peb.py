"""Paper 2 DIAGNOSTIC (external review of the draft): single-epoch estimator RMSE vs the single-epoch PEB.

Six fixed UE geometries from the held-out set 3001-3010 (job 3001, low density, UE 0):
per mount (lamppost, facade) the both-LoS epochs whose UE x is closest to -20 m and to
+20 m, and the blocked epoch (at least one O-RU with LoS model-B loss >= 10 dB) at the
median index of the blocked epochs. For each geometry, bandwidth (100, 400 MHz) and
hardware (ideal: sigma_sync 0, sigma_phi 0; main: 1 ns, 2 deg), 500 independent draws of
noise, O-RU time offsets, element phase errors and the UE clock bias (TDoA) are passed
through the FROZEN single-epoch estimator (synthesis + per-O-RU measurement of
sim/positioning/estimator.py, WLS fix of the v1 fusion with its tuned parameters: the
lowest-cost fix over all gated O-RUs; no EKF, no tracking). Blocked LoS: variant (b)
(diffracted detour in the signal; geometric information removed in the bound).
Reported per cell: RMSE, median and p90 of the fix error, its split into the bias (distance
of the mean fix to the true UE) and the random spread (rms distance to the mean fix), and the single-epoch
LoS-only PEB of the same configuration (results/P2/peb/<set>; the PEB is computed from
the Fisher information of ONE SRS symbol, i.e. single epoch). Writes
results/P2/diag_rmse_vs_peb_<set>.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

N_DRAWS = 500


def main() -> None:
    import torch

    import p2_estimate as P
    from p2_peb_report import idx
    from p2_seeds import eval_tag
    from sim.positioning.array import element_positions
    from sim.positioning.estimator import measure_torch, noise_model, synth_torch, wls_candidates
    from sim.scenes.config import load_yaml

    tag = eval_tag()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    tuned = json.loads((ROOT / "results" / "P2" / "est_tuned.json").read_text())["tuned"]
    seed = {"heldout2": 3001, "heldout": 2001, "dev": 1001}[tag]
    df = 15e3 * 2 ** int(p2cfg["numerology"])
    link = p2cfg["link"]
    out = {"definition": __doc__, "set": tag, "n_draws": N_DRAWS, "cells": []}
    for mount in ("lamppost", "facade"):
        job = (seed, mount, "low")
        jp = P.job_paths(job, raw, p2cfg)
        u = 0
        nb = jp["blocked"][:, u, :].any(-1)
        x = jp["ue"][:, u, 0]
        los_idx = np.flatnonzero(~nb)
        blk_idx = np.flatnonzero(nb)
        picks = [("both LoS, x near -20 m", int(los_idx[np.argmin(np.abs(x[los_idx] + 20))])), ("both LoS, x near +20 m", int(los_idx[np.argmin(np.abs(x[los_idx] - 20))]))]
        if blk_idx.size:
            picks.append(("blocked", int(blk_idx[len(blk_idx) // 2])))
        peb_all = np.load(ROOT / "results" / "P2" / "peb" / tag / f"{mount}_low_{seed}.npz")["peb"]
        r = element_positions(jp["wl"])
        for gname, t in picks:
            for bw in ("100", "400"):
                n_sc = int(p2cfg["bandwidths"][bw]["n_sc"])
                f = (np.arange(n_sc) - (n_sc - 1) / 2.0) * df
                noise_p = P.K_B * P.T0 * 10 ** (float(link["oru_noise_figure_db"]) / 10.0) * n_sc * df
                p_tx = 10 ** ((float(link["ue_tx_power_dbm"]) - 30.0) / 10.0)
                beta = jp["a"][t, u] * math.sqrt(p_tx / noise_p)  # [C, P]
                tau = jp["tau_b_ns"][t, u]
                uu = jp["u_b"][t, u]
                params = tuned[f"{bw}|tdoa"]["params"]
                for hw, (ss, ph) in (("ideal", (0.0, 0.0)), ("main", (1.0, 2.0))):
                    rng = np.random.default_rng([seed, ["lamppost", "facade"].index(mount), t, int(bw), int(ss * 10), int(ph)])
                    gen = torch.Generator(device="cuda")
                    gen.manual_seed(int(rng.integers(1 << 31)))
                    C = beta.shape[0]
                    delta = ss * rng.standard_normal((N_DRAWS, C))
                    b_clk = rng.uniform(-50.0, 50.0, N_DRAWS)
                    psi = math.radians(ph) * rng.standard_normal((N_DRAWS, C, 64))
                    meas = {k: np.zeros((N_DRAWS, C)) for k in ("tau_ns", "uy", "uz", "snr", "ratio_db")}
                    for c in range(C):
                        for lo in range(0, N_DRAWS, 50):
                            n = min(50, N_DRAWS - lo)
                            bt = torch.as_tensor(np.broadcast_to(beta[c], (n, beta.shape[1])).copy(), device="cuda")
                            tt = torch.as_tensor(tau[c][None, :] + (delta[lo:lo + n, c] + b_clk[lo:lo + n])[:, None], device="cuda")
                            ut = torch.as_tensor(np.broadcast_to(uu[c], (n,) + uu[c].shape).copy(), device="cuda")
                            Y = synth_torch(bt, tt, ut, f, r, jp["wl"], psi=torch.as_tensor(psi[lo:lo + n, c], device="cuda"), gen=gen)
                            m = measure_torch(Y, f, jp["wl"])
                            for k in meas:
                                meas[k][lo:lo + n, c] = m[k]
                    meas["tau_ns"] = np.where(meas["tau_ns"] > 0.5e9 / df, meas["tau_ns"] - 1e9 / df, meas["tau_ns"])
                    oru = np.broadcast_to(jp["oru"][t], (N_DRAWS, C, 3))
                    use, st, su = noise_model(meas, params, df, n_sc)
                    cand = wls_candidates(meas, oru, use, "tdoa", st, su, P.Z_UE)
                    k = np.argmin(cand["cost"], -1)
                    fix = cand["xy"][np.arange(N_DRAWS), k]
                    err = np.linalg.norm(fix - jp["ue"][t, u, :2], axis=-1)
                    err = np.where(np.isfinite(err), err, np.nan)
                    peb = float(peb_all[(t, u) + idx(info="los", bw=bw, timing="tdoa", sync=ss, phi=ph, blocked="biased")])
                    okf = np.isfinite(fix).all(-1)
                    bias = float(np.linalg.norm(np.nanmean(fix[okf], 0) - jp["ue"][t, u, :2])) if okf.any() else np.nan
                    spread = float(np.sqrt(np.nanmean(np.sum((fix[okf] - np.nanmean(fix[okf], 0)) ** 2, -1)))) if okf.any() else np.nan
                    cell = {"mount": mount, "geometry": gname, "epoch": t, "ue_xy": jp["ue"][t, u, :2].tolist(), "blocked_orus": jp["blocked"][t, u].tolist(),
                            "bias_m": bias, "spread_rms_m": spread,
                            "bw": bw, "hardware": hw, "rmse_m": float(np.sqrt(np.nanmean(err ** 2))), "median_m": float(np.nanmedian(err)),
                            "p90_m": float(np.nanpercentile(err, 90)), "peb_los_m": peb, "rmse_to_peb": float(np.sqrt(np.nanmean(err ** 2)) / peb),
                            "share_no_fix": float(np.mean(~np.isfinite(err)))}
                    out["cells"].append(cell)
                    print(f"{mount:8s} {gname:24s} {bw} MHz {hw:5s}: RMSE {cell['rmse_m']:.3f} m, median {cell['median_m']:.3f}, p90 {cell['p90_m']:.3f}, "
                          f"PEB {peb:.4f} m, RMSE/PEB {cell['rmse_to_peb']:.1f}, bias {bias:.3f} m, spread {spread:.4f} m (spread/PEB {spread / peb:.1f})", flush=True)
    (ROOT / "results" / "P2" / f"diag_rmse_vs_peb_{tag}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
