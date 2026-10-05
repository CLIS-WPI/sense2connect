"""Paper 2 (P2-M1, Addendum A): position error bound per UE and epoch for every variant.

Per job (seed, mount, density) and 0.1 s epoch, both UEs, both O-RUs:
paths from the paper-1 comm geometry cache (read-only), complex path
coefficients from results/P2/cache (scripts/p2_trace.py), model-B loss per
path (sim/positioning/blockage.py), analytic image-method geometry and
Jacobians on the known planes (sim/positioning/geometry.py), Fisher
information (sim/positioning/fim.py).
Link (configs/p2.yaml): UL SRS 23 dBm over the N active subcarriers of one
symbol, noise kT F df per subcarrier and element (F = 7 dB); per-element
amplitude beta_p = sqrt(P_tx / (kT F N df)) a_p 10^(-L_B,p / 20).
Variants (stored as one array): bandwidth {100, 200, 400 MHz} x SNR scale
{1, 10} (limit analysis) x information {LoS-only, map-aided} x blocked LoS
{kept, biased} x timing {toa, tdoa, aoa} x sigma_sync {0, 0.3, 1, 3} ns x
sigma_phi {0, 2, 5} deg. "Blocked" = model-B LoS loss >= 10 dB on that O-RU.
Writes results/P2/peb/<set>/<mount>_<density>_<seed>.npz (peb float32
[T, U, bw, snr, info, blocked, timing, sync, phi], LoS loss per O-RU, UE
position) and checks the GPU PEB against the NumPy reference on a sample.
Run: python scripts/p2_peb.py [--set dev|tuning|heldout] (S2C_EVAL_SET as in paper 1).
"""

from __future__ import annotations

import argparse
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

K_B = 1.380649e-23
T0 = 290.0
C0 = 299_792_458.0
INFO = ("los", "map")
BLOCKED = ("kept", "biased")
TIMING = ("toa", "tdoa", "aoa")
SNR_SCALE = (1.0, 10.0)
N_THETA = 5  # x, y, delta_0, delta_1, b


def load_cfg() -> dict:
    from sim.scenes.config import load_yaml

    return load_yaml(ROOT / "configs" / "p2.yaml")


def job_inputs(job: tuple, raw: dict, cfg: dict) -> dict:
    """Per-path quantities [T, U, C, P] of one job."""
    from sim.positioning.blockage import path_losses_db
    from sim.positioning.geometry import bounce_planes, path_geometry
    from sim.scenes.traffic import prepare_scenario

    seed, mount, density = job
    gdir = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    with np.load(gdir / "comm_geometry.npz") as g:
        geom = {k: g[k] for k in g.files}
    meta = json.loads((gdir / "comm_geometry_meta.json").read_text())
    wl = float(meta["wavelength_m"])
    with np.load(ROOT / "results" / "P2" / "cache" / mount / density / f"seed_{seed}" / "p2_paths.npz") as q:
        a_center = q["a_center"]
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    bl = path_losses_db(geom, sc, wl)
    P = geom["points_m"]
    n = geom["n_points"].astype(np.int64)
    ue = np.broadcast_to(geom["ue_position_m"][:, :, None, None, :], P.shape[:-2] + (3,))
    oru = np.broadcast_to(geom["oru_position_m"][:, None, :, None, :], P.shape[:-2] + (3,))
    N, A, V = bounce_planes(P, n, cfg["known_planes"])
    gm = path_geometry(oru, ue, N, A, V)
    valid = geom["path_class"] >= 0
    loss = np.where(valid, bl["loss_db"], np.inf)
    amp = np.where(np.isfinite(loss), 10.0 ** (-np.nan_to_num(loss, posinf=0.0) / 20.0), 0.0)
    cls = geom["path_class"].astype(np.int64)
    los_loss = np.where(cls == 0, loss, -1.0).max(-1)  # [T, U, C]
    los_loss = np.where(los_loss < 0, np.inf, los_loss)
    return {"wl": wl, "a": np.where(valid, a_center, 0.0) * amp, "cls": cls, "geo": gm, "los_loss": los_loss,
            "ue": geom["ue_position_m"], "valid": valid & (amp > 0)}


def build_job(job: tuple, raw: dict, cfg: dict, device: str = "cuda") -> dict:
    import torch

    from sim.positioning.array import element_positions
    from sim.positioning.fim import calibrated_efim, geometric_efim, gram_torch, peb_from_theta, theta_information

    inp = job_inputs(job, raw, cfg)
    wl = inp["wl"]
    r = element_positions(wl)
    link = cfg["link"]
    T, U, C, P = inp["a"].shape
    g = inp["geo"]
    tau_ns = torch.as_tensor(g["tau"] * 1e9, device=device)
    az = torch.as_tensor(g["az"], device=device)
    el = torch.as_tensor(g["el"], device=device)
    H = np.zeros((T, U, C, 3 * P, N_THETA))  # theta = (x, y, delta_0, delta_1, b)
    for c in range(C):
        for p in range(P):
            H[:, :, c, 3 * p, 0:2] = g["dtau"][:, :, c, p] * 1e9  # ns / m
            H[:, :, c, 3 * p, 2 + c] = 1.0
            H[:, :, c, 3 * p, 4] = 1.0
            H[:, :, c, 3 * p + 1, 0:2] = g["daz"][:, :, c, p]
            H[:, :, c, 3 * p + 2, 0:2] = g["del"][:, :, c, p]
    Ht = torch.as_tensor(H, device=device)
    li = np.argmax(np.where(inp["cls"] == 0, 1.0, 0.0) * 1e6 + np.abs(inp["a"]) ** 2, -1)
    blocked = inp["los_loss"] >= float(cfg["blockage"]["blocked_los_db"])  # [T, U, C]
    is_los = torch.as_tensor(inp["cls"] == 0, device=device)
    is_nlos = torch.as_tensor((inp["cls"] > 0), device=device)
    absent = torch.as_tensor(~inp["valid"], device=device)
    blk = torch.as_tensor(blocked, device=device)
    sync = [float(x) for x in cfg["hardware"]["sigma_sync_ns"]]
    phi = [math.radians(float(x)) for x in cfg["hardware"]["sigma_phi_deg"]]
    bws = list(cfg["bandwidths"].items())
    df = 15e3 * 2 ** int(cfg["numerology"])
    out = np.full((T, U, len(bws), len(SNR_SCALE), len(INFO), len(BLOCKED), len(TIMING), len(sync), len(phi)), np.inf, dtype=np.float32)
    snr_db = np.zeros((T, U, C, len(bws)))
    betas = {}
    for bi, (_bw, spec) in enumerate(bws):
        n_sc = int(spec["n_sc"])
        f = (np.arange(n_sc) - (n_sc - 1) / 2.0) * df
        noise = K_B * T0 * 10 ** (float(link["oru_noise_figure_db"]) / 10.0) * n_sc * df
        p_tx = 10 ** ((float(link["ue_tx_power_dbm"]) - 30.0) / 10.0) * int(link.get("srs_symbols_per_epoch", 1))
        beta_np = inp["a"] * math.sqrt(p_tx / noise)  # per element and subcarrier, / sigma
        betas[bi] = (beta_np, f)
        beta = torch.as_tensor(beta_np, device=device)
        snr_db[..., bi] = 10 * np.log10(np.maximum(np.take_along_axis(np.abs(inp["a"]) ** 2, li[..., None], -1)[..., 0] * p_tx / noise * n_sc * 64, 1e-30))
        Ks = []
        for lo in range(0, T, 25):
            Jf = gram_torch(beta[lo:lo + 25], tau_ns[lo:lo + 25], az[lo:lo + 25], el[lo:lo + 25], f, r, wl)
            Ks.append(geometric_efim(Jf, P))
        K = torch.cat(Ks, 0)  # [T, U, C, 3P + 64, 3P + 64]
        for ii, info in enumerate(INFO):
            for bj, bv in enumerate(BLOCKED):
                for ti, tm in enumerate(TIMING):
                    free_path = absent.clone()
                    if info == "los":
                        free_path |= is_nlos
                    if bv == "biased":
                        free_path |= is_los & blk[..., None]
                    free = free_path[..., None].expand(*free_path.shape, 3).clone()
                    if tm == "aoa":
                        free[..., 0] = True
                    free = free.reshape(T, U, C, 3 * P)
                    for (pi_, ph), (si, sc_) in itertools.product(enumerate(phi), enumerate(SNR_SCALE)):
                        J0 = theta_information(calibrated_efim(K * sc_, free, ph, 3 * P), Ht)  # SNR scales the data, not the priors
                        for ki, ss in enumerate(sync):
                            prior = torch.zeros((T, U, N_THETA, N_THETA), dtype=torch.float64, device=device)
                            present = torch.zeros((T, U, N_THETA), dtype=torch.bool, device=device)
                            if tm != "aoa" and ss > 0:
                                for c in range(C):
                                    present[..., 2 + c] = True
                                    prior[..., 2 + c, 2 + c] = 1.0 / ss ** 2
                            if tm == "tdoa":
                                present[..., 4] = True
                            pe = peb_from_theta(J0, prior, present)
                            out[:, :, bi, si, ii, bj, ti, ki, pi_] = pe.cpu().numpy()
    return {"peb": out, "los_loss": inp["los_loss"], "ue": inp["ue"], "snr_los_db": snr_db, "H": H, "absent": ~inp["valid"],
            "cls": inp["cls"], "blocked": blocked, "betas": betas, "geo": g, "wl": wl, "P": P}


def reference_check(res: dict, cfg: dict, rng: np.random.Generator, n: int = 12) -> float:
    """Max relative difference GPU vs an independent NumPy chain (explicit channel synthesis, gram_numpy + peb_numpy)."""
    from sim.positioning.array import element_positions
    from sim.positioning.fim import gram_numpy, peb_numpy

    sync = [float(x) for x in cfg["hardware"]["sigma_sync_ns"]]
    phi = [math.radians(float(x)) for x in cfg["hardware"]["sigma_phi_deg"]]
    r = element_positions(res["wl"])
    g = res["geo"]
    T, U = res["peb"].shape[:2]
    P = res["P"]
    worst = 0.0
    for _ in range(n):
        t, u = int(rng.integers(T)), int(rng.integers(U))
        bi = int(rng.integers(res["peb"].shape[2]))
        si, ii, bj, ti, ki, pi_ = (int(rng.integers(s)) for s in res["peb"].shape[3:])
        beta_np, f = res["betas"][bi]
        C = beta_np.shape[2]
        Jf = np.stack([gram_numpy(beta_np[t, u, c], g["tau"][t, u, c] * 1e9, g["az"][t, u, c], g["el"][t, u, c], f, r, res["wl"]) for c in range(C)])
        free = np.zeros((C, 3 * P), dtype=bool)
        for c in range(C):
            for p in range(P):
                fr = res["absent"][t, u, c, p] or (INFO[ii] == "los" and res["cls"][t, u, c, p] > 0) or \
                    (BLOCKED[bj] == "biased" and res["cls"][t, u, c, p] == 0 and res["blocked"][t, u, c])
                free[c, 3 * p:3 * p + 3] = fr
                if TIMING[ti] == "aoa":
                    free[c, 3 * p] = True
        prior = np.zeros((N_THETA, N_THETA))
        present = np.zeros(N_THETA, dtype=bool)
        present[:2] = True
        if TIMING[ti] != "aoa" and sync[ki] > 0:
            for c in range(C):
                present[2 + c] = True
                prior[2 + c, 2 + c] = 1.0 / sync[ki] ** 2
        if TIMING[ti] == "tdoa":
            present[4] = True
        ref = peb_numpy(Jf * SNR_SCALE[si], P, res["H"][t, u], free, phi[pi_], prior, present)
        got = float(res["peb"][t, u, bi, si, ii, bj, ti, ki, pi_])
        if math.isinf(ref) or math.isinf(got):
            if math.isinf(ref) != math.isinf(got):
                worst = max(worst, math.inf)
            continue
        worst = max(worst, abs(got - ref) / ref)
    return worst


def main() -> None:
    import torch

    from p2_seeds import eval_tag
    from p2_seeds import load as load_seeds
    from sim.scenes.config import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="tuning | evaluation (default: evaluation = dev or held-out by S2C_EVAL_SET)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_cfg()
    seeds = load_seeds()
    which = args.set or "evaluation"
    tag = "tuning" if which == "tuning" else eval_tag()
    jobs = [(int(s), m, d) for s in seeds[which] for m in ("lamppost", "facade") for d in ("low", "high")]
    if args.limit:
        jobs = jobs[: args.limit]
    out_dir = ROOT / "results" / "P2" / "peb" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    clock = time.perf_counter()
    worst_ref = 0.0
    for job in jobs:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        res = build_job(job, raw, cfg)
        worst_ref = max(worst_ref, reference_check(res, cfg, rng))
        seed, mount, density = job
        np.savez(out_dir / f"{mount}_{density}_{seed}.npz", peb=res["peb"], los_loss_db=res["los_loss"], ue=res["ue"], snr_los_db=res["snr_los_db"])
        print(f"{job}: {time.perf_counter() - t0:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, ref check {worst_ref:.1e}, "
              f"median PEB (400 MHz, map, biased, tdoa, ideal HW) {np.median(res['peb'][:, :, 2, 0, 1, 1, 1, 0, 0]):.3f} m", flush=True)
    meta = {"set": tag, "jobs": [list(j) for j in jobs], "axes": {"bw": list(cfg["bandwidths"]), "snr_scale": SNR_SCALE, "info": INFO, "blocked": BLOCKED,
            "timing": TIMING, "sigma_sync_ns": cfg["hardware"]["sigma_sync_ns"], "sigma_phi_deg": cfg["hardware"]["sigma_phi_deg"]},
            "gpu_vs_numpy_max_rel": worst_ref, "wall_s": time.perf_counter() - clock}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta)[:400])


if __name__ == "__main__":
    main()
