"""ADDED AFTER THE SECOND EXTERNAL REVIEW (2b): uncertainty-aware planner.

A5 (tuned per margin, results/M5/planner.json) runs by default. At every E2
decision epoch the planner may ADVANCE a handover or VETO A5 until the next
epoch, based on the expected outage reduction of switching now, estimated
from K posterior samples of the scene:
- tracks: the real map tracker (replayed from the stage-validated detection
  caches; detector budget and horizon H fixed per margin at the
  sensing-planner's values tuned on the tuning seeds); each confirmed
  track's state is sampled from N(state, alpha * P) with P the diagonal of
  the tracker's EKF covariance on x, y, vx, vy (pinned axes: ~0);
- UE position: N(fix, alpha * sigma_UE^2) per horizontal axis around the
  main runs' fix (sigma_UE = 1 m);
- per sample: predicted SNR = unblocked SNR minus the predicted model-B LoS
  loss (xapp/predict_torch.py); minimum outage over H (Viterbi as the
  receding-horizon planner, switches at 0.1 s epochs, tau_HO outage per
  switch) with the first action forced to stay or to switch;
- Delta = mean over samples of (outage if staying - outage if switching) [s];
  advance if Delta > theta; veto A5 if Delta < -theta.
Tuned per margin on the tuning seeds (101-105) over K {8, 32} (the first K
of the same 32 samples), alpha {1, 4, 16}, theta {0.01, 0.03, 0.1, 0.3} s
(24 points), objective as before; evaluated on the development seeds
(1001-1010; or the held-out seeds with S2C_EVAL_SET=heldout) and compared with A5 by the paired test (Wilcoxon p = the
significance test; t-based 95 % CI = effect size).
Check: the zero-noise sample reproduces the cached sensing-planner
predictions (within 1e-3 dB).

Also reported: the share of decision epochs (lane x report) at which the
advance or veto condition holds for the serving cell.

Writes results/M5/review2/uaplanner.json.
"""

from __future__ import annotations

import itertools
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

import run_m3 as R  # noqa: E402
from review2_calibrate import replay_rows  # noqa: E402
from review_b6_robust import run_robust  # noqa: E402
from xapp.metrics import lane_summary  # noqa: E402
from xapp.schemes import REASON, simulate  # noqa: E402
from run_m5_paired import paired  # noqa: E402
from run_m5_planner import per_job_outage  # noqa: E402
from sim.comm.linkbudget import sensing_overhead, snr_ref_db  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402

KS = (8, 32)
ALPHAS = (1.0, 4.0, 16.0)
THETAS = (0.01, 0.03, 0.1, 0.3)
OUT = ROOT / "results" / "M5" / "review2" / "uaplanner.json"


def plan_costs(bad, cur, tau: int, epoch: int, device: str = "cuda"):
    """Minimum outage steps over the horizon with the first action forced to stay / switch.

    Same recursion as run_m5_planner.plan_first_switch (which compares the
    two); returns (stay, switch) int arrays [N].
    """
    import torch

    b = torch.as_tensor(bad, device=device, dtype=torch.int32)
    c0 = torch.as_tensor(cur, device=device, dtype=torch.long)
    n, hs, _ = b.shape
    n_p = tau + 1
    big = 10**7
    idx = torch.arange(n, device=device)
    results = []
    for switch in (False, True):
        cost = torch.full((n, 2, n_p), big, device=device, dtype=torch.int32)
        c = (1 - c0) if switch else c0
        if switch and tau > 0:
            cost[idx, c, tau] = 1
        else:
            cost[idx, c, 0] = b[idx, 0, c]
        for h in range(1, hs):
            bh = b[:, h, :]
            new = torch.full_like(cost, big)
            for cc in range(2):
                for p in range(n_p):
                    q = max(p - 1, 0)
                    add = 1 if q > 0 else bh[:, cc]
                    new[:, cc, q] = torch.minimum(new[:, cc, q], cost[:, cc, p] + add)
            if h % epoch == 0:
                for cc in range(2):
                    best = cost[:, 1 - cc, :].min(dim=1).values
                    if tau > 0:
                        new[:, cc, tau] = torch.minimum(new[:, cc, tau], best + 1)
                    else:
                        new[:, cc, 0] = torch.minimum(new[:, cc, 0], best + bh[:, cc])
            cost = new
        results.append(cost.reshape(n, -1).min(dim=1).values.cpu().numpy())
    return results[0], results[1]


def pack_with_var(frames: list[list[dict]], sizes: dict) -> dict[str, np.ndarray]:
    """xapp.predict_torch.pack_tracks plus the EKF variances [R, K, 4] (x, y, vx, vy); pinned axes 0."""
    n_r = len(frames)
    k = max(1, max((len(f) for f in frames), default=1))
    state, size = np.zeros((n_r, k, 5)), np.ones((n_r, k, 3))
    var, valid = np.zeros((n_r, k, 4)), np.zeros((n_r, k), dtype=bool)
    for r, rows in enumerate(frames):
        for i, row in enumerate(rows):
            x, y, z, vx, vy = row["x_m"], row["y_m"], row["z_m"], row["vx_mps"], row["vy_mps"]
            v = list(row["var"])
            mode = row.get("mode", "free")
            if mode in ("lane", "sidewalk") and row.get("line_y_m") is not None:
                y, vy = float(row["line_y_m"]), 0.0
                v[1] = v[3] = 0.0
            state[r, i] = (x, y, z, vx, vy)
            size[r, i] = sizes["pedestrian"] if mode == "sidewalk" else sizes["bus"]
            var[r, i] = v
            valid[r, i] = True
    return {"state": state, "size": size, "valid": valid, "var": var}


def _replay(item: tuple) -> tuple:
    job, budget = item
    import torch

    torch.set_num_threads(1)
    return (job, budget), replay_rows(job, budget, covariance=True)


def delta_table(pred_samples: list[np.ndarray], unb_job: np.ndarray, taus, h_s, d, rs, tau, bw, rate_req, ovh) -> np.ndarray:
    """Per-sample (stay - switch) outage [s] for every report, UE and current cell: [S, R, U, 2]."""
    hs = int(round(h_s / R.DT_COMM))
    dtau = float(taus[1] - taus[0])
    seqs_all = []
    for pr_all in pred_samples:
        n_r, n_u = pr_all.shape[0], pr_all.shape[1]
        ks = (np.arange(n_r) * rs + d + 1)[:, None] + np.arange(hs)[None, :]
        ks_c = np.minimum(ks, unb_job.shape[0] - 1)
        lead = (ks - (np.arange(n_r) * rs)[:, None]) * R.DT_COMM
        x = np.clip(lead / dtau, 0.0, len(taus) - 1.0)
        i0 = np.minimum(np.floor(x).astype(np.int64), len(taus) - 2)
        fr = x - i0
        rr = np.arange(n_r)[:, None]
        bad = np.zeros((n_u, n_r, hs, 2), dtype=bool)
        for u in range(n_u):
            pr = np.clip(np.nan_to_num(pr_all[:, u], posinf=200.0), 0.0, 200.0)
            for c in range(2):
                loss = pr[rr, c, i0] * (1 - fr) + pr[rr, c, i0 + 1] * fr
                snr = unb_job[ks_c, u, c] - loss
                rate = bw * (1.0 - ovh) * np.minimum(np.log2(1.0 + 10.0 ** (snr / 10.0)), MAX_NR_SE)
                bad[u, :, :, c] = rate < rate_req
        seqs_all.append(np.repeat(bad.reshape(n_u * n_r, hs, 2), 2, axis=0))
    n_s = len(seqs_all)
    seqs = np.concatenate(seqs_all)
    cur = np.tile(np.array([0, 1]), seqs.shape[0] // 2)
    stay, sw = plan_costs(seqs, cur, tau, rs)
    return ((stay - sw) * R.DT_COMM).reshape(n_s, n_u, n_r, 2).transpose(0, 2, 1, 3)


def run_ua(jobs, a5p, snr_m, built, rw, bw, rate_req, orates, snr_req, adv_veto_fn, ovh, d: int, rs: int) -> tuple[list[dict], dict]:
    """review_b6_robust.run_robust plus the intervention share.

    Share = fraction of decision epochs (lane x report) at which the planner's
    advance or veto condition holds for the cell serving when the decision
    applies (report time + E2 delay).
    """
    a3c = [{"offset_db": 0.0, "hysteresis_db": 0.0, "ttt_s": a5p["ttt_s"]}]
    lanes, index = R.make_lanes(jobs, a3c, snr_m, built, "a3", {}, rw, None)
    n = len(index)
    lanes.scheme = np.full(n, REASON["a5"])
    lanes.a5_thr1 = np.full(n, snr_req + a5p["t1_db"])
    lanes.a5_thr2 = np.full(n, snr_req + a5p["t2_db"])
    adv, veto = adv_veto_fn(index)
    lanes.planner_mask = np.ones(n, dtype=bool)
    lanes.trigger = adv
    lanes.trigger_end_steps = np.zeros(adv.shape, dtype=np.int64)
    lanes.veto = veto
    lanes.hold_steps = np.zeros(n, dtype=np.int64)
    lanes.block_db = float("inf")
    lanes.overhead = np.full(n, ovh)
    sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
    rows = []
    for lane, (_, job, u) in enumerate(index):
        evs = [ev for ev in built[job]["events"] if ev["ue"] == u]
        s = lane_summary(sim, lane, evs, orates[job][:, u], R.DT_COMM, proactive_reasons=(REASON["xapp"],))
        s.update({"combo": 0, "job": job, "ue": u})
        rows.append(s)
    serving = sim["serving"]
    n_r = adv.shape[1]
    k = np.minimum(np.arange(n_r) * rs + d, serving.shape[1] - 1)
    srv = serving[:, k].astype(np.int64)  # [L, R]
    li, ri = np.meshgrid(np.arange(n), np.arange(n_r), indexing="ij")
    act_a = adv[li, ri, srv]
    act_v = veto[li, ri, srv]
    share = {"intervene": float((act_a | act_v).mean()), "advance": float(act_a.mean()), "veto": float(act_v.mean()), "epochs": int(act_a.size)}
    return rows, share


def main() -> None:
    import torch

    from xapp.predict_torch import predict_torch, ue_fixes

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    dev_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    jobs = tune_jobs + dev_jobs
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    built = R.build(jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, jobs, rw, bw, info["extra_loss_db"])
    b, pr = rw["budget"], rw["predict"]
    sigma_ue = float(pr["ue_sigma_m"])
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    wl = 299792458.0 / float(raw["carrier_hz"])
    fixed = {}
    for m in pl["margins"]:
        h = f"{m['sensing_planner_tuned_H']:.1f}"
        fixed[m["label"]] = {"H_s": float(h), "budget": int(m["sensing_planner"][h]["budget"])}
    budgets = sorted({v["budget"] for v in fixed.values()})
    with mp.get_context("spawn").Pool(4) as pool:
        tracks = dict(pool.map(_replay, [(job, bu) for job in jobs for bu in budgets], chunksize=1))
    print(f"replayed {len(tracks)} track sets", flush=True)
    from sim.scenes.motion import states_at
    from sim.scenes.traffic import prepare_scenario

    geo = {}
    for job in jobs:
        sc = prepare_scenario(raw, seed=job[0], mount=job[1], density=job[2], duration_s=60.0, dt_s=0.1)
        n_r = int(built[job]["data"]["pred_2"].shape[0])
        t_r = (np.arange(n_r) * rs) * R.DT_COMM
        ue = np.array([[states_at(sc, float(t))[x["name"]]["position_m"] for x in sc["ues"]] for t in t_r])
        uev = np.array([[states_at(sc, float(t))[x["name"]]["velocity_mps"] for x in sc["ues"]] for t in t_r])
        sizes = {k: (float(v["length_m"]), float(v["width_m"]), float(v["height_m"])) for k, v in sc["blocker_kinds"].items()}
        geo[job] = {"fix": ue_fixes(ue, uev, sigma_ue, job[0] * 17 + 3), "vel": uev,
                    "oru": np.array([o["position_m"] for o in sc["orus"]], dtype=np.float64), "sizes": sizes}
    taus = built[jobs[0]]["data"]["taus"]
    packed = {(job, bu): pack_with_var(tracks[(job, bu)], geo[job]["sizes"]) for job in jobs for bu in budgets}
    check = 0.0
    for job in jobs:
        for bu in budgets:
            pk = packed[(job, bu)]
            p0 = predict_torch(pk, geo[job]["fix"], geo[job]["vel"], geo[job]["oru"], taus, wl)
            check = max(check, float(np.max(np.abs(np.nan_to_num(p0, posinf=1e6) - np.nan_to_num(built[job]["data"][f"pred_{bu}"], posinf=1e6)))))
    if check > 1e-3:
        raise SystemExit(f"zero-noise predictions differ from the cached sensing-planner predictions by {check} dB")
    print(f"zero-noise check: max |diff| {check:.2e} dB", flush=True)
    # per (alpha, budget, job): 32 sampled prediction tensors, reduced per margin to the mean Delta over the first K samples [R, U, 2]
    unb = {mi: {job: snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - info["extra_loss_db"][mi]
                for job in jobs} for mi in range(len(info["labels"]))}
    deltas: dict[tuple, np.ndarray] = {}
    kmax = max(KS)
    for alpha in ALPHAS:
        t0 = time.perf_counter()
        for job in jobs:
            for bu in budgets:
                pk = packed[(job, bu)]
                rng = np.random.default_rng([job[0], R.MOUNTS.index(job[1]), R.DENSITIES.index(job[2]), bu, int(alpha * 10)])
                samples = []
                for _ in range(kmax):
                    st = pk["state"].copy()
                    st[..., [0, 1, 3, 4]] += np.sqrt(alpha * pk["var"]) * rng.normal(0.0, 1.0, pk["var"].shape)
                    fix = geo[job]["fix"].copy()
                    fix[..., :2] += np.sqrt(alpha) * sigma_ue * rng.normal(0.0, 1.0, fix[..., :2].shape)
                    samples.append(predict_torch({"state": st, "size": pk["size"], "valid": pk["valid"]}, fix, geo[job]["vel"], geo[job]["oru"], taus, wl))
                for mi, lab in enumerate(info["labels"]):
                    if fixed[lab]["budget"] != bu:
                        continue
                    dt = delta_table(samples, unb[mi][job], taus, fixed[lab]["H_s"], d, rs, tau, bw, rate_req, ovh)
                    for k in KS:
                        deltas[(k, alpha, mi, job)] = dt[:k].mean(0).astype(np.float32)  # [R, U, 2]
        print(f"alpha {alpha}: sampled and planned in {time.perf_counter() - t0:.0f} s", flush=True)
    combos = list(itertools.product(KS, ALPHAS, THETAS))
    from seedsets import eval_set

    out: dict[str, Any] = {"definition": __doc__, "eval_set": eval_set(), "seeds": [int(x) for x in seeds["evaluation"]],
                           "grid": {"K": KS, "alpha": ALPHAS, "theta_s": THETAS}, "fixed": fixed, "zero_noise_check_db": check, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        t0 = time.perf_counter()
        snr_t = {j: snr_all[j][mi] for j in tune_jobs}
        snr_e = {j: snr_all[j][mi] for j in dev_jobs}
        or_t = R.stateless_rows(tune_jobs, snr_t, built, "oracle", bw, rate_req)[1]
        or_e = R.stateless_rows(dev_jobs, snr_e, built, "oracle", bw, rate_req)[1]
        a5p = pl["margins"][mi]["a5"]["params"]

        def fn(index, k, alpha, theta):
            dm = [deltas[(k, alpha, mi, job)][:, u, :] for _, job, u in index]
            dm = np.stack(dm)
            return dm > theta, dm < -theta

        objs = {}
        for k, alpha, theta in combos:
            rows = run_robust(tune_jobs, a5p, snr_t, built, rw, bw, rate_req, or_t, info["snr_req_db"], lambda ix: fn(ix, k, alpha, theta), ovh)
            objs[(k, alpha, theta)] = R.objective(rows, 1)[0]
        best = min(combos, key=lambda c: (objs[c][0], objs[c][1], c))
        k, alpha, theta = best
        rows, share = run_ua(dev_jobs, a5p, snr_e, built, rw, bw, rate_req, or_e, info["snr_req_db"], lambda ix: fn(ix, k, alpha, theta), ovh, d, rs)
        agg = R.aggregate(rows)
        pj = per_job_outage(rows)
        res = {"label": lab, "tuned": {"K": k, "alpha": alpha, "theta_s": theta, **fixed[lab]}, "tuning_objective": objs[best],
               "eval": {kk: agg[kk] for kk in ("outage_req_s_per_min", "ho_per_min", "ping_pong", "precision")}, "per_job": pj,
               "vs_a5": paired(pj, pl["margins"][mi]["a5"]["per_job"]), "intervention_share": share, "wall_s": time.perf_counter() - t0}
        out["margins"].append(res)
        v = res["vs_a5"]
        print(f"{lab}: UA planner {agg['outage_req_s_per_min']['mean']:.3f} (K {k}, alpha {alpha}, theta {theta}) | diff {v['mean_diff']:+.4f} "
              f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] p={v['wilcoxon_p_two_sided']:.3g} | {res['wall_s']:.0f} s", flush=True)
    out["wall_s"] = time.perf_counter() - clock
    OUT.write_text(json.dumps(out, indent=1, default=R._json) + "\n")
    print(f"wrote {OUT} in {out['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
