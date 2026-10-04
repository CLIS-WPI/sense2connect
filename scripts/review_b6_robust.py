"""ADDED AFTER EXTERNAL REVIEW (B6): robust planner, a post-review baseline.

A5 (tuned per margin, results/M5/planner.json) runs by default. At every E2
decision epoch (report time + E2 delay) a receding-horizon planner (the
sensing-planner model of scripts/run_m5_planner.py: unblocked SNR along the
UE path minus the predicted model-B LoS loss, Viterbi over H, sensing
overhead charged) may only
- ADVANCE a handover: its plan switches now AND the serving cell is
  predicted >= 10 dB blocked within H;
- VETO A5 until the next epoch: its plan stays AND the other cell is
  predicted >= 10 dB blocked within H;
where the prediction uses ONLY confirmed map-tracker tracks older than T_c
(final M2 map tracker replayed from the stage-validated detection caches;
detector budget 2 or 4; UE fix with sigma 1 m, same draws as the
sensing-planner). Tuned on tuning seeds per margin over T_c {0.3, 0.5, 1.0,
2.0} s x H {0.5, 1, 2} s x budget {2, 4} (24 points), objective as before;
evaluated on evaluation seeds and compared with A5 by the paired test.
Check: with T_c = 0 the predictions match the cached M3 predictions (within
1e-3 dB; the replay's BLAS thread count differs from the cached build).

Writes results/M5/review_b6.json.
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
from run_m5_paired import paired  # noqa: E402
from run_m5_planner import per_job_outage, plan_first_switch  # noqa: E402
from sim.comm.linkbudget import sensing_overhead, snr_ref_db  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.metrics import lane_summary  # noqa: E402
from xapp.schemes import REASON, simulate  # noqa: E402

T_C = (0.3, 0.5, 1.0, 2.0)
HORIZONS = (0.5, 1.0, 2.0)
BUDGETS = (2, 4)


def _replay(item: tuple) -> tuple:
    job, = item
    import torch

    torch.set_num_threads(1)
    from sim.sensing.provenance import require_detection_stage
    from xapp.tracks import load_detections, replay_map

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    seed, mount, density = job
    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    det = load_detections(directory)
    require_detection_stage(det)
    spec = raw["sensing_radar"]
    out = {}
    for budget in BUDGETS:
        dd = cfg["sensing"]["budgets"][budget]
        frames = replay_map(det, train=int(dd["train"]), pfa=float(dd["pfa"]), eps_m=float(dd["eps_m"]), ghost_association_m=float(dd["ghost_association_m"]),
                            walls=[float(v) for v in spec["wall_y_m"]], spec=spec, lanes=list(raw["lanes"]), sidewalks=list(raw["sidewalks"]), tuned=cfg["sensing"]["map_tracker"])
        out[budget] = [[{k: row[k] for k in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "confirmed", "mode", "line_y_m", "age_s")} for row in fr if row["confirmed"]] for fr in frames]
    return job, out


def tables(index, preds, unb, taus, h_s, d, rs, tau, bw, rate_req, ovh) -> tuple[np.ndarray, np.ndarray]:
    """Advance [L, R, 2] and veto [L, R, 2] tables (current cell = last axis)."""
    hs = int(round(h_s / R.DT_COMM))
    dtau = float(taus[1] - taus[0])
    adv, veto = [], []
    for _, job, u in index:
        pr = np.clip(np.nan_to_num(preds[job][:, u, :, :], posinf=200.0), 0.0, 200.0)
        n_r = pr.shape[0]
        un = unb[job][:, u, :]
        ks = (np.arange(n_r) * rs + d + 1)[:, None] + np.arange(hs)[None, :]
        ks_c = np.minimum(ks, un.shape[0] - 1)
        lead = (ks - (np.arange(n_r) * rs)[:, None]) * R.DT_COMM
        x = np.clip(lead / dtau, 0.0, len(taus) - 1.0)
        i0 = np.minimum(np.floor(x).astype(np.int64), len(taus) - 2)
        fr = x - i0
        rr = np.arange(n_r)[:, None]
        bad = np.zeros((n_r, hs, 2), dtype=bool)
        blocked = np.zeros((n_r, 2), dtype=bool)
        for c in range(2):
            loss = pr[rr, c, i0] * (1 - fr) + pr[rr, c, i0 + 1] * fr
            blocked[:, c] = (loss >= 10.0).any(axis=1)
            snr = un[ks_c, c] - loss
            rate = bw * (1.0 - ovh) * np.minimum(np.log2(1.0 + 10.0 ** (snr / 10.0)), MAX_NR_SE)
            bad[:, :, c] = rate < rate_req
        sw = plan_first_switch(np.repeat(bad, 2, axis=0), np.tile(np.array([0, 1]), n_r), tau, rs).reshape(n_r, 2)
        a = sw & blocked  # current cell c predicted blocked
        v = ~sw & blocked[:, ::-1]  # other cell predicted blocked
        adv.append(a)
        veto.append(v)
    return np.stack(adv), np.stack(veto)


def run_robust(jobs, a5p, snr_m, built, rw, bw, rate_req, orates, snr_req, adv_veto_fn, ovh) -> list[dict]:
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
    return rows


def main() -> None:
    import torch

    from xapp.predict_torch import pack_tracks, predict_torch, ue_fixes

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    jobs = tune_jobs + eval_jobs
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    built = R.build(jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, jobs, rw, bw, info["extra_loss_db"])
    b = rw["budget"]
    pr = rw["predict"]
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    wl = 299792458.0 / float(raw["carrier_hz"])
    with mp.get_context("spawn").Pool(4) as pool:
        tracks = dict(pool.map(_replay, [(job,) for job in jobs], chunksize=1))
    print(f"replayed tracks for {len(tracks)} jobs", flush=True)
    from sim.scenes.traffic import prepare_scenario
    from sim.scenes.motion import states_at

    geo = {}
    for job in jobs:
        sc = prepare_scenario(raw, seed=job[0], mount=job[1], density=job[2], duration_s=60.0, dt_s=0.1)
        n_r = int(built[job]["data"]["pred_2"].shape[0])
        t_r = (np.arange(n_r) * rs) * R.DT_COMM  # same time expression as the cached M3 build (comm_times grid)
        ue = np.array([[states_at(sc, float(t))[x["name"]]["position_m"] for x in sc["ues"]] for t in t_r])
        uev = np.array([[states_at(sc, float(t))[x["name"]]["velocity_mps"] for x in sc["ues"]] for t in t_r])
        sizes = {k: (float(v["length_m"]), float(v["width_m"]), float(v["height_m"])) for k, v in sc["blocker_kinds"].items()}
        geo[job] = {"fix": ue_fixes(ue, uev, float(pr["ue_sigma_m"]), job[0] * 17 + 3), "vel": uev,
                    "oru": np.array([o["position_m"] for o in sc["orus"]], dtype=np.float64), "sizes": sizes}
    taus = built[jobs[0]]["data"]["taus"]
    preds: dict[tuple, dict] = {}
    out_check: dict[int, float] = {}
    for budget in BUDGETS:
        for tc in (0.0,) + T_C:
            preds[(budget, tc)] = {}
            for job in jobs:
                rows = [[t for t in fr if t["age_s"] >= tc - 1e-9] for fr in tracks[job][budget]]
                pk = pack_tracks(rows, {"bus": geo[job]["sizes"]["bus"], "pedestrian": geo[job]["sizes"]["pedestrian"]})
                preds[(budget, tc)][job] = predict_torch(pk, geo[job]["fix"], geo[job]["vel"], geo[job]["oru"], taus, wl)
        check = max(float(np.nanmax(np.abs(np.nan_to_num(preds[(budget, 0.0)][j], posinf=1e6) - np.nan_to_num(built[j]["data"][f"pred_{budget}"], posinf=1e6)))) for j in jobs)
        # The tracker replay here runs with 1 BLAS thread per worker, the cached M3 build with 2; the EKF's
        # floating-point summation order differs, which near a knife edge moves the loss by ~1e-5 dB.
        if check > 1e-3:
            raise SystemExit(f"T_c = 0 predictions differ from the cached M3 predictions (budget {budget}, max {check} dB)")
        out_check[budget] = check
    print(f"T_c = 0 predictions match the cached M3 predictions within {max(out_check.values()):.2e} dB", flush=True)
    combos = list(itertools.product(T_C, HORIZONS, BUDGETS))
    out: dict[str, Any] = {"definition": __doc__, "grid": {"T_c_s": T_C, "H_s": HORIZONS, "budget": BUDGETS},
                           "tc0_check_max_abs_db": out_check, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        t0 = time.perf_counter()
        loss = info["extra_loss_db"][mi]
        unb = {job: snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - loss for job in jobs}
        snr_t = {j: snr_all[j][mi] for j in tune_jobs}
        snr_e = {j: snr_all[j][mi] for j in eval_jobs}
        or_t = R.stateless_rows(tune_jobs, snr_t, built, "oracle", bw, rate_req)[1]
        or_e = R.stateless_rows(eval_jobs, snr_e, built, "oracle", bw, rate_req)[1]
        a5p = pl["margins"][mi]["a5"]["params"]
        objs = {}
        for tc, h, budget in combos:
            fn = lambda ix, tc=tc, h=h, budget=budget: tables(ix, preds[(budget, tc)], unb, taus, h, d, rs, tau, bw, rate_req, ovh)
            rows = run_robust(tune_jobs, a5p, snr_t, built, rw, bw, rate_req, or_t, info["snr_req_db"], fn, ovh)
            objs[(tc, h, budget)] = R.objective(rows, 1)[0]
        best = min(combos, key=lambda c: (objs[c][0], objs[c][1], c))
        tc, h, budget = best
        rows = run_robust(eval_jobs, a5p, snr_e, built, rw, bw, rate_req, or_e, info["snr_req_db"],
                          lambda ix: tables(ix, preds[(budget, tc)], unb, taus, h, d, rs, tau, bw, rate_req, ovh), ovh)
        agg = R.aggregate(rows)
        pj = per_job_outage(rows)
        res = {"label": lab, "tuned": {"T_c_s": tc, "H_s": h, "budget": budget}, "tuning_objective": objs[best],
               "eval": {k: agg[k] for k in ("outage_req_s_per_min", "ho_per_min", "ping_pong", "precision")}, "per_job": pj,
               "vs_a5": paired(pj, pl["margins"][mi]["a5"]["per_job"]), "wall_s": time.perf_counter() - t0}
        out["margins"].append(res)
        v = res["vs_a5"]
        print(f"{lab}: robust {agg['outage_req_s_per_min']['mean']:.3f} (T_c {tc}, H {h}, b {budget}) vs A5 {pl['margins'][mi]['a5']['eval']['outage_req_s_per_min']['mean']:.3f} | "
              f"diff {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] p={v['wilcoxon_p_two_sided']:.3g} | {res['wall_s']:.0f} s", flush=True)
    out["wall_s"] = time.perf_counter() - clock
    dest = ROOT / "results" / "M5" / "review_b6.json"
    dest.write_text(json.dumps(out, indent=1, default=R._json) + "\n")
    print(f"wrote {dest} in {out['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
