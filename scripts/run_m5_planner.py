"""Equal-effort reactive baselines and a receding-horizon planner (xApp v2).

ADDED AFTER THE COST-AWARE-ORACLE RESULT (results/M5/dporacle.json).
Tuning on tuning seeds 101-105 per margin (objective: outage at the
service rate incl. interruption, ties -> fewer HO/UE-min), evaluation on
1001-1010, tau_HO = 20 ms, E2 loop delay 20 ms, report period 0.1 s.

1. Reactive baselines with equal effort:
   - A3 (wide): offset {0,1,3,6,10,15} dB x hysteresis {0,1,3,6,10,15} dB x
     TTT {40,80,160,320,640} ms (180 points);
   - A5: serving filtered SNR < SNR_req + t1 AND neighbour filtered SNR >
     SNR_req + t2 for TTT; t1 {-3,0,3,6,10} dB x t2 {0,3,6,10} dB x
     TTT {40,80,160,320,640} ms (100 points);
   - best reactive = the one with the lower tuning objective per margin.
2. Value of foresight: best reactive vs the cost-aware oracle (Viterbi with
   perfect SNR of both cells, switches only at 0.1 s epochs, tau_HO 20 ms),
   absolute and as a share of the best-reactive-to-instantaneous-oracle gap
   (pooled), split by the dominant LoS blocker of the best reactive's cell.
3. Receding-horizon planner: at every report r, from the current cell, a
   2-state Viterbi over the horizon H in {0.5, 1, 2, 3} s (steps from
   r*0.1 s + E2 delay + 10 ms, switches only at future 0.1 s decision
   epochs, every switch = tau_HO outage) on PREDICTED SNR of both cells; the
   first decision is applied after the E2 delay. No A3 underneath; no hold.
   a) genie-planner: true future blocked SNR;
   b) sensing-planner: predicted SNR = unblocked SNR along the UE path
      (digital-twin radio map, i.e. the true unblocked SNR) minus the
      predicted model-B LoS loss of the tracked blockers (final M2 map
      tracker, detector budget 2 or 4 tuned, UE position fix with sigma 1 m,
      xapp/predict_torch.py); sensing overhead charged in the plan and the rate.
   c) diagnostic only (not a policy): the sensing-planner model with the TRUE
      LoS loss instead of the predicted one (separates predictor error from
      the "unblocked SNR minus LoS loss" approximation), overhead charged,
      every H on the evaluation seeds.
4. Per margin: A3, A5, best reactive, genie-planner(H), sensing-planner(H),
   cost-aware oracle, instantaneous oracle (mean, 95 % CI, HO/UE-min,
   ping-pong).

Writes results/M5/planner.json.
"""

from __future__ import annotations

import itertools
import json
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
from run_m5_dporacle import viterbi  # noqa: E402
from run_m5_foresight import CLASS_OF, blocker_kinds, usable  # noqa: E402
from sim.comm.linkbudget import sensing_overhead, snr_ref_db  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.metrics import lane_summary  # noqa: E402
from xapp.schemes import REASON, simulate  # noqa: E402

A3W = {"offset_db": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0], "hysteresis_db": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0], "ttt_s": [0.04, 0.08, 0.16, 0.32, 0.64]}
A5G = {"t1_db": [-3.0, 0.0, 3.0, 6.0, 10.0], "t2_db": [0.0, 3.0, 6.0, 10.0], "ttt_s": [0.04, 0.08, 0.16, 0.32, 0.64]}
HORIZONS = (0.5, 1.0, 2.0, 3.0)
BUDGETS = (2, 4)
MAX_LANES = 6000
CLASSES = ("bus/truck", "pedestrian", "car", "no LoS blocker")


# ------------------------------------------------------------------ planner core


def plan_first_switch(bad, cur, tau: int, epoch: int, device: str = "cuda"):
    """Receding-horizon decision: does the min-outage plan switch at its first step?

    ``bad`` [N, Hs, 2] bool (predicted cell below the service rate), ``cur``
    [N] current cell. Two forced first actions (stay / switch), then the
    same Viterbi as run_m5_dporacle.viterbi with switches only at h % epoch == 0.
    Returns bool [N]; ties keep the current cell.
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
        results.append(cost.reshape(n, -1).min(dim=1).values)
    return (results[1] < results[0]).cpu().numpy()


def _check_planner() -> None:
    """plan_first_switch against an exhaustive search on random small cases."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        hs = int(rng.integers(2, 13))
        tau = int(rng.integers(0, 3))
        epoch = int(rng.choice([1, 3]))
        bad = rng.random((1, hs, 2)) < rng.uniform(0.1, 0.8)
        cur = np.array([int(rng.integers(0, 2))])
        got = bool(plan_first_switch(bad, cur, tau, epoch)[0])

        def brute(first_switch: bool) -> int:
            best = None
            c = 1 - cur[0] if first_switch else cur[0]
            p = tau if (first_switch and tau > 0) else 0
            acc = 1 if (first_switch and tau > 0) else int(bad[0, 0, c])
            stack = [(1, c, p, acc)]
            while stack:
                k, cc, pp, aa = stack.pop()
                if k == hs:
                    best = aa if best is None else min(best, aa)
                    continue
                q = max(pp - 1, 0)
                stack.append((k + 1, cc, q, aa + (1 if q > 0 else int(bad[0, k, cc]))))
                if k % epoch == 0:
                    o = 1 - cc
                    stack.append((k + 1, o, tau, aa + (1 if tau > 0 else int(bad[0, k, o]))))
            return int(best)

        if got != (brute(True) < brute(False)):
            raise SystemExit("planner first decision differs from exhaustive search")


def planner_table(bad_fn, n_reports: int, n_lanes: int, hs: int, k0_of_r, tau: int) -> np.ndarray:
    """Switch decision [L, R, 2] for every lane, report and current cell."""
    out = np.zeros((n_lanes, n_reports, 2), dtype=bool)
    chunk = max(1, 200_000 // (n_reports * 2))
    for lo in range(0, n_lanes, chunk):
        lanes = list(range(lo, min(n_lanes, lo + chunk)))
        seqs = np.stack([bad_fn(l_, k0_of_r(r), hs) for l_ in lanes for r in range(n_reports)])  # [n*R, Hs, 2]
        seqs = np.repeat(seqs, 2, axis=0)
        cur = np.tile(np.array([0, 1]), len(lanes) * n_reports)
        dec = plan_first_switch(seqs, cur, tau, int(round(R.DT_SENSE / R.DT_COMM)))
        out[lo : lo + len(lanes)] = dec.reshape(len(lanes), n_reports, 2)
    return out


# ------------------------------------------------------------------ lanes


def run_reactive(jobs, combos, kind, snr_m, built, rw, bw, rate_req, orates, snr_req) -> list[dict]:
    rows = []
    lanes_per_combo = sum(built[j]["data"]["blocked_power"].shape[1] for j in jobs)
    per = max(1, MAX_LANES // lanes_per_combo)
    for lo in range(0, len(combos), per):
        chunk = combos[lo : lo + per]
        a3c = [{"offset_db": c.get("offset_db", 0.0), "hysteresis_db": c.get("hysteresis_db", 0.0), "ttt_s": c["ttt_s"]} for c in chunk]
        lanes, index = R.make_lanes(jobs, a3c, snr_m, built, "a3", {}, rw, None)
        if kind == "a5":
            lanes.scheme = np.full(len(index), REASON["a5"])
            lanes.a5_thr1 = np.array([snr_req + chunk[ci]["t1_db"] for ci, _, _ in index])
            lanes.a5_thr2 = np.array([snr_req + chunk[ci]["t2_db"] for ci, _, _ in index])
        sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
        for lane, (ci, job, u) in enumerate(index):
            evs = [ev for ev in built[job]["events"] if ev["ue"] == u]
            s = lane_summary(sim, lane, evs, orates[job][:, u], R.DT_COMM)
            s.update({"combo": lo + ci, "job": job, "ue": u})
            rows.append(s)
        del sim
    return rows


def sim_reactive_masks(jobs, combo, kind, snr_m, built, rw, bw, rate_req, snr_req):
    a3c = [{"offset_db": combo.get("offset_db", 0.0), "hysteresis_db": combo.get("hysteresis_db", 0.0), "ttt_s": combo["ttt_s"]}]
    lanes, index = R.make_lanes(jobs, a3c, snr_m, built, "a3", {}, rw, None)
    if kind == "a5":
        lanes.scheme = np.full(len(index), REASON["a5"])
        lanes.a5_thr1 = np.full(len(index), snr_req + combo["t1_db"])
        lanes.a5_thr2 = np.full(len(index), snr_req + combo["t2_db"])
    return lanes, index, simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)


def run_planner(jobs, table_fn, snr_m, built, rw, bw, rate_req, orates, overhead: float) -> list[dict]:
    """Planner lanes: xApp lanes with the planner's switch table, A3 disabled, no hold."""
    lanes, index = R.make_lanes(jobs, [{"budget": 2, "horizon_s": 0.5, "hold_s": 0.0}], snr_m, built, "xapp",
                                {"offset_db": 0.0, "hysteresis_db": 0.0, "ttt_s": 0.04}, rw, None, hybrid=True)
    lanes.offset_db = np.full(len(index), 1e9)  # A3 never fires
    lanes.block_db = float("inf")  # no RSRP condition on planner switches
    lanes.overhead = np.full(len(index), overhead)
    lanes.trigger = table_fn(index)
    lanes.trigger_end_steps = np.zeros_like(lanes.trigger, dtype=np.int64)
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

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--margins", nargs="*", help="labels to run (default: all), e.g. '10 dB'")
    parser.add_argument("--out", default="planner.json")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    _check_planner()
    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    m3 = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    dp = json.loads((ROOT / "results" / "M5" / "dporacle.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_req = info["snr_req_db"]
    snr_all = R.all_snr(built, tune_jobs + eval_jobs, rw, bw, info["extra_loss_db"])
    kinds = {job: blocker_kinds(raw, job) for job in eval_jobs}
    b = rw["budget"]
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    a3_combos = R._grid(A3W)
    a5_combos = R._grid(A5G)
    out: dict[str, Any] = {"label": "added after the cost-aware-oracle result", "definition": __doc__, "grids": {"a3_wide": A3W, "a5": A5G, "horizons_s": HORIZONS, "budgets": BUDGETS}, "margins": []}

    for mi, lab in enumerate(info["labels"]):
        if args.margins and lab not in args.margins:
            continue
        t0 = time.perf_counter()
        loss = info["extra_loss_db"][mi]
        snr_t = {j: snr_all[j][mi] for j in tune_jobs}
        snr_e = {j: snr_all[j][mi] for j in eval_jobs}
        or_t = R.stateless_rows(tune_jobs, snr_t, built, "oracle", bw, rate_req)[1]
        oracle_rows, or_e = R.stateless_rows(eval_jobs, snr_e, built, "oracle", bw, rate_req)
        res: dict[str, Any] = {"label": lab, "margin_db": info["points_db"][mi], "oracle_inst": R.aggregate(oracle_rows)}

        # 1. reactive baselines
        tuned = {}
        for kind, combos in (("a3_wide", a3_combos), ("a5", a5_combos)):
            rows_t = run_reactive(tune_jobs, combos, "a5" if kind == "a5" else "a3", snr_t, built, rw, bw, rate_req, or_t, snr_req)
            obj = R.objective(rows_t, len(combos))
            i = min(range(len(combos)), key=lambda c: (obj[c][0], obj[c][1], c))
            tuned[kind] = (combos[i], obj[i])
            rows_e = run_reactive(eval_jobs, [combos[i]], "a5" if kind == "a5" else "a3", snr_e, built, rw, bw, rate_req, or_e, snr_req)
            res[kind] = {"params": combos[i], "tuning_objective": obj[i], "eval": R.aggregate(rows_e)}
        best_kind = min(("a3_wide", "a5"), key=lambda k: (tuned[k][1][0], tuned[k][1][1]))
        res["best_reactive"] = best_kind

        # 2. value of foresight: best reactive vs cost-aware oracle (0.1 s epochs, tau_HO 20 ms)
        lanes, index, sim = sim_reactive_masks(eval_jobs, tuned[best_kind][0], "a5" if best_kind == "a5" else "a3", snr_e, built, rw, bw, rate_req, snr_req)
        bad = ~usable(lanes.snr_db, bw, rate_req)
        ca, _ = viterbi(bad, tau, rs)
        inst = bad.all(-1)
        br = sim["outage_req"]
        n_t = bad.shape[1]
        srv = sim["serving"].astype(np.int64)
        tot = {"br": int(br.sum()), "ca": int(ca.sum()), "inst": int(inst.sum())}
        by_cls = {c: 0 for c in CLASSES}
        for lane, (_, job, u) in enumerate(index):
            dom = built[job]["data"]["los_blocker"][np.arange(n_t), u, srv[lane]]
            cls = np.array(["no LoS blocker" if x < 0 else CLASS_OF.get(kinds[job][x], "other") for x in dom])
            net = (br[lane] & ~ca[lane]).astype(np.int64) - (ca[lane] & ~br[lane]).astype(np.int64)
            for c in CLASSES:
                by_cls[c] += int(net[cls == c].sum())
        gap = tot["br"] - tot["inst"]
        ca_ref = dp["margins"][mi]["0.020"]["costaware_epoch"]
        res["value_of_foresight"] = {
            "best_reactive_minus_costaware_s_per_ue_min": res[best_kind]["eval"]["outage_req_s_per_min"]["mean"] - ca_ref["mean"],
            "share_of_best_reactive_gap_pooled": (tot["br"] - tot["ca"]) / gap if gap > 0 else None,
            "by_class_share_of_gap_pooled": {c: (v / gap if gap > 0 else None) for c, v in by_cls.items()},
            "costaware_epoch": ca_ref,
            "costaware_recomputed_check": abs(tot["ca"] * R.DT_COMM / (n_t * R.DT_COMM / 60.0) / len(index) - ca_ref["mean"]) < 1e-9 or None,
        }

        # 3. planners
        def genie_tables(jobs, snr_m_, h_s):
            hs = int(round(h_s / R.DT_COMM))

            def fn(index_):
                lanes_snr = [snr_m_[job][:, u, :] for _, job, u in index_]
                bad_l = [~usable(s, bw, rate_req) for s in lanes_snr]
                n_r = int(built[index_[0][1]]["data"]["pred_2"].shape[0])

                def bad_fn(li, k0, hs_):
                    seq = bad_l[li][k0 : k0 + hs_]
                    if seq.shape[0] < hs_:
                        seq = np.concatenate([seq, np.repeat(seq[-1:], hs_ - seq.shape[0], axis=0)]) if seq.shape[0] else np.zeros((hs_, 2), dtype=bool)
                    return seq

                return planner_table(bad_fn, n_r, len(index_), hs, lambda r: r * rs + d + 1, tau)
            return fn

        def sensing_tables(jobs, h_s, budget):
            hs = int(round(h_s / R.DT_COMM))

            def fn(index_):
                n_r = int(built[index_[0][1]]["data"]["pred_2"].shape[0])
                cache = []
                for _, job, u in index_:
                    dd = built[job]["data"]
                    unb = snr_ref_db(dd["unblocked_power"][:, u, :], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - loss
                    pred = np.nan_to_num(dd[f"pred_{budget}"][:, u, :, :], posinf=200.0)  # [R, C, tau]
                    cache.append((unb, pred, dd["taus"]))

                def bad_fn(li, k0, hs_):
                    unb, pred, taus = cache[li]
                    r = (k0 - d - 1) // rs
                    ks = np.minimum(np.arange(k0, k0 + hs_), unb.shape[0] - 1)
                    lead = (ks - r * rs) * R.DT_COMM
                    lossp = np.stack([np.interp(lead, taus, pred[r, c]) for c in range(2)], axis=-1)
                    snr_p = unb[ks] - lossp
                    rate = bw * (1.0 - ovh) * np.minimum(np.log2(1.0 + 10.0 ** (snr_p / 10.0)), MAX_NR_SE)
                    return rate < rate_req

                return planner_table(bad_fn, n_r, len(index_), hs, lambda r: r * rs + d + 1, tau)
            return fn

        def trueloss_tables(jobs, h_s):
            """Diagnostic: sensing-planner model (unblocked SNR - LoS loss) with the TRUE LoS loss."""
            hs = int(round(h_s / R.DT_COMM))

            def fn(index_):
                n_r = int(built[index_[0][1]]["data"]["pred_2"].shape[0])
                cache = []
                for _, job, u in index_:
                    dd = built[job]["data"]
                    unb = snr_ref_db(dd["unblocked_power"][:, u, :], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - loss
                    cache.append((unb, np.clip(np.nan_to_num(dd["los_loss_db"][:, u, :], posinf=200.0), 0.0, 200.0)))

                def bad_fn(li, k0, hs_):
                    unb, los = cache[li]
                    ks = np.minimum(np.arange(k0, k0 + hs_), unb.shape[0] - 1)
                    snr_p = unb[ks] - los[ks]
                    rate = bw * (1.0 - ovh) * np.minimum(np.log2(1.0 + 10.0 ** (snr_p / 10.0)), MAX_NR_SE)
                    return rate < rate_req

                return planner_table(bad_fn, n_r, len(index_), hs, lambda r: r * rs + d + 1, tau)
            return fn

        res["diag_trueloss_planner"] = {}
        for h_s in HORIZONS:
            rows_e = run_planner(eval_jobs, trueloss_tables(eval_jobs, h_s), snr_e, built, rw, bw, rate_req, or_e, ovh)
            res["diag_trueloss_planner"][f"{h_s:.1f}"] = {"eval": R.aggregate(rows_e)}
        res["genie_planner"] = {}
        res["sensing_planner"] = {}
        gp_obj, sp_obj = {}, {}
        for h_s in HORIZONS:
            rows_t = run_planner(tune_jobs, genie_tables(tune_jobs, snr_t, h_s), snr_t, built, rw, bw, rate_req, or_t, 0.0)
            gp_obj[h_s] = R.objective(rows_t, 1)[0]
            rows_e = run_planner(eval_jobs, genie_tables(eval_jobs, snr_e, h_s), snr_e, built, rw, bw, rate_req, or_e, 0.0)
            res["genie_planner"][f"{h_s:.1f}"] = {"tuning_objective": gp_obj[h_s], "eval": R.aggregate(rows_e)}
            best_b, best_o = None, None
            for budget in BUDGETS:
                rows_t = run_planner(tune_jobs, sensing_tables(tune_jobs, h_s, budget), snr_t, built, rw, bw, rate_req, or_t, ovh)
                o = R.objective(rows_t, 1)[0]
                if best_o is None or (o[0], o[1]) < (best_o[0], best_o[1]):
                    best_b, best_o = budget, o
            rows_e = run_planner(eval_jobs, sensing_tables(eval_jobs, h_s, best_b), snr_e, built, rw, bw, rate_req, or_e, ovh)
            sp_obj[h_s] = best_o
            res["sensing_planner"][f"{h_s:.1f}"] = {"budget": best_b, "tuning_objective": best_o, "eval": R.aggregate(rows_e)}
        res["genie_planner_tuned_H"] = min(HORIZONS, key=lambda h: (gp_obj[h][0], gp_obj[h][1], h))
        res["sensing_planner_tuned_H"] = min(HORIZONS, key=lambda h: (sp_obj[h][0], sp_obj[h][1], h))
        res["wall_s"] = time.perf_counter() - t0
        out["margins"].append(res)
        g = res["genie_planner"][f"{res['genie_planner_tuned_H']:.1f}"]["eval"]["outage_req_s_per_min"]["mean"]
        s_ = res["sensing_planner"][f"{res['sensing_planner_tuned_H']:.1f}"]["eval"]["outage_req_s_per_min"]["mean"]
        print(f"{lab}: A3w {res['a3_wide']['eval']['outage_req_s_per_min']['mean']:.3f} A5 {res['a5']['eval']['outage_req_s_per_min']['mean']:.3f} "
              f"best {best_kind} | genie-planner(H={res['genie_planner_tuned_H']}) {g:.3f} | sensing-planner(H={res['sensing_planner_tuned_H']}) {s_:.3f} | "
              f"cost-aware {res['value_of_foresight']['costaware_epoch']['mean']:.3f} inst {res['oracle_inst']['outage_req_s_per_min']['mean']:.3f} | "
              f"foresight share {res['value_of_foresight']['share_of_best_reactive_gap_pooled']} | {res['wall_s']:.0f} s", flush=True)
    out["wall_s"] = time.perf_counter() - clock
    dest = ROOT / "results" / "M5" / args.out
    dest.write_text(json.dumps(out, indent=1, default=R._json) + "\n")
    print(f"wrote {dest} in {out['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
