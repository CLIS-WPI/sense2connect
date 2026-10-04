"""ADDED AFTER EXTERNAL REVIEW (B5): error-injection ablation of the planner's blockage prediction.

Diagnostic, not a new method. The planner is the receding-horizon planner of
scripts/run_m5_planner.py with the sensing-planner model (predicted SNR =
unblocked SNR along the UE path minus the predicted model-B LoS loss of the
"tracked" blockers; sensing overhead charged; no A3; first decision after
the E2 delay), H tuned per margin and condition on the tuning seeds (as
before), evaluated on the evaluation seeds and compared with A5 by a paired
test per margin (scripts/run_m5_paired.py: mean per-job difference with
t-based 95 % CI and Wilcoxon signed-rank p).

The "tracks" are built from ground truth at each 0.1 s report: every
blocker's true position, velocity and size (length/width/height of its
class), extrapolated at constant velocity by xapp/predict_torch.py; the UE
position and velocity are exact. Conditions (each injected independently):
- baseline: perfect tracks;
- miss p in {0.1, 0.2, 0.3, 0.4, 0.5}: each blocker is dropped from a
  report independently with probability p;
- spurious 1x / 2x: Poisson(lambda) extra tracks per report, lambda = 1x / 2x
  the measured false confirmed-track rate of the final M2 map tracker
  (track_fragment + track_ghost + track_other per CPI, results/M5/tracking.json);
  each is a bus-size box on a random lane (vehicle speed 6-12 m/s in the
  lane direction) or a pedestrian box on a random sidewalk (1.0-1.5 m/s),
  with probability 1/2 each, at a uniform x in [-40, 40] m (assumption);
- noise s in {0.5, 1, 2, 4}: Gaussian position error with sigma s m and
  velocity error with sigma s m/s per horizontal axis;
- noise measured: per-class position / velocity RMSE of the final map
  tracker divided by sqrt(2) per axis;
- combined: per-class miss probability 1 - track Pd, spurious 1x and the
  measured per-class noise;
- miss measured: per-class miss probability 1 - track Pd alone (added to
  isolate each error type at its measured rate).
Random draws are seeded per (job, condition).

Writes results/M5/review_b5.json.
"""

from __future__ import annotations

import json
import math
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
from run_m5_planner import per_job_outage, plan_first_switch, run_planner  # noqa: E402
from sim.comm.linkbudget import sensing_overhead, snr_ref_db  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from sim.scenes.motion import states_at  # noqa: E402
from sim.scenes.traffic import prepare_scenario  # noqa: E402

HORIZONS = (0.5, 1.0, 2.0, 3.0)
CLASS_OF = {"bus": "bus/truck", "truck": "bus/truck", "pedestrian": "pedestrian", "car": "car"}


def truth_tracks(raw: dict, job: tuple, n_r: int) -> dict[str, Any]:
    seed, mount, density = job
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    bl = list(sc["vehicles"]) + list(sc["pedestrians"])
    n_b = len(bl)
    st = np.zeros((n_r, n_b, 5))
    ue = np.zeros((n_r, len(sc["ues"]), 3))
    uev = np.zeros_like(ue)
    for r in range(n_r):
        s = states_at(sc, r * R.DT_SENSE)
        for b, spec in enumerate(bl):
            p_, v_ = s[spec["name"]]["position_m"], s[spec["name"]]["velocity_mps"]
            st[r, b] = (p_[0], p_[1], p_[2], v_[0], v_[1])
        for u, spec in enumerate(sc["ues"]):
            ue[r, u] = s[spec["name"]]["position_m"]
            uev[r, u] = s[spec["name"]]["velocity_mps"]
    size = np.array([[float(b["length_m"]), float(b["width_m"]), float(b["height_m"])] for b in bl])
    cls = [CLASS_OF.get(str(b["kind"]), "other") for b in bl]
    oru = np.array([o["position_m"] for o in sc["orus"]], dtype=np.float64)
    return {"state": st, "size": size, "cls": cls, "ue": ue, "ue_vel": uev, "lanes": list(raw["lanes"]), "sidewalks": list(raw["sidewalks"]),
            "kinds": sc["blocker_kinds"], "oru": oru}


def inject(tr: dict, cond: dict, rng: np.random.Generator, meas: dict) -> dict[str, np.ndarray]:
    st = tr["state"].copy()
    n_r, n_b, _ = st.shape
    valid = np.ones((n_r, n_b), dtype=bool)
    size = np.broadcast_to(tr["size"], (n_r, n_b, 3)).copy()
    cls = np.array(tr["cls"])
    miss = cond.get("miss")
    if miss == "measured":
        p = np.array([meas["miss"].get(c, 0.0) for c in cls])[None, :]
        valid &= rng.random((n_r, n_b)) >= p
    elif miss:
        valid &= rng.random((n_r, n_b)) >= float(miss)
    noise = cond.get("noise")
    if noise is not None:
        if noise == "measured":
            sp = np.array([meas["pos_axis"].get(c, 0.0) for c in cls])[None, :]
            sv = np.array([meas["vel_axis"].get(c, 0.0) for c in cls])[None, :]
        else:
            sp = sv = float(noise)
        st[:, :, 0] += rng.normal(0.0, 1.0, (n_r, n_b)) * sp
        st[:, :, 1] += rng.normal(0.0, 1.0, (n_r, n_b)) * sp
        st[:, :, 3] += rng.normal(0.0, 1.0, (n_r, n_b)) * sv
        st[:, :, 4] += rng.normal(0.0, 1.0, (n_r, n_b)) * sv
    lam = cond.get("spurious", 0.0) * meas["false_tracks_per_cpi"]
    if lam > 0:
        counts = rng.poisson(lam, n_r)
        k_extra = int(counts.max()) if n_r else 0
        ex_st = np.zeros((n_r, k_extra, 5))
        ex_sz = np.ones((n_r, k_extra, 3))
        ex_ok = np.zeros((n_r, k_extra), dtype=bool)
        bus, ped = tr["kinds"]["bus"], tr["kinds"]["pedestrian"]
        for r in range(n_r):
            for i in range(counts[r]):
                if rng.random() < 0.5:
                    line = tr["lanes"][int(rng.integers(len(tr["lanes"])))]
                    speed = rng.uniform(6.0, 12.0)
                    dims, z = (float(bus["length_m"]), float(bus["width_m"]), float(bus["height_m"])), float(bus["height_m"]) / 2
                else:
                    line = tr["sidewalks"][int(rng.integers(len(tr["sidewalks"])))]
                    speed = rng.uniform(1.0, 1.5) * (1 if rng.random() < 0.5 else -1)
                    dims, z = (float(ped["length_m"]), float(ped["width_m"]), float(ped["height_m"])), float(ped["height_m"]) / 2
                direction = float(np.sign(line["direction"][0]) or 1.0)
                ex_st[r, i] = (rng.uniform(-40.0, 40.0), float(line["origin_m"][1]), z, direction * speed, 0.0)
                ex_sz[r, i] = dims
                ex_ok[r, i] = True
        st = np.concatenate([st, ex_st], axis=1)
        size = np.concatenate([size, ex_sz], axis=1)
        valid = np.concatenate([valid, ex_ok], axis=1)
    return {"state": st, "size": size, "valid": valid}


def tables(index, preds, unb, taus, h_s, d, rs, tau, bw, rate_req, ovh) -> np.ndarray:
    """Planner switch table [L, R, 2] from predicted LoS loss [R, U, C, tau] (vectorised)."""
    hs = int(round(h_s / R.DT_COMM))
    dtau = float(taus[1] - taus[0])
    out = []
    for _, job, u in index:
        pr = np.clip(np.nan_to_num(preds[job][:, u, :, :], posinf=200.0), 0.0, 200.0)  # [R, C, tau]
        n_r = pr.shape[0]
        un = unb[job][:, u, :]  # [T, C]
        ks = (np.arange(n_r) * rs + d + 1)[:, None] + np.arange(hs)[None, :]
        ks_c = np.minimum(ks, un.shape[0] - 1)
        lead = (ks - (np.arange(n_r) * rs)[:, None]) * R.DT_COMM
        x = np.clip(lead / dtau, 0.0, len(taus) - 1.0)
        i0 = np.minimum(np.floor(x).astype(np.int64), len(taus) - 2)
        fr = x - i0
        rr = np.arange(n_r)[:, None]
        bad = np.zeros((n_r, hs, 2), dtype=bool)
        for c in range(2):
            loss = pr[rr, c, i0] * (1 - fr) + pr[rr, c, i0 + 1] * fr
            snr = un[ks_c, c] - loss
            rate = bw * (1.0 - ovh) * np.minimum(np.log2(1.0 + 10.0 ** (snr / 10.0)), MAX_NR_SE)
            bad[:, :, c] = rate < rate_req
        seqs = np.repeat(bad, 2, axis=0)
        cur = np.tile(np.array([0, 1]), n_r)
        out.append(plan_first_switch(seqs, cur, tau, rs).reshape(n_r, 2))
    return np.stack(out)


def main() -> None:
    import torch

    from xapp.predict_torch import predict_torch

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*", help="condition names to run (default: all)")
    ap.add_argument("--margins", nargs="*", help="margin labels to run (default: all)")
    ap.add_argument("--out", default="review_b5.json")
    args = ap.parse_args()
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
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    trk = json.loads((ROOT / "results" / "M5" / "tracking.json").read_text())["variants"]["image|map"]["summary"]
    meas = {
        "miss": {c: 1.0 - v["track_pd"] for c, v in trk["classes"].items()},
        "pos_axis": {c: v["position_rmse_m"] / math.sqrt(2.0) for c, v in trk["classes"].items()},
        "vel_axis": {c: v["velocity_rmse_mps"] / math.sqrt(2.0) for c, v in trk["classes"].items()},
        "false_tracks_per_cpi": trk["track_fragment_per_cpi"] + trk["track_ghost_per_cpi"] + trk["track_other_per_cpi"],
    }
    conds = [("baseline", {})] + [(f"miss {p}", {"miss": p}) for p in (0.1, 0.2, 0.3, 0.4, 0.5)] + \
            [(f"spurious {k}x", {"spurious": float(k)}) for k in (1, 2)] + [(f"noise {s}", {"noise": s}) for s in (0.5, 1.0, 2.0, 4.0)] + \
            [("noise measured", {"noise": "measured"}), ("combined measured", {"miss": "measured", "spurious": 1.0, "noise": "measured"}),
             ("miss measured", {"miss": "measured"})]
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, tune_jobs + eval_jobs, rw, bw, info["extra_loss_db"])
    b = rw["budget"]
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    jobs = tune_jobs + eval_jobs
    n_r = int(built[jobs[0]]["data"]["pred_2"].shape[0])
    taus = built[jobs[0]]["data"]["taus"]
    wl = 299792458.0 / float(raw["carrier_hz"])
    truth = {job: truth_tracks(raw, job, n_r) for job in jobs}
    out: dict[str, Any] = {"definition": __doc__, "measured": meas, "conditions": {}}
    for ci, (cname, cond) in enumerate(conds):
        if args.conds and cname not in args.conds:
            continue
        t0 = time.perf_counter()
        preds = {}
        for ji, job in enumerate(jobs):
            rng = np.random.default_rng(1_000_003 * (ci + 1) + 9_973 * job[0] + 17 * R.MOUNTS.index(job[1]) + R.DENSITIES.index(job[2]))
            tk = inject(truth[job], cond, rng, meas)
            preds[job] = predict_torch(tk, truth[job]["ue"], truth[job]["ue_vel"], truth[job]["oru"], taus, wl)
        res_c = {"margins": []}
        for mi, lab in enumerate(info["labels"]):
            if args.margins and lab not in args.margins:
                continue
            loss = info["extra_loss_db"][mi]
            unb = {job: snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - loss for job in jobs}
            snr_t = {j: snr_all[j][mi] for j in tune_jobs}
            snr_e = {j: snr_all[j][mi] for j in eval_jobs}
            or_t = R.stateless_rows(tune_jobs, snr_t, built, "oracle", bw, rate_req)[1]
            or_e = R.stateless_rows(eval_jobs, snr_e, built, "oracle", bw, rate_req)[1]
            objs = {}
            for h in HORIZONS:
                rows = run_planner(tune_jobs, lambda ix, h=h: tables(ix, preds, unb, taus, h, d, rs, tau, bw, rate_req, ovh), snr_t, built, rw, bw, rate_req, or_t, ovh)
                objs[h] = R.objective(rows, 1)[0]
            h_best = min(HORIZONS, key=lambda h: (objs[h][0], objs[h][1], h))
            rows = run_planner(eval_jobs, lambda ix: tables(ix, preds, unb, taus, h_best, d, rs, tau, bw, rate_req, ovh), snr_e, built, rw, bw, rate_req, or_e, ovh)
            agg = R.aggregate(rows)
            pj = per_job_outage(rows)
            a5 = pl["margins"][mi]["a5"]["per_job"]
            res_c["margins"].append({"label": lab, "H": h_best, "eval": {k: agg[k] for k in ("outage_req_s_per_min", "ho_per_min", "ping_pong")},
                                     "per_job": pj, "vs_a5": paired(pj, a5)})
        res_c["wall_s"] = time.perf_counter() - t0
        out["conditions"][cname] = res_c
        line = " ".join(f"{m['label'].split(' ')[0]}:{m['eval']['outage_req_s_per_min']['mean']:.3f}({m['vs_a5']['mean_diff']:+.3f})" for m in res_c["margins"])
        print(f"{cname} [{res_c['wall_s']:.0f} s]: {line}", flush=True)
    out["wall_s"] = time.perf_counter() - clock
    dest = ROOT / "results" / "M5" / args.out
    dest.write_text(json.dumps(out, indent=1, default=R._json) + "\n")
    print(f"wrote {dest} in {out['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
