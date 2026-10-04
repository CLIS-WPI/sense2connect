"""ADDED AFTER EXTERNAL REVIEW (A3, A4).

A3 -- car share of the closable gap per margin (fig_value): A5 (tuned) vs the
any-step cost-aware oracle on the evaluation jobs; net closed steps (A5 in
outage, oracle not, minus the reverse) attributed to the class of the
dominant LoS blocker of A5's cell; the car-attributed part is split by the
LoS loss of A5's cell at that step (< 10 dB vs >= 10 dB). Shares of the
A5-to-instantaneous-oracle gap, pooled.

A4 -- onset at 1 ms: for every evaluation-seed 10 dB event (bus/truck and
pedestrian), model B is evaluated analytically every 1 ms on the LoS
segment O-RU -> UE of the event cell from the exact analytic poses of the
UE and every blocker (sim.scenes.motion.states_at; no re-trace), over
[event start - 1 s, event end + 0.2 s], and the 10-90 % onset is computed
with the same rule as at 10 ms (xapp.metrics.onset_10_90; the event
interval is the 10 ms event mapped to 1 ms). Medians per class at 1 ms and
the same events at 10 ms. Uses 4 CPU worker processes.

Writes results/M5/review_a34.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DT1 = 0.001


def _onset_job(item: tuple) -> list[dict[str, Any]]:
    job, events = item
    import torch

    torch.set_num_threads(1)
    from sim.comm.blockage import path_blocker_loss, sum_blockage_db
    from sim.scenes.config import load_yaml
    from sim.scenes.motion import states_at
    from sim.scenes.traffic import prepare_scenario
    from xapp.metrics import ONSET_CAP_DB, onset_10_90

    seed, mount, density = job
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    wavelength = 299792458.0 / float(raw["carrier_hz"])
    blockers = list(sc["vehicles"]) + list(sc["pedestrians"])
    oru_pos = [np.asarray(o["position_m"], dtype=np.float64) for o in sc["orus"]]
    ues = [u["name"] for u in sc["ues"]]
    out = []
    for ev in events:
        t0 = max(0.0, ev["start_s"] - 1.0)
        t1 = ev["end_s"] + 0.2
        ts = np.arange(int(round(t0 / DT1)), int(round(t1 / DT1)) + 1) * DT1
        loss = np.zeros(len(ts))
        for i, t in enumerate(ts):
            st = states_at(sc, float(t))
            ue = st[ues[ev["ue"]]]["position_m"]
            seg = [(oru_pos[ev["cell"]], ue)]
            per = []
            for b in blockers:
                lb, _ = path_blocker_loss(seg, st[b["name"]]["position_m"], float(b["length_m"]), float(b["width_m"]), float(b["height_m"]), wavelength)
                per.append(lb)
            loss[i] = sum_blockage_db(per)
        series = np.clip(np.nan_to_num(loss, posinf=ONSET_CAP_DB), 0.0, ONSET_CAP_DB)
        k0 = int(round((ev["start_s"] - ts[0]) / DT1))
        k1 = int(round((ev["end_s"] + 0.01 - ts[0]) / DT1)) - 1
        k1 = min(k1, len(ts) - 1)
        out.append({"job": list(job), "ue": ev["ue"], "class": ev["class"], "start_s": ev["start_s"], "onset_10ms_s": ev["onset_s"],
                    "onset_1ms_s": onset_10_90(series, k0, k1, DT1)})
    return out


def a3(built, eval_jobs, snr_all, info, rw, raw, bw, rate_req) -> list[dict[str, Any]]:
    import run_m3 as R
    from run_m5_dporacle import viterbi
    from run_m5_foresight import CLASS_OF, blocker_kinds, usable
    from run_m5_planner import sim_reactive_masks

    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    kinds = {job: blocker_kinds(raw, job) for job in eval_jobs}
    rows = []
    for mi, lab in enumerate(info["labels"]):
        snr_m = {j: snr_all[j][mi] for j in eval_jobs}
        a5p = pl["margins"][mi]["a5"]["params"]
        lanes, index, sim = sim_reactive_masks(eval_jobs, a5p, "a5", snr_m, built, rw, bw, rate_req, info["snr_req_db"])
        bad = ~usable(lanes.snr_db, bw, rate_req)
        ca, _ = viterbi(bad, tau, 1)
        br = sim["outage_req"]
        n_t = bad.shape[1]
        srv = sim["serving"].astype(np.int64)
        gap = int(br.sum() - bad.all(-1).sum())
        net = {"bus/truck": 0, "pedestrian": 0, "car": 0, "car<10dB": 0, "car>=10dB": 0, "no LoS blocker": 0}
        for lane, (_, job, u) in enumerate(index):
            d = built[job]["data"]
            k = np.arange(n_t)
            dom = d["los_blocker"][k, u, srv[lane]]
            los = d["los_loss_db"][k, u, srv[lane]]
            cls = np.array(["no LoS blocker" if x < 0 else CLASS_OF.get(kinds[job][x], "other") for x in dom])
            delta = (br[lane] & ~ca[lane]).astype(np.int64) - (ca[lane] & ~br[lane]).astype(np.int64)
            for c in ("bus/truck", "pedestrian", "car", "no LoS blocker"):
                net[c] += int(delta[cls == c].sum())
            net["car<10dB"] += int(delta[(cls == "car") & (los < 10.0)].sum())
            net["car>=10dB"] += int(delta[(cls == "car") & (los >= 10.0)].sum())
        rows.append({"label": lab, "gap_steps": gap, "shares_of_gap": {k_: (v / gap if gap > 0 else None) for k_, v in net.items()},
                     "car_sub10_share_of_car": (net["car<10dB"] / net["car"]) if net["car"] else None})
        print(f"A3 {lab}: car {rows[-1]['shares_of_gap']['car']}, car from <10 dB {rows[-1]['car_sub10_share_of_car']}", flush=True)
    return rows


def main() -> None:
    import run_m3 as R
    from sim.scenes.config import load_yaml

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, eval_jobs, rw, bw, info["extra_loss_db"])
    out: dict[str, Any] = {"definition": __doc__}
    out["a3_car_share"] = a3(built, eval_jobs, snr_all, info, rw, raw, bw, rate_req)
    items = []
    for job in eval_jobs:
        evs = [ev for ev in built[job]["events"] if ev["class"] in ("bus/truck", "pedestrian")]
        if evs:
            items.append((job, evs))
    with mp.get_context("spawn").Pool(4) as pool:
        rows = [r for part in pool.imap(_onset_job, items, chunksize=1) for r in part]
    summ = {}
    for cls in ("bus/truck", "pedestrian"):
        sel = [r for r in rows if r["class"] == cls]
        v1 = np.array([r["onset_1ms_s"] for r in sel if r["onset_1ms_s"] is not None and np.isfinite(r["onset_1ms_s"])])
        v10 = np.array([r["onset_10ms_s"] for r in sel if r["onset_10ms_s"] is not None and np.isfinite(r["onset_10ms_s"])])
        summ[cls] = {"n_events": len(sel), "median_1ms_s": float(np.median(v1)), "p10_p90_1ms_s": np.percentile(v1, [10, 90]).tolist(),
                     "median_10ms_s": float(np.median(v10)), "p10_p90_10ms_s": np.percentile(v10, [10, 90]).tolist(),
                     "share_below_10ms_at_1ms": float(np.mean(v1 < 0.01))}
        print(f"A4 {cls}: n {len(sel)} median onset 1 ms {summ[cls]['median_1ms_s']:.4f} s (10 ms: {summ[cls]['median_10ms_s']:.3f} s)", flush=True)
    out["a4_onset"] = {"summary": summ, "events": rows}
    dest = ROOT / "results" / "M5" / "review_a34.json"
    dest.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
