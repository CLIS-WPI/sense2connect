"""ADDED AFTER EXTERNAL REVIEW (A1): oracle accounting check.

The simulator (xapp/schemes.py) counts outage at the service rate as the
UNION of HO interruption and rate < service rate (an interrupted step counts
once). The cost-aware Viterbi (scripts/run_m5_dporacle.py) uses the same
union objective: its state carries the remaining interruption steps p; a
step costs 1 if p > 0 (interrupted) else 1 if the cell is below the service
rate. This script verifies, per evaluation job and margin (tau_HO 20 ms),
that no scheme has a lower outage than the cost-aware oracle:
- any-step oracle: switches at any 10 ms step (a lower bound for every
  causal or non-causal policy with the simulator's switching rule);
- epoch oracle: switches only at k mod 10 == 0 (bounds only policies that
  switch on that grid; violations by schemes that switch at other times are
  reported separately, not as errors).
Schemes re-simulated here: A3 (main), xApp + A3, hybrid (joint), genie +
A3 (no overhead), onset-advance genie (no overhead), A3 wide, A5; planners
(genie, sensing, true-LoS-loss; every H) use their per-job outages in
results/M5/planner.json. Per job = mean over its 2 UEs.

Writes results/M5/review_a1.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_m3 as R  # noqa: E402
from run_m3_genie import add_genie  # noqa: E402
from run_m5_dporacle import viterbi  # noqa: E402
from run_m5_foresight import usable  # noqa: E402
from run_m5_planner import sim_reactive_masks  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.schemes import simulate  # noqa: E402


def per_job(index, out: np.ndarray, dt: float) -> dict[str, float]:
    minutes = out.shape[1] * dt / 60.0
    acc: dict[str, list[float]] = {}
    for lane, (_, job, _u) in enumerate(index):
        acc.setdefault("/".join(str(x) for x in job), []).append(float(out[lane].sum()) * dt / minutes)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    m3 = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    hyb = json.loads((ROOT / "results" / "M3" / "hybrid.json").read_text())
    gen = json.loads((ROOT / "results" / "M3" / "genie.json").read_text())
    gen2 = json.loads((ROOT / "results" / "M3" / "genie2.json").read_text())
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    add_genie(built)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, eval_jobs, rw, bw, info["extra_loss_db"])
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    min_start = float(rw["e2"]["loop_delay_s"]) + R.DT_COMM
    out = {"definition": __doc__, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        snr_m = {j: snr_all[j][mi] for j in eval_jobs}
        schemes: dict[str, dict[str, float]] = {}
        # oracles on the same lanes (A3 lanes; SNR does not depend on the scheme)
        a3p = m3["tuned"][mi]["a3"]["params"]
        lanes, index = R.make_lanes(eval_jobs, [a3p], snr_m, built, "a3", {}, rw, None)
        bad = ~usable(lanes.snr_db, bw, rate_req)
        ca_any, _ = viterbi(bad, tau, 1)
        ca_ep, _ = viterbi(bad, tau, rs)
        oracle_any = per_job(index, ca_any, R.DT_COMM)
        oracle_ep = per_job(index, ca_ep, R.DT_COMM)
        inst = per_job(index, bad.all(-1), R.DT_COMM)

        def sim_pj(lanes_, index_):
            sim = simulate(lanes_, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
            return per_job(index_, sim["outage_req"], R.DT_COMM)

        schemes["A3 (main)"] = sim_pj(lanes, index)
        xp = m3["tuned"][mi]["xapp"]["params"]
        ln, ix = R.make_lanes(eval_jobs, [xp], snr_m, built, "xapp", {k: xp[k] for k in ("offset_db", "hysteresis_db", "ttt_s")}, rw, None)
        schemes["xApp + A3"] = sim_pj(ln, ix)
        hp = hyb["margins"][mi]["tuned"]["hybrid_joint"]["params"]
        ln, ix = R.make_lanes(eval_jobs, [hp], snr_m, built, "xapp", {}, rw, None, hybrid=True)
        schemes["hybrid (joint)"] = sim_pj(ln, ix)
        gp = gen["margins"][mi]["tau_ho"]["0.020"]["genie_no_overhead_params"]
        ln, ix = R.make_lanes(eval_jobs, [gp], snr_m, built, "xapp", {k: gp[k] for k in ("offset_db", "hysteresis_db", "ttt_s")}, rw, None)
        ln.overhead = np.zeros_like(ln.overhead)
        schemes["genie + A3"] = sim_pj(ln, ix)
        op = gen2["margins"][mi]["tau_ho"]["0.020"]["onset_genie_no_overhead_params"]
        ln, ix = R.make_lanes(eval_jobs, [op], snr_m, built, "xapp", {k: op[k] for k in ("offset_db", "hysteresis_db", "ttt_s")}, rw, None, min_start_s=min_start)
        ln.overhead = np.zeros_like(ln.overhead)
        schemes["onset genie"] = sim_pj(ln, ix)
        pr = pl["margins"][mi]
        for kind, name in (("a3_wide", "A3 (wide)"), ("a5", "A5")):
            ln, ix, sim = sim_reactive_masks(eval_jobs, pr[kind]["params"], "a5" if kind == "a5" else "a3", snr_m, built, rw, bw, rate_req, info["snr_req_db"])
            schemes[name] = per_job(ix, sim["outage_req"], R.DT_COMM)
            if any(abs(schemes[name][j] - pr[kind]["per_job"][j]) > 1e-9 for j in schemes[name]):
                raise SystemExit(f"{lab} {name}: re-simulation differs from planner.json per-job outage")
        for fam, name in (("genie_planner", "genie-planner"), ("sensing_planner", "sensing-planner"), ("diag_trueloss_planner", "true-LoS-loss planner")):
            for h, v in pr[fam].items():
                schemes[f"{name} H={h}"] = v["per_job"]
        res = {"label": lab, "checks": {}}
        for name, pj in schemes.items():
            below_any = [j for j in pj if pj[j] < oracle_any[j] - 1e-9]
            below_ep = [j for j in pj if pj[j] < oracle_ep[j] - 1e-9]
            res["checks"][name] = {"jobs_below_anystep_oracle": below_any, "jobs_below_epoch_oracle": len(below_ep),
                                   "max_excess_below_epoch": max([oracle_ep[j] - pj[j] for j in below_ep], default=0.0)}
        res["oracle_any_mean"] = float(np.mean(list(oracle_any.values())))
        res["oracle_epoch_mean"] = float(np.mean(list(oracle_ep.values())))
        res["instantaneous_mean"] = float(np.mean(list(inst.values())))
        res["any_le_epoch_every_job"] = all(oracle_any[j] <= oracle_ep[j] + 1e-12 for j in oracle_any)
        res["inst_le_any_every_job"] = all(inst[j] <= oracle_any[j] + 1e-12 for j in inst)
        out["margins"].append(res)
        viol = {n: len(c["jobs_below_anystep_oracle"]) for n, c in res["checks"].items() if c["jobs_below_anystep_oracle"]}
        ep = {n: c["jobs_below_epoch_oracle"] for n, c in res["checks"].items() if c["jobs_below_epoch_oracle"]}
        print(f"{lab}: any-step oracle {res['oracle_any_mean']:.3f} epoch {res['oracle_epoch_mean']:.3f} inst {res['instantaneous_mean']:.3f} | "
              f"violations of any-step bound: {viol or 'none'} | jobs below epoch oracle: {ep or 'none'} | "
              f"ordering ok: {res['any_le_epoch_every_job'] and res['inst_le_any_every_job']}", flush=True)
    dest = ROOT / "results" / "M5" / "review_a1.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
