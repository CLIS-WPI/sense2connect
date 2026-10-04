"""ADDED AFTER EXTERNAL REVIEW: grid-aligned cost-aware oracle and per-job oracles.

Cost-aware oracle (Viterbi, union objective, tau_HO 20 ms, perfect SNR of both
cells) in two variants, per evaluation job (mean over its UEs) and margin:
- any-step: switches at any 10 ms step -- THE bound (strict lower bound for
  every scheme, scripts/review_a1_oracle.py);
- grid-aligned: switches only on the planner's decision grid, i.e. 0.1 s
  epochs offset by the E2 loop delay + 10 ms (k mod 10 = tau_E2/10 ms + 1),
  the bound for policies that can act only at E2 decision epochs.
Plus the instantaneous oracle. Checks that no planner (genie, sensing,
true-LoS-loss; every H; per-job outages from results/M5/planner.json)
beats the grid-aligned oracle on any job.

Writes results/M5/grid_oracle.json.
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
from review_a1_oracle import per_job  # noqa: E402
from run_m5_dporacle import viterbi  # noqa: E402
from run_m5_foresight import usable  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402


def main() -> None:
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    m3 = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, eval_jobs, rw, bw, info["extra_loss_db"])
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    offset = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM)) + 1
    out = {"definition": __doc__, "grid": {"epoch_steps": rs, "offset_steps": offset}, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        snr_m = {j: snr_all[j][mi] for j in eval_jobs}
        lanes, index = R.make_lanes(eval_jobs, [m3["tuned"][mi]["a3"]["params"]], snr_m, built, "a3", {}, rw, None)
        bad = ~usable(lanes.snr_db, bw, rate_req)
        o_any, _ = viterbi(bad, tau, 1)
        o_grid, _ = viterbi(bad, tau, rs, offset)
        pj = {"any": per_job(index, o_any, R.DT_COMM), "grid": per_job(index, o_grid, R.DT_COMM), "inst": per_job(index, bad.all(-1), R.DT_COMM)}
        res = {"label": lab, "margin_db": info["points_db"][mi]}
        for k, v in pj.items():
            res[k] = {"mean": float(np.mean(list(v.values()))), "ci": R.ci95(list(v.values()))["ci"], "per_job": v}
        res["grid_cost_rel"] = (res["grid"]["mean"] - res["any"]["mean"]) / res["any"]["mean"] if res["any"]["mean"] > 0 else 0.0
        viol = {}
        pr = pl["margins"][mi]
        for fam in ("genie_planner", "sensing_planner", "diag_trueloss_planner"):
            for h, v in pr[fam].items():
                below = [j for j in v["per_job"] if v["per_job"][j] < pj["grid"][j] - 1e-9]
                if below:
                    viol[f"{fam} H={h}"] = below
        res["planner_jobs_below_grid_oracle"] = viol
        res["ordering_ok"] = all(pj["inst"][j] <= pj["any"][j] + 1e-12 <= pj["grid"][j] + 2e-12 for j in pj["any"])
        out["margins"].append(res)
        print(f"{lab}: any {res['any']['mean']:.3f} grid {res['grid']['mean']:.3f} (+{100 * res['grid_cost_rel']:.1f} %) inst {res['inst']['mean']:.3f} | "
              f"planners below grid oracle: {viol or 'none'} | ordering ok {res['ordering_ok']}", flush=True)
    dest = ROOT / "results" / "M5" / "grid_oracle.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
