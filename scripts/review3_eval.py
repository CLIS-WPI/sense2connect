"""ADDED AFTER THE THIRD EXTERNAL REVIEW: extra evaluations with the frozen code (v1.1-freeze).

No method or parameter changes: only frozen functions are called, with
the frozen parameters (tuned values and horizons read from the results of
the frozen pipeline).

1. "A5 + sensing overhead": A5 (tuned per margin, results/M5/planner.json)
   with the sensing overhead charged and no planner (all-zero advance and
   veto tables), via review_b6_robust.run_robust -- the same variant as
   "A5 + overhead" in scripts/review_b6_diag.py, here with per-job outages.
2. Dense break-even points (perfect-track planner, H fixed per margin as in
   scripts/review2_sweeps.py, same random-draw seeding by condition name and
   job):
   - UE position error, white Gaussian, sigma per horizontal axis in
     {0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2} m, perfect tracks;
   - blocker position error, realistic model (along-line AR(1); for the
     map-constrained classes the error is on the along-line axis only),
     sigma in {0.05, 0.1, 0.125, 0.15, 0.2, 0.3} m, UE position exact.
   Check first: the conditions "perfect | ue 0.1" and "R pos 0.1 | ue 0.0"
   reproduce results/M5/review2/sweeps.json per job exactly.
Uses S2C_EVAL_SET like the pipeline (run with S2C_EVAL_SET=heldout).

Writes results/M5/review3/eval.json.
"""

from __future__ import annotations

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
from review2_inject import inject_realistic  # noqa: E402
from review2_sweeps import _seed  # noqa: E402
from review_b5_ablation import tables, truth_tracks  # noqa: E402
from review_b6_robust import run_robust  # noqa: E402
from run_m5_paired import paired  # noqa: E402
from run_m5_planner import per_job_outage, run_planner  # noqa: E402
from seedsets import eval_set, load_seeds  # noqa: E402
from sim.comm.linkbudget import sensing_overhead, snr_ref_db  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402

UE_SIGMAS = (0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2)
POS_SIGMAS = (0.05, 0.1, 0.125, 0.15, 0.2, 0.3)
OUT = ROOT / "results" / "M5" / "review3" / "eval.json"


def main() -> None:
    import torch

    from xapp.predict_torch import predict_torch, ue_fixes

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    ev_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    sw = json.loads((ROOT / "results" / "M5" / "review2" / "sweeps.json").read_text())
    if sw.get("eval_set", "dev") != eval_set():
        raise SystemExit("results/M5/review2/sweeps.json is from another seed set")
    h_fixed = {k: float(v) for k, v in sw["H_fixed"].items()}
    cal = json.loads((ROOT / "results" / "M5" / "review2" / "calibration.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    built = R.build(tune_jobs + ev_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, ev_jobs, rw, bw, info["extra_loss_db"])
    b = rw["budget"]
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    n_r = int(built[ev_jobs[0]]["data"]["pred_2"].shape[0])
    taus = built[ev_jobs[0]]["data"]["taus"]
    wl = 299792458.0 / float(raw["carrier_hz"])
    labels = info["labels"]
    unb = {mi: {job: snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - info["extra_loss_db"][mi]
                for job in ev_jobs} for mi in range(len(labels))}
    or_e = {mi: R.stateless_rows(ev_jobs, {j: snr_all[j][mi] for j in ev_jobs}, built, "oracle", bw, rate_req)[1] for mi in range(len(labels))}
    out: dict[str, Any] = {"definition": __doc__, "eval_set": eval_set(), "seeds": [int(s) for s in seeds["evaluation"]], "H_fixed": h_fixed,
                           "ue_sigmas": UE_SIGMAS, "pos_sigmas": POS_SIGMAS, "a5_overhead": [], "conditions": {}}
    # 1. A5 + sensing overhead
    for mi, lab in enumerate(labels):
        snr_e = {j: snr_all[j][mi] for j in ev_jobs}
        zero = lambda ix: (np.zeros((len(ix), n_r, 2), dtype=bool), np.zeros((len(ix), n_r, 2), dtype=bool))  # noqa: E731
        rows = run_robust(ev_jobs, pl["margins"][mi]["a5"]["params"], snr_e, built, rw, bw, rate_req, or_e[mi], info["snr_req_db"], zero, ovh)
        pj = per_job_outage(rows)
        out["a5_overhead"].append({"label": lab, "outage_mean": float(np.mean(list(pj.values()))), "per_job": pj,
                                   "vs_a5": paired(pj, pl["margins"][mi]["a5"]["per_job"])})
    print(f"A5 + overhead done ({time.perf_counter() - clock:.0f} s)", flush=True)
    # 2. dense sweep points
    truth = {job: truth_tracks(raw, job, n_r) for job in ev_jobs}
    conds = [(f"perfect | ue {s}", {"ue": s}) for s in UE_SIGMAS] + [(f"R pos {s} | ue 0.0", {"pos": s}) for s in POS_SIGMAS]
    for name, c in conds:
        t0 = time.perf_counter()
        preds = {}
        for job in ev_jobs:
            rng = np.random.default_rng(_seed(name, job))
            tr = truth[job]
            if "pos" in c:
                tk = inject_realistic(tr, {"noise": ("pos", c["pos"])}, rng, cal, job[1])
                fix = tr["ue"]
            else:
                st = tr["state"]
                tk = {"state": st.copy(), "size": np.broadcast_to(tr["size"], (st.shape[0], st.shape[1], 3)).copy(), "valid": np.ones(st.shape[:2], dtype=bool)}
                fix = ue_fixes(tr["ue"], tr["ue_vel"], c["ue"], job[0] * 17 + 3)
            preds[job] = predict_torch(tk, fix, tr["ue_vel"], tr["oru"], taus, wl)
        res: dict[str, Any] = {"spec": c, "margins": []}
        for mi, lab in enumerate(labels):
            h = h_fixed[lab]
            rows = run_planner(ev_jobs, lambda ix, h=h, mi=mi: tables(ix, preds, unb[mi], taus, h, d, rs, tau, bw, rate_req, ovh),
                               {j: snr_all[j][mi] for j in ev_jobs}, built, rw, bw, rate_req, or_e[mi], ovh)
            pj = per_job_outage(rows)
            res["margins"].append({"label": lab, "H": h, "per_job": pj, "vs_a5": paired(pj, pl["margins"][mi]["a5"]["per_job"])})
        if name in sw["conditions"]:
            ref = {m["label"]: m["per_job"] for m in sw["conditions"][name]["margins"]}
            worst = max(abs(m["per_job"][k] - ref[m["label"]][k]) for m in res["margins"] for k in m["per_job"])
            res["reproduces_sweeps_json_max_abs"] = worst
            if worst > 1e-9:
                raise SystemExit(f"{name}: does not reproduce results/M5/review2/sweeps.json (max |diff| {worst})")
        out["conditions"][name] = res
        print(f"{name} [{time.perf_counter() - t0:.0f} s]: " + " ".join(f"{m['label'].split(' ')[0]}:{m['vs_a5']['mean_diff']:+.3f}" for m in res["margins"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, default=R._json) + "\n")
    print(f"wrote {OUT} in {(time.perf_counter() - clock) / 60:.1f} min")


if __name__ == "__main__":
    main()
