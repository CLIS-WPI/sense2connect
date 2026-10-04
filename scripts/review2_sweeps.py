"""ADDED AFTER THE SECOND EXTERNAL REVIEW: accuracy break-even sweeps and the error budget (Table II).

Development seeds 1001-1010 (40 jobs; the held-out seeds 2001-2010 with
S2C_EVAL_SET=heldout), no tuning: the planner is the
perfect-track planner of scripts/review_b5_ablation.py (receding horizon,
sensing overhead charged, no A3) with H fixed per margin at the value
tuned on the tuning seeds for perfect tracks (results/M5/review_b5.json,
condition "baseline"). Each condition perturbs the ground-truth tracks
(scripts/review2_inject.py) and/or the UE position fix (white Gaussian,
sigma per horizontal axis, the main runs' draws: seed * 17 + 3) and is
compared with A5 (results/M5/planner.json, same jobs) by the paired test:
two-sided Wilcoxon signed-rank p (the significance test) and the mean
per-job difference with its t-based 95 % CI (effect size).

Conditions (R = realistic model, primary; M = memoryless, limiting case):
- perfect | ue s: perfect tracks, UE error s in {0, 0.1, 0.25, 0.5, 1.0} m;
- R/M pos s | ue u, R/M vel s | ue u, R/M both s | ue u: blocker position,
  velocity, or both, s in {0.1, 0.25, 0.5, 1.0} (m, m/s), UE error u in {0, 1} m;
- R table: miss, false, noise, size, all (= all four), all + ue 1;
- M table (as in the first review, B5): miss measured, spurious 1x, noise
  measured, combined measured, combined measured + ue 1.
Random draws are seeded per (job, condition name).

Writes results/M5/review2/sweeps.json (updated after every condition; an
existing file is resumed).
"""

from __future__ import annotations

import argparse
import hashlib
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
from review2_inject import inject_memoryless, inject_realistic  # noqa: E402
from review_b5_ablation import inject as inject_b5  # noqa: E402
from review_b5_ablation import tables, truth_tracks  # noqa: E402
from run_m5_paired import paired  # noqa: E402
from run_m5_planner import per_job_outage, run_planner  # noqa: E402
from sim.comm.linkbudget import sensing_overhead, snr_ref_db  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402

SIGMAS = (0.1, 0.25, 0.5, 1.0)
OUT = ROOT / "results" / "M5" / "review2" / "sweeps.json"


def conditions() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = [(f"perfect | ue {u}", {"ue": u}) for u in (0.0,) + SIGMAS]
    for model in ("R", "M"):
        for kind in ("pos", "vel", "both"):
            for u in (0.0, 1.0):
                out += [(f"{model} {kind} {s} | ue {u}", {"model": model, "kind": kind, "s": s, "ue": u}) for s in SIGMAS]
    out += [(f"R table {k}", {"model": "R", "comp": c, "ue": 0.0}) for k, c in (
        ("miss", {"miss": True}), ("false", {"false": True}), ("noise", {"noise": "cal"}), ("size", {"size": True}),
        ("all", {"miss": True, "false": True, "noise": "cal", "size": True}))]
    out.append(("R table all + ue 1", {"model": "R", "comp": {"miss": True, "false": True, "noise": "cal", "size": True}, "ue": 1.0}))
    out += [(f"M table {k}", {"model": "B5", "cond": c, "ue": 0.0}) for k, c in (
        ("miss", {"miss": "measured"}), ("false", {"spurious": 1.0}), ("noise", {"noise": "measured"}),
        ("all", {"miss": "measured", "spurious": 1.0, "noise": "measured"}))]
    out.append(("M table all + ue 1", {"model": "B5", "cond": {"miss": "measured", "spurious": 1.0, "noise": "measured"}, "ue": 1.0}))
    return out


def _seed(name: str, job: tuple) -> int:
    h = hashlib.sha256(f"{name}|{job[0]}|{job[1]}|{job[2]}".encode()).hexdigest()
    return int(h[:15], 16)


def main() -> None:
    import torch

    from xapp.predict_torch import predict_torch, ue_fixes

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="condition names (default: all)")
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
    dev_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    b5 = json.loads((ROOT / "results" / "M5" / "review_b5.json").read_text())
    h_fixed = {m["label"]: float(m["H"]) for m in b5["conditions"]["baseline"]["margins"]}
    meas_b5 = b5["measured"]
    cal = json.loads((ROOT / "results" / "M5" / "review2" / "calibration.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    built = R.build(tune_jobs + dev_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, dev_jobs, rw, bw, info["extra_loss_db"])
    b = rw["budget"]
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    n_r = int(built[dev_jobs[0]]["data"]["pred_2"].shape[0])
    taus = built[dev_jobs[0]]["data"]["taus"]
    wl = 299792458.0 / float(raw["carrier_hz"])
    truth = {job: truth_tracks(raw, job, n_r) for job in dev_jobs}
    unb = {mi: {job: snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - info["extra_loss_db"][mi]
                for job in dev_jobs} for mi in range(len(info["labels"]))}
    or_e = {mi: R.stateless_rows(dev_jobs, {j: snr_all[j][mi] for j in dev_jobs}, built, "oracle", bw, rate_req)[1] for mi in range(len(info["labels"]))}
    from seedsets import eval_set

    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    if out.get("eval_set", "dev") != eval_set():
        out = {}  # never resume across seed sets
    out.update({"definition": __doc__, "H_fixed": h_fixed, "sigmas": SIGMAS, "eval_set": eval_set(), "seeds": [int(s) for s in seeds["evaluation"]]})
    out.setdefault("conditions", {})
    for name, c in conditions():
        if (args.only and name not in args.only) or (not args.only and name in out["conditions"]):
            continue
        t0 = time.perf_counter()
        preds = {}
        diag: dict = {}
        for job in dev_jobs:
            rng = np.random.default_rng(_seed(name, job))
            tr = truth[job]
            if c.get("model") == "R":
                comp = c.get("comp") or {"noise": (c["kind"], c["s"])}
                tk = inject_realistic(tr, comp, rng, cal, job[1], diag)
            elif c.get("model") == "M":
                tk = inject_memoryless(tr, c["kind"], c["s"], rng)
            elif c.get("model") == "B5":
                tk = inject_b5(tr, c["cond"], rng, meas_b5)
            else:
                tk = inject_memoryless(tr, "pos", 0.0, rng)
            fix = ue_fixes(tr["ue"], tr["ue_vel"], c["ue"], job[0] * 17 + 3) if c["ue"] > 0 else tr["ue"]
            preds[job] = predict_torch(tk, fix, tr["ue_vel"], tr["oru"], taus, wl)
        res: dict[str, Any] = {"spec": c, "margins": []}
        for mi, lab in enumerate(info["labels"]):
            h = h_fixed[lab]
            snr_e = {j: snr_all[j][mi] for j in dev_jobs}
            rows = run_planner(dev_jobs, lambda ix, h=h, mi=mi: tables(ix, preds, unb[mi], taus, h, d, rs, tau, bw, rate_req, ovh),
                               snr_e, built, rw, bw, rate_req, or_e[mi], ovh)
            pj = per_job_outage(rows)
            agg = R.aggregate(rows)
            res["margins"].append({"label": lab, "H": h, "outage": agg["outage_req_s_per_min"], "ho_per_min": agg["ho_per_min"],
                                   "per_job": pj, "vs_a5": paired(pj, pl["margins"][mi]["a5"]["per_job"])})
        if diag:
            res["validation"] = diag
        res["wall_s"] = time.perf_counter() - t0
        out["conditions"][name] = res
        OUT.write_text(json.dumps(out, default=R._json) + "\n")
        line = " ".join(f"{m['label'].split(' ')[0]}:{m['vs_a5']['mean_diff']:+.3f}" for m in res["margins"])
        print(f"{name} [{res['wall_s']:.0f} s]: {line}", flush=True)
    print(f"wrote {OUT} in {(time.perf_counter() - clock) / 60:.1f} min")


if __name__ == "__main__":
    main()
