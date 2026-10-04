"""M3 follow-up 3 (genie bound only): foresight bound and an onset-advance genie policy.

1. Foresight bound (policy-independent): with the tuned A3 (tau_HO = 20 ms)
   on the evaluation seeds, the wrong-cell time inside 10 dB events of the
   UE -- serving cell below the service rate while the other cell was at
   or above it -- per margin and blocker class, in s/UE-min and as a
   fraction of A3's total outage_req. Prediction can recover at most this
   part of A3's outage inside events; it cannot remove handover
   interruption or the both-cells-unusable time.
2. Genie, onset-advance policy (added after the first genie result): A3 is
   always active and there is no global hold. The genie (ground-truth LoS
   loss, as in run_m3_genie.py) only advances a handover when the
   predicted blockage of the serving cell starts later than the E2 loop
   delay + 10 ms after the report (so the switch lands before onset) and
   the other cell is predicted clear; after such a handover A3's hand-back
   is blocked until the predicted end of the blockage. Grid H {0.5,1,2,3} s,
   A3 underlay = the tuned A3 of that margin and tau_HO, tuned on tuning
   seeds; with the sensing overhead and without it; tau_HO 20 ms and 0.

Writes results/M3/genie2.json.
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
from run_m3_genie import add_genie  # noqa: E402
from run_m3_hybrid import decomp_summary, decompose  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.metrics import CLASSES, lane_summary  # noqa: E402
from xapp.schemes import REASON, simulate  # noqa: E402

ONSET = {"budget": ["genie"], "horizon_s": [0.5, 1.0, 2.0, 3.0], "hold_s": [0.0]}


def foresight(jobs, a3p, snr_m, built, rw, bw, rate_req, tau_ho) -> dict[str, Any]:
    """Wrong-cell time inside 10 dB events per class, absolute and as a share of A3 outage."""
    lanes, index = R.make_lanes(jobs, [a3p], snr_m, built, "a3", {}, rw, None, tau_ho_s=tau_ho)
    sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
    se = np.minimum(np.log2(1.0 + 10.0 ** (lanes.snr_db / 10.0)), MAX_NR_SE)
    usable = bw * (1.0 - lanes.overhead)[:, None, None] * se >= rate_req
    out = sim["outage_req"]
    a = out & ~(~usable.any(-1)) & ~sim["interrupted"]
    n_t = out.shape[1]
    minutes = n_t * R.DT_COMM / 60.0
    per_job: dict[tuple, dict[str, float]] = {}
    for lane, (_, job, u) in enumerate(index):
        acc = per_job.setdefault(job, {"outage": 0.0, **{f"a_{c}": 0.0 for c in ("all",) + CLASSES}})
        acc["outage"] += float(out[lane].sum())
        masks = {c: np.zeros(n_t, dtype=bool) for c in ("all",) + CLASSES}
        for ev in built[job]["events"]:
            if ev["ue"] != u:
                continue
            masks["all"][ev["start_k"] : ev["end_k"] + 1] = True
            if ev["class"] in masks:
                masks[ev["class"]][ev["start_k"] : ev["end_k"] + 1] = True
        for c, m in masks.items():
            acc[f"a_{c}"] += float((a[lane] & m).sum())
    n_ue = {job: sum(1 for _, j, _ in index if j == job) for job in per_job}
    res: dict[str, Any] = {"a3_outage_req": R.ci95([v["outage"] * R.DT_COMM / (minutes * n_ue[j]) for j, v in per_job.items()])}
    tot_out = sum(v["outage"] for v in per_job.values())
    for c in ("all",) + CLASSES:
        res[c] = {
            "s_per_ue_min": R.ci95([v[f"a_{c}"] * R.DT_COMM / (minutes * n_ue[j]) for j, v in per_job.items()]),
            "share_of_a3_outage_per_job": R.ci95([v[f"a_{c}"] / v["outage"] for v in per_job.values() if v["outage"] > 0]),
            "share_of_a3_outage_pooled": (sum(v[f"a_{c}"] for v in per_job.values()) / tot_out) if tot_out > 0 else None,
        }
    return res


def run(jobs, combos, snr_m, built, rw, bw, rate_req, orates, *, a3, tau_ho, overhead, min_start) -> list[dict[str, Any]]:
    lanes, index = R.make_lanes(jobs, combos, snr_m, built, "xapp", a3, rw, None, tau_ho_s=tau_ho, min_start_s=min_start)
    if not overhead:
        lanes.overhead = np.zeros_like(lanes.overhead)
    sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
    evs = [[ev for ev in built[j]["events"] if ev["ue"] == u] for _, j, u in index]
    dec = decompose(lanes, sim, evs, bw, rate_req, R.DT_COMM)
    rows = []
    for lane, (ci, job, u) in enumerate(index):
        s = lane_summary(sim, lane, evs[lane], orates[job][:, u], R.DT_COMM, proactive_reasons=(REASON["xapp"],))
        s.update({"combo": ci, "job": job, "ue": u, "decomp": dec[lane]})
        rows.append(s)
    return rows


def main() -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    main_metrics = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    genie1 = json.loads((ROOT / "results" / "M3" / "genie.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    add_genie(built)
    info = R.budget(built, tune_jobs, rw, bw)
    if abs(info["margin_ref_db"] - main_metrics["budget"]["margin_ref_db"]) > 1e-9:
        raise SystemExit("reference margin differs from the main run")
    snr_all = R.all_snr(built, tune_jobs + eval_jobs, rw, bw, info["extra_loss_db"])
    combos = R._grid(ONSET)
    tau_default = float(rw["e2"]["tau_ho_s"])
    min_start = float(rw["e2"]["loop_delay_s"]) + R.DT_COMM
    out = {"labels": info["labels"], "grid": ONSET, "min_predicted_start_s": min_start, "tau_ho_default_s": tau_default, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        t0 = time.perf_counter()
        snr_t = {j: snr_all[j][mi] for j in tune_jobs}
        snr_e = {j: snr_all[j][mi] for j in eval_jobs}
        orates_t = R.stateless_rows(tune_jobs, snr_t, built, "oracle", bw, rate_req)[1]
        _, orates_e = R.stateless_rows(eval_jobs, snr_e, built, "oracle", bw, rate_req)
        res: dict[str, Any] = {"label": lab, "margin_db": info["points_db"][mi], "tau_ho": {}}
        a3_main = main_metrics["tuned"][mi]["a3"]["params"]
        res["foresight"] = foresight(eval_jobs, a3_main, snr_e, built, rw, bw, rate_req, tau_default)
        for tau in (tau_default, 0.0):
            a3p = genie1["margins"][mi]["tau_ho"][f"{tau:.3f}"]["a3_params"]
            block: dict[str, Any] = {"a3_params": a3p}
            for variant, ovh in (("onset_genie", True), ("onset_genie_no_overhead", False)):
                rows_t = run(tune_jobs, combos, snr_t, built, rw, bw, rate_req, orates_t, a3=a3p, tau_ho=tau, overhead=ovh, min_start=min_start)
                obj = R.objective(rows_t, len(combos))
                i = min(range(len(combos)), key=lambda c: (obj[c][0], obj[c][1], c))
                params = {**a3p, **combos[i]}
                rows = run(eval_jobs, [combos[i]], snr_e, built, rw, bw, rate_req, orates_e, a3=a3p, tau_ho=tau, overhead=ovh, min_start=min_start)
                block[variant] = R.aggregate(rows)
                block[f"{variant}_decomp"] = decomp_summary(rows)
                block[f"{variant}_params"] = params
                block[f"{variant}_tuning_objective"] = obj[i]
            res["tau_ho"][f"{tau:.3f}"] = block
        res["wall_s"] = time.perf_counter() - t0
        out["margins"].append(res)
        b = res["tau_ho"][f"{tau_default:.3f}"]
        g1 = genie1["margins"][mi]["tau_ho"][f"{tau_default:.3f}"]
        print(
            f"{lab}: A3 {g1['a3']['outage_req_s_per_min']['mean']:.3f} | genie(1st) {g1['genie_no_overhead']['outage_req_s_per_min']['mean']:.3f} | "
            f"onset genie {b['onset_genie_no_overhead']['outage_req_s_per_min']['mean']:.3f} (ovh {b['onset_genie']['outage_req_s_per_min']['mean']:.3f}) | "
            f"foresight share {res['foresight']['all']['share_of_a3_outage_pooled']:.3f} | {res['wall_s']:.0f} s",
            flush=True,
        )
    out["wall_s"] = time.perf_counter() - clock
    (ROOT / "results" / "M3" / "genie2.json").write_text(json.dumps(out, indent=1, default=R._json) + "\n", encoding="utf-8")
    print(f"wrote results/M3/genie2.json in {out['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
