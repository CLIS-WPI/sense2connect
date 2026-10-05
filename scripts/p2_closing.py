"""Paper 2 (P2-M3 / M4): closing experiment with the frozen paper-1 planner.

The paper-1 code is imported READ-ONLY from the working tree after checking at
run time that every paper-1 file it uses is byte-identical to tag
v1.4.1-wcnc2027 (git diff --quiet). Planner as in scripts/review3_eval.py
(perfect blocker tracks, receding horizon, H fixed per margin = the
perfect-track H tuned on the tuning seeds, results/M5/review_b5.json, sensing
overhead charged); only the UE position fed to the blockage predictor is
replaced (the UE velocity stays the true one, as in paper 1):
- "perfect": exact UE position (regression: must reproduce paper-1
  "perfect | ue 0.0" per job);
- "white 1 m": paper-1 white Gaussian error, sigma 1 m per axis, draws
  seed * 17 + 3 (regression: must reproduce "perfect | ue 1.0");
- "PEB <variant>": per-epoch Gaussian error with sigma = PEB / sqrt(2) per
  horizontal axis from results/P2/peb (LoS-only and map-aided, main
  configuration: 400 MHz, TDoA, sigma_sync 1 ns, sigma_phi 2 deg, blocked
  LoS biased); bound-level positioning, an upper reference;
- "estimator <cfg>": the P2-M2 estimator's EKF output per epoch
  (results/P2/est/<set>/<cfg>/*_track.npz); epochs before the first
  estimate take the first estimate.
Paired seed-level comparison with A5 (paper-1 per-job A5 outages of the same
jobs): exact Wilcoxon over 10 seeds, cluster-bootstrap 95 % CI over seeds
(scripts/review3_seedlevel.seed_paired, read-only); per-margin tests are
pointwise (exploratory).

Writes results/P2/closing_<set>.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TAG = "v1.4.1-wcnc2027"
PAPER1_FILES = ["scripts/run_m3.py", "scripts/review_b5_ablation.py", "scripts/run_m5_planner.py", "scripts/run_m5_paired.py", "scripts/review3_seedlevel.py",
                "scripts/seedsets.py", "xapp", "sim/comm", "sim/scenes", "sim/sensing", "configs/m2_scenario.yaml", "configs/m3.yaml", "configs/seeds.yaml",
                "configs/seeds_heldout.yaml"]
EST_CFGS = ["bw400_tdoa_s1_p2_b", "bw200_tdoa_s1_p2_b", "bw100_tdoa_s1_p2_b", "bw400_toa_s1_p2_b", "bw400_aoa_s1_p2_b", "bw400_tdoa_s0_p0_b"]


def check_paper1_frozen() -> None:
    r = subprocess.run(["git", "-c", "safe.directory=*", "diff", "--quiet", TAG, "--", *PAPER1_FILES], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"paper-1 files differ from {TAG}; refusing to run the closing experiment")


def main() -> None:
    import torch

    check_paper1_frozen()
    import run_m3 as R
    from review3_seedlevel import seed_paired
    from review_b5_ablation import tables, truth_tracks
    from run_m5_planner import per_job_outage, run_planner
    from seedsets import eval_set, load_seeds
    from sim.comm.linkbudget import sensing_overhead, snr_ref_db
    from sim.scenes.config import load_yaml
    from xapp.predict_torch import predict_torch, ue_fixes

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    tag = "heldout" if eval_set() == "heldout" else "dev"
    p1 = ROOT / "results" / ("M5" if tag == "heldout" else "dev/M5")  # paper-1 results of the same seeds
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    rw = cfg["rework"]
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    ev_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    pl = json.loads((p1 / "planner.json").read_text())
    h_fixed = {m["label"]: float(m["H"]) for m in json.loads((ROOT / "results" / "M5" / "review_b5.json").read_text())["conditions"]["baseline"]["margins"]}
    sw = json.loads((p1 / "review2" / "sweeps.json").read_text())
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
    truth = {job: truth_tracks(raw, job, n_r) for job in ev_jobs}
    from p2_peb_report import AX, idx

    def peb_sigma(job, info_v):
        f = ROOT / "results" / "P2" / "peb" / tag / f"{job[1]}_{job[2]}_{job[0]}.npz"
        peb = np.load(f)["peb"][(slice(None), slice(None)) + idx(info=info_v)]
        return np.where(np.isfinite(peb), peb, 100.0) / np.sqrt(2.0)  # [T, U] per-axis sigma

    def est_fix(job, name):
        f = ROOT / "results" / "P2" / "est" / tag / name / f"{job[1]}_{job[2]}_{job[0]}_track.npz"
        xy = np.load(f)["xy_ekf"]  # [T, U, 2]
        for u in range(xy.shape[1]):
            ok = np.isfinite(xy[:, u]).all(-1)
            first = np.flatnonzero(ok)
            xy[~ok, u] = xy[first[0], u] if first.size else 0.0
            # forward-fill any later gaps
            for t in range(1, xy.shape[0]):
                if not ok[t]:
                    xy[t, u] = xy[t - 1, u]
        return xy

    conds: dict[str, callable] = {
        "perfect": lambda job, tr: tr["ue"],
        "white 1 m": lambda job, tr: ue_fixes(tr["ue"], tr["ue_vel"], 1.0, job[0] * 17 + 3),
    }
    for info_v in ("los", "map"):
        def mk(info_v=info_v):
            def f(job, tr):
                rng = np.random.default_rng([job[0], ["lamppost", "facade"].index(job[1]), ["low", "high"].index(job[2]), 77])
                s = peb_sigma(job, info_v)[: tr["ue"].shape[0]]
                fix = tr["ue"].copy()
                fix[..., :2] += rng.standard_normal(fix[..., :2].shape) * s[..., None]
                return fix
            return f
        conds[f"PEB {info_v}-only" if info_v == "los" else "PEB map-aided"] = mk()
    for name in EST_CFGS:
        def mk2(name=name):
            def f(job, tr):
                fix = tr["ue"].copy()
                fix[..., :2] = est_fix(job, name)[: fix.shape[0]]
                return fix
            return f
        conds[f"estimator {name}"] = mk2()
    a5 = {m["label"]: m["a5"]["per_job"] for m in pl["margins"]}
    out = {"definition": __doc__, "set": tag, "seeds": [int(s) for s in seeds["evaluation"]], "H_fixed": h_fixed, "conditions": {}}
    for cname, fn in conds.items():
        t0 = time.perf_counter()
        preds = {}
        for job in ev_jobs:
            tr = truth[job]
            tk = {"state": tr["state"].copy(), "size": np.broadcast_to(tr["size"], (tr["state"].shape[0], tr["state"].shape[1], 3)).copy(),
                  "valid": np.ones(tr["state"].shape[:2], dtype=bool)}
            preds[job] = predict_torch(tk, fn(job, tr), tr["ue_vel"], tr["oru"], taus, wl)
        res = {"margins": []}
        for mi, lab in enumerate(labels):
            h = h_fixed[lab]
            rows = run_planner(ev_jobs, lambda ix, h=h, mi=mi: tables(ix, preds, unb[mi], taus, h, d, rs, tau, bw, rate_req, ovh),
                               {j: snr_all[j][mi] for j in ev_jobs}, built, rw, bw, rate_req, or_e[mi], ovh)
            pj = per_job_outage(rows)
            res["margins"].append({"label": lab, "H": h, "outage_mean": float(np.mean(list(pj.values()))), "per_job": pj, "vs_a5": seed_paired(pj, a5[lab])})
        ref = {"perfect": "perfect | ue 0.0", "white 1 m": "perfect | ue 1.0"}.get(cname)
        if ref:
            old = {m["label"]: m["per_job"] for m in sw["conditions"][ref]["margins"]}
            worst = max(abs(m["per_job"][k] - old[m["label"]][k]) for m in res["margins"] for k in m["per_job"])
            res["reproduces_paper1_max_abs"] = worst
            if worst > 1e-9:
                raise SystemExit(f"{cname}: does not reproduce paper-1 '{ref}' (max |diff| {worst})")
        res["wall_s"] = time.perf_counter() - t0
        out["conditions"][cname] = res
        print(f"{cname} [{res['wall_s']:.0f} s]: " + " ".join(f"{m['label'].split(' ')[0]}:{m['vs_a5']['mean_diff']:+.3f}(p{m['vs_a5']['wilcoxon_p_two_sided']:.2g})"
                                                       for m in res["margins"]), flush=True)
    dest = ROOT / "results" / "P2" / f"closing_{tag}.json"
    dest.write_text(json.dumps(out, default=R._json) + "\n")
    print(f"wrote {dest} in {(time.perf_counter() - clock) / 60:.1f} min")


if __name__ == "__main__":
    main()
