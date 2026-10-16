"""Paper 2 camera-ready phase 1 (DEVELOPMENT check of a stronger baseline): closing experiment with UE
positions from the TVT map-aided multipath tracker.

Tracker: branch tvt at 36223bc (last isotropic-array commit), visibility condition "none" (constant
prior, no visibility prediction), frozen parameters configs/tvt_frozen/tracker.json (tuned on seeds
101-105), measurements synthesised exactly as the paper-2 main configuration (bw400_tdoa_s1_p2_b;
scripts/tvt_t4_measure.py regression against results/P2/est). Its per-epoch output
(<tracker-dir>/<mount>_<density>_<seed>.npz, xy [T, U, 2]) is fed to the planner exactly as the
estimator's EKF output (epochs before the first fix take the first fix; later gaps forward-filled).
Conditions: perfect (regression vs paper 1), estimator A (regression vs results/P2/closing_<set>.json),
tracker. Also position-error metrics (after the first 2 s) for estimator A and the tracker: pooled and
seed-level median and p90, and the median over epochs with at least one blocked O-RU; and the error in
the 1 s (10 epochs) after each UE wrap (|x_t - x_(t-1)| > half the 80 m period in the true UE track),
pooled and seed level, with the same statistics outside these windows (both after the first 2 s).
Writes results/P2/tracker_closing_<set>.json. Derived from scripts/p2_closing.py (unchanged); its
description follows.

Paper 2 (P2-M3 / M4): closing experiment with the frozen paper-1 planner.

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

After the review (before p2-freeze2): the A5 per-job outages are computed here
with the paper-1 A5 parameters (tuned on the tuning seeds, results/M5/planner.json)
by the paper-1 run_reactive (on dev / held-out they must reproduce the stored
paper-1 A5 outages; the script aborts otherwise), so that any seed set works
(S2C_EVAL_SET = dev | heldout | heldout2, scripts/p2_seeds.py); estimator
conditions for the variants v1 (frozen), A and AB (results/P2/est[_<variant>]);
each estimator condition also "capped at p90" (every error vector shortened to the
pooled 90th-percentile error of that condition; diagnostic of the tail); tail
metrics (median, p90, p99, share > 0.5 m, RMSE) per estimator condition.

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
MAIN = "bw400_tdoa_s1_p2_b"
VARIANT_CFGS = {"v1": EST_CFGS, "A": EST_CFGS, "AB": [MAIN]}


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
    from p2_seeds import eval_tag
    from p2_seeds import load as load_seeds
    from run_m5_planner import per_job_outage, run_planner, run_reactive
    from sim.comm.linkbudget import sensing_overhead, snr_ref_db
    from sim.scenes.config import load_yaml
    from xapp.predict_torch import predict_torch, ue_fixes

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    tag = eval_tag()
    p1 = {"heldout": ROOT / "results" / "M5", "dev": ROOT / "results" / "dev" / "M5"}.get(tag)  # paper-1 results of the same seeds (none for heldout2)
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    rw = cfg["rework"]
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    ev_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())  # A5 parameters (tuning seeds; identical in every paper-1 run)
    h_fixed = {m["label"]: float(m["H"]) for m in json.loads((ROOT / "results" / "M5" / "review_b5.json").read_text())["conditions"]["baseline"]["margins"]}
    sw = json.loads((p1 / "review2" / "sweeps.json").read_text()) if p1 else None
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

    def est_fix(job, name, variant="v1"):
        d_ = "est" if variant == "v1" else f"est_{variant}"
        f = ROOT / "results" / "P2" / d_ / tag / name / f"{job[1]}_{job[2]}_{job[0]}_track.npz"
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

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker-dir", required=True, help="directory with the tracker's <mount>_<density>_<seed>.npz (xy [T, U, 2])")
    a_ = ap.parse_args()
    tdir = Path(a_.tracker_dir)

    def fill(xy):
        xy = xy.copy()
        for u in range(xy.shape[1]):
            ok = np.isfinite(xy[:, u]).all(-1)
            first = np.flatnonzero(ok)
            xy[~ok, u] = xy[first[0], u] if first.size else 0.0
            for t in range(1, xy.shape[0]):
                if not ok[t]:
                    xy[t, u] = xy[t - 1, u]
        return xy

    def trk_fix(job):
        return fill(np.load(tdir / f"{job[1]}_{job[2]}_{job[0]}.npz")["xy"])

    def blocked(job):
        with np.load(ROOT / "results" / "P2" / "est" / tag / MAIN / f"{job[1]}_{job[2]}_{job[0]}.npz") as g:
            return g["blocked"].any(-1)  # [T, U]

    metrics = {}
    for name, get in (("estimator A", lambda job: est_fix(job, MAIN, "A")), ("tracker", trk_fix)):
        per_seed, pooled, pooled_b = {}, [], []
        wrap_in, wrap_out, wrap_seed, n_wraps = [], [], {}, 0
        for job in ev_jobs:
            e_all = np.linalg.norm(get(job)[: truth[job]["ue"].shape[0]] - truth[job]["ue"][..., :2], axis=-1)  # [T, U]
            jump = np.zeros(e_all.shape, bool)
            jump[1:] = np.abs(np.diff(truth[job]["ue"][..., 0], axis=0)) > 40.0
            win = np.zeros(e_all.shape, bool)
            for t_, u_ in zip(*np.nonzero(jump)):
                win[t_:t_ + 10, u_] = True
                n_wraps += int(t_ >= 20)
            win[:20] = False
            out_ = np.ones(e_all.shape, bool)
            out_[:20] = False
            out_ &= ~win
            wrap_in.append(e_all[win])
            wrap_out.append(e_all[out_])
            wrap_seed.setdefault(job[0], ([], []))
            wrap_seed[job[0]][0].append(e_all[win])
            wrap_seed[job[0]][1].append(e_all[out_])
            e = np.linalg.norm(get(job)[20:] - truth[job]["ue"][20:, :, :2], axis=-1)
            bk = blocked(job)[20:e.shape[0] + 20]
            per_seed.setdefault(job[0], ([], []))
            per_seed[job[0]][0].append(e.ravel())
            per_seed[job[0]][1].append(e[bk].ravel())
            pooled.append(e.ravel())
            pooled_b.append(e[bk].ravel())
        e, eb = np.concatenate(pooled), np.concatenate(pooled_b)
        s = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in per_seed.items()}
        metrics[name] = {"pooled": {"median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)), "median_blocked_m": float(np.median(eb)),
                                    "n_epochs": int(e.size), "n_blocked": int(eb.size)},
                         "per_seed": {int(k): {"median_m": float(np.median(v[0])), "p90_m": float(np.percentile(v[0], 90)),
                                               "median_blocked_m": float(np.median(v[1]))} for k, v in s.items()}}
        wi, wo = np.concatenate(wrap_in), np.concatenate(wrap_out)
        ws = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in wrap_seed.items()}
        metrics[name]["wrap_1s"] = {"n_wraps": n_wraps, "n_epochs_in": int(wi.size), "share_of_epochs": float(wi.size / (wi.size + wo.size)),
                                    "pooled_in": {"median_m": float(np.median(wi)), "p90_m": float(np.percentile(wi, 90)), "mean_m": float(np.mean(wi))},
                                    "pooled_out": {"median_m": float(np.median(wo)), "p90_m": float(np.percentile(wo, 90)), "mean_m": float(np.mean(wo))},
                                    "seed_mean_in": {"median_m": float(np.mean([np.median(v[0]) for v in ws.values()])),
                                                     "p90_m": float(np.mean([np.percentile(v[0], 90) for v in ws.values()]))},
                                    "seed_mean_out": {"median_m": float(np.mean([np.median(v[1]) for v in ws.values()])),
                                                      "p90_m": float(np.mean([np.percentile(v[1], 90) for v in ws.values()]))}}
        ps = metrics[name]["per_seed"]
        metrics[name]["seed_mean"] = {k: float(np.mean([ps[s_][k] for s_ in ps])) for k in ("median_m", "p90_m", "median_blocked_m")}
    from scipy.stats import wilcoxon
    seeds_ = sorted(metrics["tracker"]["per_seed"])
    metrics["tracker_vs_A_wilcoxon_p"] = {k: float(wilcoxon([metrics["tracker"]["per_seed"][s_][k] for s_ in seeds_],
                                                            [metrics["estimator A"]["per_seed"][s_][k] for s_ in seeds_], method="exact").pvalue)
                                          for k in ("median_m", "p90_m", "median_blocked_m")}
    print(json.dumps({k: metrics[k] if k.startswith("tracker_vs") else {kk: metrics[k][kk] for kk in ("pooled", "seed_mean", "wrap_1s")} for k in metrics}, indent=1), flush=True)
    conds: dict[str, callable] = {
        "perfect": lambda job, tr: tr["ue"],
        "estimator A " + MAIN: lambda job, tr: np.concatenate([est_fix(job, MAIN, "A")[: tr["ue"].shape[0]], tr["ue"][..., 2:]], -1),
        "tracker (TVT 36223bc, no visibility)": lambda job, tr: np.concatenate([trk_fix(job)[: tr["ue"].shape[0]], tr["ue"][..., 2:]], -1),
    }
    for info_v in ():
        def mk(info_v=info_v):
            def f(job, tr):
                rng = np.random.default_rng([job[0], ["lamppost", "facade"].index(job[1]), ["low", "high"].index(job[2]), 77])
                s = peb_sigma(job, info_v)[: tr["ue"].shape[0]]
                fix = tr["ue"].copy()
                fix[..., :2] += rng.standard_normal(fix[..., :2].shape) * s[..., None]
                return fix
            return f
        conds[f"PEB {info_v}-only" if info_v == "los" else "PEB map-aided"] = mk()
    tails = {}
    for variant, cfg_list in ():
        if not (ROOT / "results" / "P2" / ("est" if variant == "v1" else f"est_{variant}") / tag / MAIN).exists():
            continue
        for name in cfg_list:
            label = f"estimator {name}" if variant == "v1" else f"estimator {variant} {name}"
            errs = [np.linalg.norm(est_fix(job, name, variant)[20:] - truth[job]["ue"][20:, :, :2], axis=-1).ravel() for job in ev_jobs]
            e = np.concatenate(errs)
            p90 = float(np.percentile(e, 90))
            tails[label] = {"median_m": float(np.median(e)), "p90_m": p90, "p99_m": float(np.percentile(e, 99)), "share_gt_0.5m": float(np.mean(e > 0.5)),
                            "rmse_m": float(np.sqrt(np.mean(e ** 2)))}

            def mk2(name=name, variant=variant, cap=None):
                def f(job, tr):
                    fix = tr["ue"].copy()
                    est = est_fix(job, name, variant)[: fix.shape[0]]
                    if cap is not None:
                        dlt = est - tr["ue"][..., :2]
                        n = np.linalg.norm(dlt, axis=-1, keepdims=True)
                        est = tr["ue"][..., :2] + dlt * np.minimum(1.0, cap / np.maximum(n, 1e-12))
                    fix[..., :2] = est
                    return fix
                return f
            conds[label] = mk2()
            if name == MAIN:
                conds[label + " capped at p90"] = mk2(cap=p90)
    a5_params = {m["label"]: m["a5"]["params"] for m in pl["margins"]}
    a5 = {}
    for mi, lab in enumerate(labels):
        rows = run_reactive(ev_jobs, [a5_params[lab]], "a5", {j: snr_all[j][mi] for j in ev_jobs}, built, rw, bw, rate_req, or_e[mi], info["snr_req_db"])
        a5[lab] = per_job_outage(rows)
    if p1 is not None:  # regression: the paper-1 A5 outages of the same seeds
        old = {m["label"]: m["a5"]["per_job"] for m in json.loads((p1 / "planner.json").read_text())["margins"]}
        worst = max(abs(a5[lab][k] - old[lab][k]) for lab in labels for k in old[lab])
        if worst > 1e-9:
            raise SystemExit(f"A5 does not reproduce the paper-1 per-job outages (max |diff| {worst})")
    out = {"definition": __doc__, "set": tag, "seeds": [int(s) for s in seeds["evaluation"]], "H_fixed": h_fixed, "tails": tails, "metrics": metrics,
           "tracker_dir": str(tdir),
           "a5_outage": {lab: float(np.mean(list(a5[lab].values()))) for lab in labels}, "conditions": {}}
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
        ref = {"perfect": "perfect | ue 0.0"}.get(cname)
        if cname.startswith("estimator A"):  # regression against the stored paper-2 closing result of the same set
            old_c = {m["label"]: m["per_job"] for m in json.loads((ROOT / "results" / "P2" / f"closing_{tag}.json").read_text())["conditions"][cname]["margins"]}
            worst_c = max(abs(m["per_job"][k] - old_c[m["label"]][k]) for m in res["margins"] for k in m["per_job"])
            res["reproduces_paper2_max_abs"] = worst_c
            if worst_c > 1e-9:
                raise SystemExit(f"{cname}: does not reproduce results/P2/closing_{tag}.json (max |diff| {worst_c})")
        if ref and sw is not None:
            old = {m["label"]: m["per_job"] for m in sw["conditions"][ref]["margins"]}
            worst = max(abs(m["per_job"][k] - old[m["label"]][k]) for m in res["margins"] for k in m["per_job"])
            res["reproduces_paper1_max_abs"] = worst
            if worst > 1e-9:
                raise SystemExit(f"{cname}: does not reproduce paper-1 '{ref}' (max |diff| {worst})")
        res["wall_s"] = time.perf_counter() - t0
        out["conditions"][cname] = res
        print(f"{cname} [{res['wall_s']:.0f} s]: " + " ".join(f"{m['label'].split(' ')[0]}:{m['vs_a5']['mean_diff']:+.3f}(p{m['vs_a5']['wilcoxon_p_two_sided']:.2g})"
                                                       for m in res["margins"]), flush=True)
    dest = ROOT / "results" / "P2" / f"tracker_closing_{tag}.json"
    dest.write_text(json.dumps(out, default=R._json) + "\n")
    print(f"wrote {dest} in {(time.perf_counter() - clock) / 60:.1f} min")


if __name__ == "__main__":
    main()
