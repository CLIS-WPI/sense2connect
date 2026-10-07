"""TVT T5: risk-aware planner and baselines on the 10 ms comm timeline (best-beam service model).

Service model: configs/tvt.yaml service.model (best_beam, human decision T0), through the T0 wrapper
(scripts/tvt_service.py install). Paper-1 link budget, margins (15, 20, 25, 30 dB, 3GPP reference),
tau_HO 20 ms, E2 loop delay 20 ms, sensing overhead charged to every sensing-based scheme.
Every scheme is tuned per margin on the TUNING seeds (objective: mean outage at the service rate,
ties by handovers/min) and evaluated once on the DEVELOPMENT seeds; tuning budgets are listed in
GRIDS (no proposed scheme has a larger grid than the A5 baseline, 100 points).
Inputs (per job): blocker tracks "perfect" (truth) or "real" (paper-1 radar map tracker, covariance
from sim/tvt/tracks.py); UE position "true", "tvt" (T4 tracker, cond pred_real, sigma_map 0, with its
covariance) or "p2" (paper-2 estimator A, main configuration). Predicted LoS loss over the look-ahead
by the frozen paper-1 predictor (xapp.predict_torch.predict_torch).
Schemes:
  A3, A5          paper-1 reactive events, grids A3W (180) / A5G (100) of scripts/run_m5_planner.py
  CHO             conditional handover (see Signaling below): prepared candidate, condEventA5 execution
                  condition, no handover command; tuned over the A5 grid (100 points).
  trigger_tvt     sensing trigger in the spirit of Look-Before-Switch: hand over when a LoS blockage of
                  the serving cell is predicted within H and the other cell is predicted clear (paper-1
                  xApp trigger lanes), inputs real tracks + tvt UE; grid H x hold (12)
  trigger_learned the same trigger with a learned blockage predictor (sim/tvt/learned.py, trained on the
                  training seeds 5001-5040 only): grid H x hold x threshold (36)
  planner_p2      paper-1 planner (receding-horizon Viterbi on the predicted SNR), real tracks + paper-2
                  estimator UE; grid H (4)
  planner_tvt     the same planner with real tracks + tvt UE (4)
  risk_tvt        risk-aware planner: K joint samples of the UE (N(tvt, Sigma_t)) and of the real
                  blocker tracks (N(mean, cov)); per report the min-outage cost of "switch now" and
                  "stay now" (Viterbi over the horizon, later switches allowed) under every sample;
                  decision: switch iff mean + lambda CVaR_alpha (switch) + theta < same for stay;
                  grid H x lambda x theta (27), alpha 0.9, K 16
  riskneutral_tvt the same with lambda = 0 (J4 ablation; grid H x theta, 9)
  riskneutral_tvt_perfect    risk_tvt_perfect with lambda = 0 (J4 ablation with perfect tracks)
  planner_tvt_perfect / risk_tvt_perfect   perfect blocker tracks + tvt UE (J3 "perfect tracks")
  planner_true_perfect                       perfect tracks + true UE (paper-1 "perfect" planner)
  planner_true_future                        true UE + the TRUE future blocker trajectories over the horizon (J3 part e,
                                             configs/tvt.yaml diagnosis; vs planner_true_perfect: trajectory predictability)
  planner_white_perfect                      perfect tracks + true UE plus white Gaussian errors with the per-axis
                                             RMS of the tvt UE error over the development jobs (J3 diagnosis)
  planner_whitenw_perfect                    the same with the RMS over the epochs outside the 1 s after a UE wrap
                                             (the scenario wraps UE positions at the street ends: x jumps by 80 m)
  genie / cost-aware oracle                  references from the T0 best-beam sandbox (paper-1 code)
Metrics: outage at the service rate [s/UE-min], handovers/min, ping-pong share, outage inside
10 dB events per blocker class; seed level (per seed mean over 4 runs x 2 UEs), paired vs A5:
mean difference, bootstrap CI, exact Wilcoxon. Writes results/TVT/T5/handover.json.
--dump-steps DIR (J3 diagnosis): per margin and scheme, the per-10-ms-step timeline of the development
evaluation (serving cell, SNR of both cells, rate, outage, interruption, handovers, overhead) and the
UE positions / blockage events per job, for scripts/tvt_j3_diagnosis.py.
Signaling (human decision before the freeze): --signaling ideal (paper-1 simulator) or failure_aware
(sim/tvt/signaling.py, configs/tvt.yaml signaling: RLF with T310/N310/N311 and re-establishment,
handover-command failure, CHO with a prepared candidate), identical for every scheme; --t310 /
--q-offset / --tau-re override the failure model (pre-declared T6 sweep). CHO = condEventA5 executed
without a command, tuned over the A5 grid (100 points, the A5 budget); under ideal signaling it equals A5.
Metrics: PRIMARY outage / handovers / failures exclude the 1 s after each UE wrap (configs/tvt.yaml
evaluation:), also as tuning objective; *_unmasked fields are the supplement.
Run: python scripts/tvt_t5_handover.py [--schemes ...] [--margins ...] [--signaling failure_aware]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "results" / "TVT" / "T5"
CFG = "bw400_tdoa_s1_p2_b"
MARGINS = ("15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference")
K_SAMPLES = 16
ALPHA = 0.9
GRIDS = {
    "trigger_tvt": {"horizon_s": [0.5, 1.0, 2.0, 3.0], "hold_s": [0.2, 0.5, 1.0]},
    "trigger_learned": {"horizon_s": [0.5, 1.0, 2.0, 3.0], "hold_s": [0.2, 0.5, 1.0], "threshold": [0.3, 0.5, 0.7]},
    "planner": {"H": [0.5, 1.0, 2.0, 3.0]},
    "risk": {"H": [0.5, 1.0, 2.0], "lam": [0.0, 0.5, 1.0], "theta": [0.0, 2.0, 5.0]},
}


# ------------------------------------------------------------------ inputs
def fill_xy(xy: np.ndarray) -> np.ndarray:
    xy = xy.copy()
    for u in range(xy.shape[1]):
        ok = np.isfinite(xy[:, u]).all(-1)
        first = np.flatnonzero(ok)
        xy[~ok, u] = xy[first[0], u] if first.size else 0.0
        for t in range(1, xy.shape[0]):
            if not ok[t]:
                xy[t, u] = xy[t - 1, u]
    return xy


def tvt_xy(set_name: str, job, tag: str = "pred_real_map0") -> tuple[np.ndarray, np.ndarray]:
    f = np.load(ROOT / "results" / "TVT" / "T4" / "track" / set_name / tag / f"{job[1]}_{job[2]}_{job[0]}.npz")
    P = f["P"].copy()
    P = np.where(np.isfinite(P), P, 1.0)
    return fill_xy(f["xy"]), P


def p2_xy(set_name: str, job) -> np.ndarray | None:
    """Paper-2 estimator-A positions (None where the estimator was not run, e.g. the intersection)."""
    from sim.tvt.panels import est_track_path

    f = est_track_path(set_name, CFG, job)
    return fill_xy(np.load(f)["xy_ekf"]) if f.exists() else None


def future_truth_states(raw: dict, job, n_r: int, taus) -> np.ndarray:
    """True blocker states [n_tau, R, B, 5] at t_r + tau (exact motion model, sim.scenes.motion.states_at), velocities 0."""
    from sim.scenes.motion import states_at
    from sim.scenes.traffic import prepare_scenario

    sc = prepare_scenario(raw, seed=job[0], mount=job[1], density=job[2], duration_s=60.0, dt_s=0.1)
    bl = list(sc["vehicles"]) + list(sc["pedestrians"])
    taus = np.asarray(taus, dtype=np.float64)
    out = np.zeros((taus.size, n_r, len(bl), 5))
    cache = {}
    for r in range(n_r):
        for j, tau in enumerate(taus):
            t = round(r * 0.1 + float(tau), 6)
            if t not in cache:
                st = states_at(sc, t)
                cache[t] = np.array([[st[b["name"]]["position_m"][0], st[b["name"]]["position_m"][1], st[b["name"]]["position_m"][2], 0.0, 0.0]
                                     for b in bl])
            out[j, r] = cache[t]
    return out


def predict(tracks: dict, fix: np.ndarray, vel: np.ndarray, oru: np.ndarray, taus: np.ndarray, wl: float) -> np.ndarray:
    from xapp.predict_torch import predict_torch

    return predict_torch({"state": tracks["state"], "size": tracks["size"], "valid": tracks["valid"]}, fix, vel, oru, taus, wl)


# ------------------------------------------------------------------ risk planner core
def viterbi_costs(bad, cur, tau: int, epoch: int, device: str = "cuda"):
    """Min outage over the horizon for first action stay / switch (as run_m5_planner.plan_first_switch). Returns two [N] arrays."""
    import torch

    b = torch.as_tensor(bad, device=device, dtype=torch.int32)
    c0 = torch.as_tensor(cur, device=device, dtype=torch.long)
    n, hs, _ = b.shape
    n_p = tau + 1
    big = 10 ** 7
    idx = torch.arange(n, device=device)
    res = []
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
        res.append(cost.reshape(n, -1).min(dim=1).values.cpu().numpy().astype(np.float64))
    return res[0], res[1]


def bad_sequences(pred_rc_tau: np.ndarray, un: np.ndarray, taus, hs: int, d: int, rs: int, bw: float, rate_req: float, ovh: float) -> np.ndarray:
    """bad [R, Hs, 2] from predicted LoS loss [R, C, tau] and unblocked SNR [T, C] (as review_b5_ablation.tables)."""
    from sim.comm.phy import MAX_NR_SE

    pr = np.clip(np.nan_to_num(pred_rc_tau, posinf=200.0), 0.0, 200.0)
    n_r = pr.shape[0]
    dtau = float(taus[1] - taus[0])
    ks = (np.arange(n_r) * rs + d + 1)[:, None] + np.arange(hs)[None, :]
    ks_c = np.minimum(ks, un.shape[0] - 1)
    lead = (ks - (np.arange(n_r) * rs)[:, None]) * 0.01
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
    return bad


def risk_costs(samples: dict, unb_mi: dict, index, taus, h_s: float, d: int, rs: int, tau: int, bw, rate_req, ovh) -> dict:
    """Per lane: costs [K, R, 2 (cur), 2 (stay/switch)] under every sample."""
    hs = int(round(h_s / 0.01))
    out = {}
    for _, job, u in index:
        if (job, u) in out:
            continue
        sp = samples[job]  # [K, R, U, C, tau]
        K, n_r = sp.shape[0], sp.shape[1]
        un = unb_mi[job][:, u, :]
        bads = np.stack([bad_sequences(sp[k, :, u], un, taus, hs, d, rs, bw, rate_req, ovh) for k in range(K)])  # [K, R, Hs, 2]
        seqs = np.repeat(bads.reshape(K * n_r, hs, 2), 2, axis=0)
        cur = np.tile(np.array([0, 1]), K * n_r)
        st, sw = viterbi_costs(seqs, cur, tau, rs)
        out[(job, u)] = np.stack([st.reshape(K, n_r, 2), sw.reshape(K, n_r, 2)], -1)
    return out


def risk_table(costs: dict, index, lam: float, theta: float, alpha: float = ALPHA) -> np.ndarray:
    tabs = []
    for _, job, u in index:
        c = costs[(job, u)]  # [K, R, 2, 2]
        K = c.shape[0]
        m = c.mean(0)
        k_tail = max(1, int(math.ceil((1 - alpha) * K)))
        tail = np.sort(c, axis=0)[-k_tail:].mean(0)
        risk = m + lam * tail  # [R, 2, 2]
        tabs.append(risk[..., 1] + theta < risk[..., 0])
    return np.stack(tabs)


# ------------------------------------------------------------------ evaluation helpers
def summarize(rows, minutes_per_run: float = None) -> dict:
    by_seed = {}
    cls_out = {}
    for r in rows:
        s = r["job"][0]
        by_seed.setdefault(s, []).append(r)
    per_seed = {}
    for s, rs_ in sorted(by_seed.items()):
        per_seed[s] = {"outage": float(np.mean([x["outage_req_s_per_min"] for x in rs_])), "ho_per_min": float(np.mean([x["ho_per_min"] for x in rs_])),
                       "ping_pong": float(np.mean([x["ping_pong"] for x in rs_]))}
        for key in EXTRA_KEYS:
            if key in rs_[0]:
                per_seed[s][key] = float(np.mean([x[key] for x in rs_]))
    for r in rows:
        for ev in r["events"]:
            cls_out.setdefault(ev["class"], []).append(ev["interruption_req_s"])
    n_min = len(rows) * 0.999  # one UE-minute per lane (60 s runs)
    return {"per_seed": per_seed, "outage": float(np.mean([x["outage_req_s_per_min"] for x in rows])), "ho_per_min": float(np.mean([x["ho_per_min"] for x in rows])),
            "ping_pong": float(np.mean([x["ping_pong"] for x in rows])),
            "event_outage_by_class_s_per_ue_min": {k: float(np.sum(v) / n_min) for k, v in cls_out.items()},
            **{key: float(np.mean([x[key] for x in rows])) for key in EXTRA_KEYS if key in rows[0]}}


def objective(rows) -> tuple[float, float]:
    return (float(np.mean([r["outage_req_s_per_min"] for r in rows])), float(np.mean([r["ho_per_min"] for r in rows])))


def strip(rows):
    return [{k: v for k, v in r.items() if k in ("job", "ue", "outage_req_s_per_min", "ho_per_min", "ping_pong", "events") + EXTRA_KEYS} for r in rows]


EXTRA_KEYS = ("outage_unmasked", "ho_per_min_unmasked", "rlf_per_min", "hof_per_min", "hof_rate", "rlf_per_min_unmasked", "hof_per_min_unmasked")


def lane_extras(out_req: np.ndarray, ho_steps: np.ndarray, rlf_steps: np.ndarray, hof_steps: np.ndarray, excl: np.ndarray, dt: float) -> dict:
    """Masked (primary) and unmasked metrics of one lane; excl [T] True = excluded (1 s after a UE wrap)."""
    keep = ~excl[: out_req.size]
    m_all = out_req.size * dt / 60.0
    m_k = max(keep.sum(), 1) * dt / 60.0
    kh = keep[ho_steps].sum()
    kf = keep[hof_steps].sum()
    return {"outage_unmasked": float(out_req.sum() * dt / m_all), "outage_req_s_per_min": float((out_req & keep).sum() * dt / m_k),
            "ho_per_min_unmasked": float(ho_steps.size / m_all), "ho_per_min": float(kh / m_k),
            "rlf_per_min": float(keep[rlf_steps].sum() / m_k), "rlf_per_min_unmasked": float(rlf_steps.size / m_all),
            "hof_per_min": float(kf / m_k), "hof_per_min_unmasked": float(hof_steps.size / m_all),
            "hof_rate": float(kf / (kf + kh)) if kf + kh else 0.0}


def main() -> None:
    import torch

    import run_m3 as R
    import tvt_service
    import tvt_t4_track as T4
    from review_b5_ablation import truth_tracks
    from run_m5_planner import A3W, A5G, run_planner, run_reactive
    from sim.comm.linkbudget import sensing_overhead, snr_ref_db
    from sim.scenes.config import load_yaml
    from sim.tvt.learned import MLP, features, predict_loss
    from sim.tvt.seeds import check, load
    from sim.tvt.stats import paired

    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="*")
    ap.add_argument("--margins", nargs="*", default=list(MARGINS))
    ap.add_argument("--out", default="handover.json")
    ap.add_argument("--fixed", default=None, help="evaluate with the per-margin parameters of this result file (no tuning; T6 sweeps)")
    ap.add_argument("--e2-ms", type=float, default=None, help="E2 control-loop delay override [ms]")
    ap.add_argument("--ovh-scale", type=float, default=1.0, help="scale of the sensing overhead (CPI duty cycle)")
    ap.add_argument("--scenario", default=None, help="scenario config of the evaluation jobs (default: the paper-1 canyon)")
    ap.add_argument("--mounts", nargs="*", default=None, help="mounts of the evaluation jobs (default: lamppost facade)")
    ap.add_argument("--densities", nargs="*", default=None)
    ap.add_argument("--track-tag", default="pred_real_map0", help="T4 track directory of the tvt UE estimates")
    ap.add_argument("--track-set", default="development", help="T4 track set directory of the evaluation jobs")
    ap.add_argument("--si-inr", type=float, default=None, help="real tracks from the residual-SI detections (T6)")
    ap.add_argument("--tune-own", action="store_true", help="tune on the tuning seeds of the evaluation scenario/mounts (second deployment)")
    ap.add_argument("--signaling", default="ideal", choices=["ideal", "failure_aware"])
    ap.add_argument("--t310", type=float, default=None, help="failure model override: T310 [s] (T6 sweep)")
    ap.add_argument("--q-offset", type=float, default=0.0, help="failure model override: Qout and Qin shift [dB] (T6 sweep)")
    ap.add_argument("--tau-re", type=float, default=None, help="failure model override: re-establishment interruption [s] (T6 sweep)")
    ap.add_argument("--learned", default=str(OUT / "learned.pt"), help="learned predictor weights")
    ap.add_argument("--learned-manifest", default=None, help="json with the expected sha256 of --learned (configs/tvt_frozen/learned.json); refused on mismatch")
    ap.add_argument("--dump-steps", default=None, help="directory for the per-step timelines of the development evaluation (J3 diagnosis)")
    a = ap.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    import copy

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    raw_ev = load_yaml(ROOT / a.scenario) if a.scenario else raw
    if a.scenario:
        from sim.tvt.scene_register import register

        register()
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
    cfg = copy.deepcopy(cfg)
    rw = cfg["rework"]
    if a.e2_ms is not None:
        rw["e2"]["loop_delay_s"] = a.e2_ms / 1000.0
    rw["sensing"]["duty_cycle"] = float(rw["sensing"]["duty_cycle"]) * a.ovh_scale
    fixed = json.loads(Path(a.fixed).read_text())["schemes"] if a.fixed else None
    GRIDS["trigger_learned"]["threshold"] = [float(x) for x in tcfg["learned"]["threshold_grid"]]
    sig_cfg = None
    if a.signaling == "failure_aware":
        sig_cfg = copy.deepcopy(tcfg["signaling"])
        if a.t310 is not None:
            sig_cfg["rlf"]["t310_s"] = a.t310
        sig_cfg["rlm"]["qout_db"] = float(sig_cfg["rlm"]["qout_db"]) + a.q_offset
        sig_cfg["rlm"]["qin_db"] = float(sig_cfg["rlm"]["qin_db"]) + a.q_offset
        if a.tau_re is not None:
            sig_cfg["reestablishment"]["tau_re_s"] = a.tau_re
    elif a.t310 is not None or a.q_offset or a.tau_re is not None:
        raise SystemExit("failure-model overrides need --signaling failure_aware")
    model = tcfg["service"]["model"]
    tvt_service.install(model)
    s = load()
    tune_jobs = [(x, m, d) for x in check(s["tuning"]) for m in ((a.mounts or R.MOUNTS) if a.tune_own else R.MOUNTS) for d in R.DENSITIES]
    dev_jobs = [(x, m, d) for x in check(s["development"]) for m in (a.mounts or R.MOUNTS) for d in (a.densities or R.DENSITIES)]
    jobs = tune_jobs + dev_jobs
    pred_jobs = dev_jobs if fixed else jobs
    raw_tune = raw_ev if a.tune_own else raw
    raw_of = lambda j: raw_tune if j in tune_jobs else raw_ev  # noqa: E731
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    built = {**R.build(tune_jobs, raw_tune, cfg, 4), **R.build(dev_jobs, raw_ev, cfg, 4)}
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, jobs, rw, bw, info["extra_loss_db"])
    b = rw["budget"]
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    n_r = int(built[jobs[0]]["data"]["pred_2"].shape[0])
    taus = built[jobs[0]]["data"]["taus"]
    wl = 299792458.0 / float(raw["carrier_hz"])
    labels = info["labels"]
    mis = [labels.index(m) for m in a.margins]
    unb = {mi: {job: snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - info["extra_loss_db"][mi]
                for job in jobs} for mi in mis}
    orates = {mi: R.stateless_rows(jobs, {j: snr_all[j][mi] for j in jobs}, built, "oracle", bw, rate_req)[1] for mi in mis}
    print(f"setup {time.perf_counter() - clock:.0f} s; model {model}; reference margin {info['margin_ref_db']:.1f} dB", flush=True)

    # --- predictions per job and source
    truth = {job: truth_tracks(raw_of(job), job, n_r) for job in pred_jobs}
    if a.si_inr is not None:
        real = {job: T4.real_tracks(job, raw_ev, det_dir=__import__("sim.tvt.panels", fromlist=["detections_dir"]).detections_dir(job[1], job[2], job[0], a.si_inr),
                                    cache_tag=f"si{a.si_inr:g}_") for job in pred_jobs}
    else:
        real = {job: T4.real_tracks(job, raw_of(job)) for job in pred_jobs}
    ev_cfg = tcfg["evaluation"]
    wrap_excl = {}
    for j in pred_jobs:
        tu = truth[j]["ue"][..., :2]
        for u in range(tu.shape[1]):
            ex = np.zeros(n_r * rs, bool)
            for r_ in np.flatnonzero(np.linalg.norm(np.diff(tu[:, u], axis=0), axis=-1) > float(ev_cfg["wrap_jump_m"])) + 1:
                ex[r_ * rs: r_ * rs + int(round(float(ev_cfg["wrap_mask_s"]) / R.DT_COMM))] = True
            wrap_excl[(j, u)] = ex
    rng0 = np.random.default_rng(20261006)
    sources = {k: {} for k in ("true_perfect", "tvt_perfect", "tvt_real", "p2_real", "white_perfect", "whitenw_perfect", "true_future")}
    want_future = "planner_true_future" in (a.schemes or [])
    # per-axis RMS of the tvt UE error over the development jobs (all epochs the planner sees)
    err, keep = [], []
    for j in dev_jobs:
        tu = truth[j]["ue"][..., :2]
        err.append((tvt_xy(a.track_set, j, a.track_tag)[0][: n_r] - tu).reshape(-1, 2))
        wrap = np.zeros(tu.shape[:2], bool)
        wrap[1:] = np.linalg.norm(np.diff(tu, axis=0), axis=-1) > 10.0
        after = np.zeros_like(wrap)
        for k_ in range(10):  # 1 s of sensing epochs after a wrap
            after[k_:] |= wrap[: wrap.shape[0] - k_]
        keep.append(~after.reshape(-1))
    err, keep = np.concatenate(err), np.concatenate(keep)
    white_rms = np.sqrt(np.mean(err ** 2, axis=0))
    whitenw_rms = np.sqrt(np.mean(err[keep] ** 2, axis=0))
    print(f"tvt UE error per-axis RMS over the development jobs: x {white_rms[0]:.3f} m, y {white_rms[1]:.3f} m; outside 1 s after a UE wrap "
          f"({100 * (1 - keep.mean()):.1f} % of epochs excluded): x {whitenw_rms[0]:.3f} m, y {whitenw_rms[1]:.3f} m", flush=True)
    ue_dump = {}
    samples = {"risk_real": {}, "risk_perfect": {}}
    learned_prob = {}
    learned = None
    lp = Path(a.learned)
    if lp.exists():
        if a.learned_manifest:
            import hashlib

            want = json.loads(Path(a.learned_manifest).read_text())["sha256"]
            got = hashlib.sha256(lp.read_bytes()).hexdigest()
            if got != want:
                raise SystemExit(f"learned weights {lp}: sha256 {got} != frozen {want}")
        learned = torch.load(lp, weights_only=False)
    for job in pred_jobs:
        set_name = "tuning" if job in tune_jobs else "development"
        tr = truth[job]
        pt = {"state": tr["state"], "size": np.broadcast_to(tr["size"], tr["state"].shape[:2] + (3,)).copy(), "valid": np.ones(tr["state"].shape[:2], bool)}
        rk = real[job]
        rt = {"state": rk["mean"], "size": rk["size"], "valid": rk["valid"]}
        xy, P = tvt_xy(set_name if set_name == "tuning" else a.track_set, job, a.track_tag)
        fix_tvt = tr["ue"].copy()
        fix_tvt[..., :2] = xy[: n_r]
        p2v = p2_xy("variant" if (a.scenario or a.mounts) and set_name != "tuning" else set_name, job)
        fix_p2 = None
        if p2v is not None:
            fix_p2 = tr["ue"].copy()
            fix_p2[..., :2] = p2v[: n_r]
        sources["true_perfect"][job] = predict(pt, tr["ue"], tr["ue_vel"], tr["oru"], taus, wl)
        if want_future:  # J3 part e: blockers at their TRUE positions at t + tau (no extrapolation), UE as in true_perfect
            fs = future_truth_states(raw_of(job), job, n_r, taus)
            per_tau = [predict({"state": fs[j_], "size": pt["size"], "valid": pt["valid"]}, tr["ue"], tr["ue_vel"], tr["oru"],
                               np.asarray(taus)[j_:j_ + 1], wl) for j_ in range(len(taus))]
            sources["true_future"][job] = np.concatenate(per_tau, axis=-1)
            assert sources["true_future"][job].shape == sources["true_perfect"][job].shape, (sources["true_future"][job].shape,
                                                                                               sources["true_perfect"][job].shape)
        rng_w = np.random.default_rng([job[0], sum(map(ord, job[1])), ["low", "high"].index(job[2]), 7])
        fix_white = tr["ue"].copy()
        z = rng_w.standard_normal(fix_white[..., :2].shape)
        fix_white[..., :2] += z * white_rms
        fix_whitenw = tr["ue"].copy()
        fix_whitenw[..., :2] += z * whitenw_rms
        sources["white_perfect"][job] = predict(pt, fix_white, tr["ue_vel"], tr["oru"], taus, wl)
        sources["whitenw_perfect"][job] = predict(pt, fix_whitenw, tr["ue_vel"], tr["oru"], taus, wl)
        if job in dev_jobs:
            ue_dump[job] = {"true": tr["ue"][..., :2], "tvt": fix_tvt[..., :2], "white": fix_white[..., :2], "whitenw": fix_whitenw[..., :2]}
        sources["tvt_perfect"][job] = predict(pt, fix_tvt, tr["ue_vel"], tr["oru"], taus, wl)
        sources["tvt_real"][job] = predict(rt, fix_tvt, tr["ue_vel"], tr["oru"], taus, wl)
        if fix_p2 is not None:
            sources["p2_real"][job] = predict(rt, fix_p2, tr["ue_vel"], tr["oru"], taus, wl)
        rng = np.random.default_rng([job[0], sum(map(ord, job[1])), ["low", "high"].index(job[2]), 5])
        sr, spf = [], []
        L = np.linalg.cholesky(P[: n_r] + 1e-6 * np.eye(2))  # [R, U, 2, 2]
        cov = rk["cov"]
        Lb = np.linalg.cholesky(cov + 1e-9 * np.eye(5))
        for k in range(K_SAMPLES):
            f = tr["ue"].copy()
            f[..., :2] = xy[: n_r] + np.einsum("ruij,ruj->rui", L, rng.standard_normal((n_r, f.shape[1], 2)))
            st = rk["mean"] + np.einsum("rkij,rkj->rki", Lb, rng.standard_normal(rk["mean"].shape))
            sr.append(predict({"state": st, "size": rk["size"], "valid": rk["valid"]}, f, tr["ue_vel"], tr["oru"], taus, wl))
            spf.append(predict(pt, f, tr["ue_vel"], tr["oru"], taus, wl))
        samples["risk_real"][job] = np.stack(sr)
        samples["risk_perfect"][job] = np.stack(spf)
        if learned is not None:
            X = np.stack([features(tr["oru"], fix_tvt[r], tr["ue_vel"][r], rk, r) for r in range(n_r)])  # [R, U, C, F]
            learned_prob[job] = learned.predict_proba(X.reshape(-1, X.shape[-1])).reshape(X.shape[:-1] + (-1,))
    print(f"predictions {time.perf_counter() - clock:.0f} s", flush=True)
    del rng0

    schemes = a.schemes or ["A3", "A5", "CHO", "trigger_tvt", "trigger_learned", "planner_p2", "planner_tvt", "risk_tvt", "riskneutral_tvt",
                            "planner_tvt_perfect", "risk_tvt_perfect", "planner_true_perfect"]
    if learned is None and "trigger_learned" in schemes:
        schemes.remove("trigger_learned")
        print("no learned predictor (results/TVT/T5/learned.pt) -> trigger_learned skipped", flush=True)
    if any(j not in sources["p2_real"] for j in pred_jobs) and "planner_p2" in schemes:
        schemes.remove("planner_p2")
        print("no paper-2 estimates for the evaluation jobs -> planner_p2 skipped", flush=True)
    if fixed:
        schemes = [x for x in schemes if all(x in fixed.get(lab, {}) for lab in a.margins)]
    import run_m5_planner as M5

    from sim.tvt.signaling import simulate_fa

    cap = {}
    simcap = []
    cho_on = {"on": False}

    def sim_patched(lanes, **kw):
        out_ = simulate_fa(lanes, **kw, sig=sig_cfg, cho=np.full(lanes.snr_db.shape[0], cho_on["on"]))
        fl = out_.get("failures", {})
        simcap.append({"outage_req": out_["outage_req"], "hl": out_["handovers"]["lane"], "hs": out_["handovers"]["step"],
                       "rl": fl.get("rlf_lane", np.zeros(0, int)), "rs": fl.get("rlf_step", np.zeros(0, int)),
                       "fl": fl.get("hof_lane", np.zeros(0, int)), "fs": fl.get("hof_step", np.zeros(0, int))})
        cap["last"] = (out_, lanes)
        return out_

    M5.simulate = sim_patched
    R.simulate = sim_patched

    def attach(rows):
        """Primary (wrap-masked) and supplementary metrics into the rows of the simulate calls since the last attach."""
        n_lanes = sum(c["outage_req"].shape[0] for c in simcap)
        assert n_lanes == len(rows), (n_lanes, len(rows))
        i = 0
        for c in simcap:
            for ln in range(c["outage_req"].shape[0]):
                r = rows[i + ln]
                r.update(lane_extras(c["outage_req"][ln], c["hs"][c["hl"] == ln], c["rs"][c["rl"] == ln], c["fs"][c["fl"] == ln],
                                     wrap_excl[(r["job"], r["ue"])], R.DT_COMM))
            i += c["outage_req"].shape[0]
        simcap.clear()
        return rows

    if a.dump_steps:
        dump_dir = Path(a.dump_steps)
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_dir / "ue.npz", **{f"{k}|{j[1]}_{j[2]}_{j[0]}": v[k] for j, v in ue_dump.items() for k in v})
        (dump_dir / "events.json").write_text(json.dumps({f"{j[1]}_{j[2]}_{j[0]}": built[j]["events"] for j in dev_jobs}, default=R._json) + "\n")
        (dump_dir / "meta.json").write_text(json.dumps({"white_rms_m": white_rms.tolist(), "whitenw_rms_m": whitenw_rms.tolist(), "tau_ho_steps": tau,
                                                         "e2_delay_steps": d, "report_steps": rs,
                                                         "rate_req_bps": rate_req, "bandwidth_hz": bw, "overhead": ovh, "dt_s": R.DT_COMM}) + "\n")
    res = {"definition": __doc__, "model": model, "white_rms_m": white_rms.tolist(), "whitenw_rms_m": whitenw_rms.tolist(), "margin_ref_db": info["margin_ref_db"], "grids": GRIDS, "schemes": {},
           "signaling": a.signaling, "signaling_config": sig_cfg, "evaluation": ev_cfg,
           "overrides": {"fixed": a.fixed, "e2_ms": a.e2_ms, "ovh_scale": a.ovh_scale, "scenario": a.scenario, "mounts": a.mounts, "densities": a.densities,
                         "track_tag": a.track_tag, "si_inr": a.si_inr}}
    a3c = R._grid(A3W)
    a5c = R._grid(A5G)

    def evaluate(kind, fn_tune, fn_eval, combos, mi):
        """Tune over combos on the tuning jobs, evaluate the best on the development jobs (fixed mode: no tuning)."""
        if fixed:
            p_ = fixed[labels[mi]][kind]["params"]
            return p_, None, fn_eval(p_)
        best, best_obj, best_rows = None, None, None
        for cmb in combos:
            rows = fn_tune(cmb)
            obj = objective(rows)
            if best_obj is None or obj < best_obj:
                best, best_obj = cmb, obj
        return best, best_obj, fn_eval(best)

    for mi, lab in zip(mis, a.margins):
        snr_m = {j: snr_all[j][mi] for j in jobs}
        t_m = time.perf_counter()

        def reactive(kind, js, cmb):
            simcap.clear()
            return attach(run_reactive(js, [cmb], kind, {j: snr_m[j] for j in js}, built, rw, bw, rate_req, orates[mi], info["snr_req_db"]))

        def reactive_tune(kind, grid):
            simcap.clear()
            rows = attach(run_reactive(tune_jobs, grid, kind, {j: snr_m[j] for j in tune_jobs}, built, rw, bw, rate_req, orates[mi], info["snr_req_db"]))
            byc = {}
            for r in rows:
                byc.setdefault(r["combo"], []).append(r)
            return {ci: objective(v) for ci, v in byc.items()}

        def planner_rows(js, table_fn):
            simcap.clear()
            return attach(run_planner(js, table_fn, {j: snr_m[j] for j in js}, built, rw, bw, rate_req, orates[mi], ovh))

        def plan_table(src, h):
            from review_b5_ablation import tables

            return lambda ix: tables(ix, sources[src], unb[mi], taus, h, d, rs, tau, bw, rate_req, ovh)

        def trigger_rows(js, src_name, preds, cmb):
            for j in js:
                built[j]["data"][f"pred_{src_name}"] = preds[j]
            lanes, index = R.make_lanes(js, [{"budget": src_name, "horizon_s": cmb["horizon_s"], "hold_s": cmb["hold_s"]}], {j: snr_m[j] for j in js}, built,
                                        "xapp", {"offset_db": 1.0, "hysteresis_db": 1.0, "ttt_s": 0.04}, rw, None)
            simcap.clear()
            return attach(R.run_lanes(lanes, index, built, bw, rate_req, orates[mi], "xapp"))

        out_m = {}
        for sch in schemes:
            t0 = time.perf_counter()
            if sch in ("A3", "A5", "CHO") and fixed:
                params = fixed[lab][sch]["params"]
                obj = None
                cho_on["on"] = params["kind"] == "cho_a5"
                rows = reactive("a5" if cho_on["on"] else params["kind"], dev_jobs, {k: v for k, v in params.items() if k != "kind"})
                cho_on["on"] = False
            elif sch in ("A3", "A5"):
                kind = sch.lower()
                grid = a3c if kind == "a3" else a5c
                sc = reactive_tune(kind, grid)
                obj, ci = min((sc[i], i) for i in sc)
                cmb = grid[ci]
                rows = reactive(kind, dev_jobs, cmb)
                params = {"kind": kind, **cmb}
            elif sch == "CHO":
                cho_on["on"] = True
                sc = reactive_tune("a5", a5c)
                obj, ci = min((sc[i], i) for i in sc)
                cmb = a5c[ci]
                rows = reactive("a5", dev_jobs, cmb)
                cho_on["on"] = False
                params = {"kind": "cho_a5", **cmb}
            elif sch == "trigger_tvt":
                combos = [dict(zip(GRIDS["trigger_tvt"], v)) for v in itertools.product(*GRIDS["trigger_tvt"].values())]
                R._TRIG.clear()
                params, obj, rows = evaluate(sch, lambda c: trigger_rows(tune_jobs, "tvt", sources["tvt_real"], c),
                                             lambda c: trigger_rows(dev_jobs, "tvt", sources["tvt_real"], c), combos, mi)
            elif sch == "trigger_learned":
                combos = [dict(zip(GRIDS["trigger_learned"], v)) for v in itertools.product(*GRIDS["trigger_learned"].values())]
                lpred = {}
                for thr in GRIDS["trigger_learned"]["threshold"]:
                    lpred[thr] = {j: predict_loss(learned_prob[j], taus, thr) for j in pred_jobs}
                R._TRIG.clear()
                params, obj, rows = evaluate(sch, lambda c: trigger_rows(tune_jobs, f"learned{c['threshold']}", lpred[c["threshold"]], c),
                                             lambda c: trigger_rows(dev_jobs, f"learned{c['threshold']}", lpred[c["threshold"]], c), combos, mi)
            elif sch.startswith("planner_"):
                src = {"planner_p2": "p2_real", "planner_tvt": "tvt_real", "planner_tvt_perfect": "tvt_perfect", "planner_true_perfect": "true_perfect", "planner_true_future": "true_future",
                       "planner_white_perfect": "white_perfect", "planner_whitenw_perfect": "whitenw_perfect"}[sch]
                combos = [{"H": h} for h in GRIDS["planner"]["H"]]
                params, obj, rows = evaluate(sch, lambda c: planner_rows(tune_jobs, plan_table(src, c["H"])),
                                             lambda c: planner_rows(dev_jobs, plan_table(src, c["H"])), combos, mi)
            elif sch in ("risk_tvt", "riskneutral_tvt", "risk_tvt_perfect", "riskneutral_tvt_perfect"):
                smp = samples["risk_perfect" if sch.endswith("_perfect") else "risk_real"]
                g = dict(GRIDS["risk"])
                if sch.startswith("riskneutral_"):
                    g["lam"] = [0.0]
                best = None
                for h in ([] if fixed else g["H"]):
                    _, idx_t = R.make_lanes(tune_jobs, [{"budget": 2, "horizon_s": 0.5, "hold_s": 0.0}], {j: snr_m[j] for j in tune_jobs}, built, "xapp",
                                            {"offset_db": 0.0, "hysteresis_db": 0.0, "ttt_s": 0.04}, rw, None, hybrid=True)
                    cst = risk_costs(smp, unb[mi], idx_t, taus, h, d, rs, tau, bw, rate_req, ovh)
                    for lam_, th in itertools.product(g["lam"], g["theta"]):
                        rows_t = planner_rows(tune_jobs, lambda ix, cst=cst, lam_=lam_, th=th: risk_table(cst, ix, lam_, th))
                        obj = objective(rows_t)
                        if best is None or obj < best[0]:
                            best = (obj, {"H": h, "lam": lam_, "theta": th})
                obj, params = (None, fixed[lab][sch]["params"]) if fixed else best
                _, idx_d = R.make_lanes(dev_jobs, [{"budget": 2, "horizon_s": 0.5, "hold_s": 0.0}], {j: snr_m[j] for j in dev_jobs}, built, "xapp",
                                        {"offset_db": 0.0, "hysteresis_db": 0.0, "ttt_s": 0.04}, rw, None, hybrid=True)
                cst = risk_costs(smp, unb[mi], idx_d, taus, params["H"], d, rs, tau, bw, rate_req, ovh)
                rows = planner_rows(dev_jobs, lambda ix: risk_table(cst, ix, params["lam"], params["theta"]))
            else:
                raise SystemExit(f"unknown scheme {sch}")
            if a.dump_steps and "last" in cap:
                sm, ln = cap.pop("last")
                keys = [f"{r['job'][1]}_{r['job'][2]}_{r['job'][0]}|{r['ue']}" for r in rows]
                assert sm["serving"].shape[0] == len(keys)
                np.savez_compressed(dump_dir / f"{lab.split()[0]}_{sch}.npz", lanes=np.array(keys), serving=sm["serving"], outage_req=sm["outage_req"],
                                    interrupted=sm["interrupted"], rate=sm["rate"], snr=ln.snr_db.astype(np.float32),
                                    overhead=np.broadcast_to(np.asarray(ln.overhead, dtype=np.float64), (len(keys),)),
                                    **{f"ho_{k}": v for k, v in sm["handovers"].items()}, **{f"fail_{k}": v for k, v in sm.get("failures", {}).items()},
                                    reestablishing=sm.get("reestablishing", np.zeros((len(keys), 0), bool)))
            out_m[sch] = {"params": params, "tuning_objective": list(obj) if isinstance(obj, tuple) else obj, **summarize(rows), "rows": strip(rows),
                          "wall_s": time.perf_counter() - t0}
            print(f"{lab[:9]:9s} {sch:20s} outage {out_m[sch]['outage']:.3f} s/UE-min (unmasked {out_m[sch].get('outage_unmasked', float('nan')):.3f}), "
                  f"HO/min {out_m[sch]['ho_per_min']:.2f}, RLF/min {out_m[sch].get('rlf_per_min', 0):.3f}, HOF rate {out_m[sch].get('hof_rate', 0):.3f}, "
                  f"ping-pong {out_m[sch]['ping_pong']:.2f}, params {params} [{out_m[sch]['wall_s']:.0f} s]", flush=True)
        if "A5" in out_m:
            seeds = sorted(out_m["A5"]["per_seed"])
            for sch, v in out_m.items():
                if sch != "A5":
                    v["vs_A5"] = paired([v["per_seed"][x]["outage"] for x in seeds], [out_m["A5"]["per_seed"][x]["outage"] for x in seeds])
        for pair in (("risk_tvt", "riskneutral_tvt"), ("risk_tvt", "planner_tvt"), ("planner_tvt", "planner_p2"), ("risk_tvt_perfect", "planner_tvt_perfect"),
                     ("risk_tvt_perfect", "riskneutral_tvt_perfect")):
            if pair[0] in out_m and pair[1] in out_m:
                seeds = sorted(out_m[pair[0]]["per_seed"])
                out_m[f"{pair[0]} vs {pair[1]}"] = paired([out_m[pair[0]]["per_seed"][x]["outage"] for x in seeds], [out_m[pair[1]]["per_seed"][x]["outage"] for x in seeds])
        res["schemes"][lab] = out_m
        print(f"margin {lab} done in {(time.perf_counter() - t_m) / 60:.1f} min", flush=True)
        (OUT / a.out).write_text(json.dumps(res, default=R._json) + "\n")
    res["wall_s"] = time.perf_counter() - clock
    (OUT / a.out).write_text(json.dumps(res, default=R._json) + "\n")
    print(f"wrote {OUT / a.out} in {res['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
