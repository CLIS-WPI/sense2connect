"""TVT T3: calibration of the path-visibility prediction (sim/tvt/visibility.py), development seeds.

Per development job and 0.1 s epoch t, for every traced path (LoS and specular reflections of the
paper-1 comm geometry; the sources of the T4 tracker) of both O-RUs and both UEs, the predicted
visibility q at epoch t + h (h in configs/tvt.yaml visibility.horizons_s) is compared with the
ground truth: model-B loss of the same path (same path_key) at t + h below 10 dB (frozen
sim/positioning/blockage.path_losses_db with the true blockers). Conditions:
- tracks: "perfect" (true blocker states, sizes and classes, zero covariance) or "real" (paper-1
  map tracker replayed from the cached radar detections of O-RU 0 with its EKF covariance;
  class-agnostic sizes as the paper-1 predictor: lane/free -> bus, sidewalk -> pedestrian;
  coverage prior for blockers outside the radar range);
- UE: "true" (exact position at t, true velocity) or "est" (paper-2 estimator A EKF output, main
  configuration, with an isotropic sigma_ue from configs/tvt.yaml; true velocity as in paper 1);
- "realcal": the real-track q passed through an isotonic map q -> P(visible) fitted on the TUNING
  seeds per (UE mode, horizon, LoS/NLoS) (the tuning seeds have no paper-2 estimates, so the "est"
  map is fitted with the true UE plus the sigma_ue spread).
K joint Monte Carlo samples. The coverage prior p_prior[O-RU, LoS/NLoS] = share of paths at the
TUNING seeds blocked (>= 10 dB) by blockers outside the radar range (sensing_range_m from O-RU 0).
Metrics (pooled and seed level): Brier score of p_block = 1 - q, AUC of p_block, reliability
diagram (10 bins), split by LoS/NLoS and by the class of the dominant blocker of blocked paths.
Writes results/TVT/T3/visibility.json and per-job arrays. Run: python scripts/tvt_t3_visibility.py [--limit N]
"""

from __future__ import annotations

import argparse
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

OUT = ROOT / "results" / "TVT" / "T3"
CLASS_OF = {"car": "car", "bus": "bus/truck", "truck": "bus/truck", "pedestrian": "pedestrian"}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def job_data(job, raw, p2cfg, tcfg, *, real: bool) -> dict:
    """Geometry, truth labels and track posteriors of one job."""
    from sim.positioning.blockage import path_losses_db
    from sim.positioning.geometry import bounce_planes
    from sim.scenes.traffic import prepare_scenario
    from sim.tvt.tracks import pack_posteriors, replay_with_covariance, truth_posteriors
    from xapp.tracks import load_detections

    seed, mount, density = job
    d = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    with np.load(d / "comm_geometry.npz") as g:
        geom = {k: g[k] for k in g.files}
    wl = float(json.loads((d / "comm_geometry_meta.json").read_text())["wavelength_m"])
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    T = geom["points_m"].shape[0]
    bl = path_losses_db(geom, sc, wl)
    N, A, V = bounce_planes(geom["points_m"], geom["n_points"].astype(np.int64), p2cfg["known_planes"])
    truth = truth_posteriors(sc, np.arange(T) * 0.1)
    kinds = [CLASS_OF.get(k, "other") for k in truth["cls"]]
    out = {"wl": wl, "geom": geom, "loss": bl["loss_db"], "dom": bl["dominant_blocker"], "N": N, "A": A, "V": V, "truth": truth, "kinds": kinds, "scenario": sc}
    if real:
        spec = raw["sensing_radar"]
        cfg = __import__("sim.scenes.config", fromlist=["load_yaml"]).load_yaml(ROOT / "configs" / "m3.yaml")
        det = cfg["sensing"]["budgets"][int(tcfg["visibility"]["radar_budget"])]
        frames = replay_with_covariance(load_detections(d), train=int(det["train"]), pfa=float(det["pfa"]), eps_m=float(det["eps_m"]),
                                        ghost_association_m=float(det["ghost_association_m"]), walls=[float(v) for v in spec["wall_y_m"]], spec=spec,
                                        lanes=list(raw["lanes"]), sidewalks=list(raw["sidewalks"]), tuned=cfg["sensing"]["map_tracker"])
        kb = raw["blocker_kinds"]
        sizes = {k: (float(v["length_m"]), float(v["width_m"]), float(v["height_m"])) for k, v in kb.items()}
        pk = pack_posteriors(frames[:T], sizes)
        pk["mean"][..., 2] = pk["size"][..., 2] / 2.0  # box resting on the ground
        out["real"] = pk
        out["radar_m"] = np.asarray(load_detections(d)["radar_position_m"], dtype=np.float64)
    return out


def predict_job(jd, tracks: str, ue_mode: str, h: float, K: int, tcfg, prior: dict | None, rng, est_xy=None) -> np.ndarray:
    """q [T', U, C, P] for epochs t = 0 .. T - 1 - h/0.1 (prediction at t for t + h)."""
    from sim.tvt.visibility import sample_blockers, visibility

    geom = jd["geom"]
    T, U, C, P = geom["path_class"].shape
    hs = int(round(h / 0.1))
    Tp = T - hs
    chunk = int(tcfg["visibility"]["epoch_chunk"])
    sig = float(tcfg["visibility"]["sigma_ue_m"])
    q = np.zeros((Tp, U, C, P))
    tr = jd["truth"]
    for lo in range(0, Tp, chunk):
        hi = min(Tp, lo + chunk)
        n = hi - lo
        oru = geom["oru_position_m"][lo:hi]  # [n, C, 3]
        if ue_mode == "true":
            ue0 = tr["ue"][lo:hi]
            ue_s = np.broadcast_to(ue0 + tr["ue_vel"][lo:hi] * h, (K,) + ue0.shape).copy()
        else:
            ue0 = np.concatenate([est_xy[lo:hi], np.full(est_xy[lo:hi].shape[:-1] + (1,), 1.5)], -1)
            ue_s = ue0[None] + tr["ue_vel"][lo:hi][None] * h
            ue_s = ue_s + np.concatenate([sig * rng.standard_normal((K, n, U, 2)), np.zeros((K, n, U, 1))], -1)
        if tracks == "perfect":
            mean, cov, size, valid = tr["mean"][lo:hi], tr["cov"][lo:hi], tr["size"][lo:hi], tr["valid"][lo:hi]
        else:
            rk = jd["real"]
            mean, cov, size, valid = rk["mean"][lo:hi], rk["cov"][lo:hi], rk["size"][lo:hi], rk["valid"][lo:hi]
        bs = sample_blockers(mean, cov, h, K, rng)
        qq = visibility(oru, ue_s, jd["N"][lo:hi], jd["A"][lo:hi], jd["V"][lo:hi], geom["path_class"][lo:hi] >= 0, bs, size, valid, jd["wl"],
                        float(tcfg["visibility"]["threshold_db"]))
        q[lo:hi] = qq
    if prior is not None:
        cls = geom["path_class"][:Tp]
        for c in range(C):
            q[:, :, c] = np.where(cls[:, :, c] == 0, q[:, :, c] * (1 - prior[f"{c}|los"]), q[:, :, c] * (1 - prior[f"{c}|nlos"]))
    return q


def labels(jd, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the prediction at t of the path in slot p: label (blocked at t + h, same path_key), valid mask, dominant class index."""
    geom = jd["geom"]
    keys = geom["path_key"]
    T, U, C, P = geom["path_class"].shape
    hs = int(round(h / 0.1))
    Tp = T - hs
    lab = np.zeros((Tp, U, C, P), bool)
    ok = np.zeros((Tp, U, C, P), bool)
    dom = np.full((Tp, U, C, P), -1)
    for t in range(Tp):
        for u in range(U):
            for c in range(C):
                k1 = keys[t + hs, u, c]
                idx = {str(k): i for i, k in enumerate(k1) if geom["path_class"][t + hs, u, c, i] >= 0}
                for p in range(P):
                    if geom["path_class"][t, u, c, p] < 0:
                        continue
                    j = idx.get(str(keys[t, u, c, p]))
                    if j is None:
                        continue
                    ok[t, u, c, p] = True
                    lab[t, u, c, p] = jd["loss"][t + hs, u, c, j] >= 10.0
                    dom[t, u, c, p] = jd["dom"][t + hs, u, c, j]
    return lab, ok, dom


def coverage_prior(jobs, raw, p2cfg, tcfg) -> dict:
    """p_prior[O-RU|los/nlos]: share of paths blocked (>= 10 dB) by blockers outside the radar range, tuning seeds."""
    from sim.tvt.visibility import sample_blockers, visibility

    rng_m = float(raw["sensing_radar"]["sensing_range_m"])
    cnt, blk = {}, {}
    rng = np.random.default_rng(0)
    for job in jobs:
        jd = job_data(job, raw, p2cfg, tcfg, real=False)
        geom = jd["geom"]
        tr = jd["truth"]
        T, U, C, P = geom["path_class"].shape
        radar = geom["oru_position_m"][:, 0]  # O-RU 0
        out_cov = np.linalg.norm(tr["mean"][..., :2] - radar[:, None, :2], axis=-1) > rng_m  # [T, B]
        for lo in range(0, T, 50):
            hi = min(T, lo + 50)
            bs = sample_blockers(tr["mean"][lo:hi], tr["cov"][lo:hi], 0.0, 1, rng)
            q = visibility(geom["oru_position_m"][lo:hi], tr["ue"][lo:hi][None], jd["N"][lo:hi], jd["A"][lo:hi], jd["V"][lo:hi],
                           geom["path_class"][lo:hi] >= 0, bs, tr["size"][lo:hi], out_cov[lo:hi], jd["wl"], 10.0)
            cls = geom["path_class"][lo:hi]
            for c in range(C):
                for name, m in (("los", cls[:, :, c] == 0), ("nlos", cls[:, :, c] > 0)):
                    key = f"{c}|{name}"
                    cnt[key] = cnt.get(key, 0) + int(m.sum())
                    blk[key] = blk.get(key, 0) + float((1 - q[:, :, c])[m].sum())
    return {k: blk[k] / max(cnt[k], 1) for k in cnt}


def isotonic_fit(q: np.ndarray, vis: np.ndarray, n_bins: int = 20) -> dict:
    """Monotone (non-decreasing) map q -> P(visible): pool-adjacent-violators on equal-count bins of q."""
    if q.size == 0:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    edges = np.unique(np.quantile(q, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, max(len(edges) - 2, 0))
    xs, ys, ws = [], [], []
    for b in range(max(len(edges) - 1, 1)):
        m = idx == b
        if m.any():
            xs.append(float(q[m].mean()))
            ys.append(float(vis[m].mean()))
            ws.append(float(m.sum()))
    ys, ws = list(ys), list(ws)
    blocks = [[y, w, [x]] for x, y, w in zip(xs, ys, ws)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            y = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [y, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    x_out, y_out = [], []
    for y, _w, xx in blocks:
        for x in xx:
            x_out.append(x)
            y_out.append(y)
    return {"x": x_out, "y": y_out}


def isotonic_apply(cal: dict, q: np.ndarray) -> np.ndarray:
    return np.interp(q, cal["x"], cal["y"])


def metrics(q, lab, cls_lab) -> dict:
    pb = 1.0 - q
    res = {"n": int(lab.size), "n_blocked": int(lab.sum()), "brier": float(np.mean((pb - lab) ** 2)) if lab.size else math.nan,
           "brier_climatology": float(np.mean((lab.mean() - lab) ** 2)) if lab.size else math.nan, "auc": auc(pb, lab), "reliability": []}
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        m = (pb >= edges[i]) & ((pb < edges[i + 1]) if i < 9 else (pb <= 1.0))
        res["reliability"].append({"bin": [float(edges[i]), float(edges[i + 1])], "n": int(m.sum()),
                                   "mean_pred": float(pb[m].mean()) if m.any() else None, "observed": float(lab[m].mean()) if m.any() else None})
    res["by_class"] = {}
    for name in ("car", "bus/truck", "pedestrian"):
        sel = (~lab) | (cls_lab == name)
        res["by_class"][name] = {"n_blocked": int((lab & (cls_lab == name)).sum()), "auc": auc(pb[sel], lab[sel])}
    return res


def main() -> None:
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load
    from sim.tvt.stats import paired, seed_summary

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
    vc = tcfg["visibility"]
    s = load()
    tune = [(x, m, d) for x in check(s["tuning"]) for m in ("lamppost", "facade") for d in ("low", "high")]
    dev = [(x, m, d) for x in check(s["development"]) for m in ("lamppost", "facade") for d in ("low", "high")]
    if a.limit:
        dev, tune = dev[: a.limit], tune[: max(1, a.limit)]
    OUT.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    prior = coverage_prior(tune, raw, p2cfg, tcfg)
    print("coverage prior (tuning seeds):", {k: round(v, 4) for k, v in prior.items()}, flush=True)
    K = int(vc["samples"])
    conds = [("perfect", "true"), ("perfect", "est"), ("real", "true"), ("real", "est")]
    # isotonic recalibration of the real-track visibility, fitted on the TUNING seeds (per UE mode, horizon, LoS/NLoS)
    calib_data = {}
    for job in tune:
        jd = job_data(job, raw, p2cfg, tcfg, real=True)
        rng = np.random.default_rng([job[0], 99])
        for h in [float(x) for x in vc["horizons_s"]]:
            lab, ok, _ = labels(jd, h)
            keep = ok.copy()
            keep[:20] = False
            pc = jd["geom"]["path_class"][: lab.shape[0]]
            for ue in ("true", "est"):
                # tuning seeds have no paper-2 estimates: the "est" mode is fitted with the true UE plus the sigma_ue spread
                q = predict_job(jd, "real", "true" if ue == "true" else "est", h, K, tcfg, prior, rng,
                                jd["truth"]["ue"][..., :2] + 0.0)
                for split, m in (("los", keep & (pc == 0)), ("nlos", keep & (pc > 0))):
                    calib_data.setdefault((ue, h, split), []).append((q[m], ~lab[m]))
        print(f"calibration {job} done", flush=True)
    calib = {k: isotonic_fit(np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v]).astype(float)) for k, v in calib_data.items()}
    pooled = {}
    per_seed = {}
    for job in dev:
        t0 = time.perf_counter()
        jd = job_data(job, raw, p2cfg, tcfg, real=True)
        est = np.load(ROOT / "results" / "P2" / "est_A" / "dev" / "bw400_tdoa_s1_p2_b" / f"{job[1]}_{job[2]}_{job[0]}_track.npz")["xy_ekf"]
        est = np.where(np.isfinite(est), est, jd["truth"]["ue"][..., :2])  # before the first estimate: truth (first 2 s are excluded below)
        kinds = np.array(jd["kinds"] + ["none"])
        rng = np.random.default_rng([job[0], ["lamppost", "facade"].index(job[1]), ["low", "high"].index(job[2])])
        for h in [float(x) for x in vc["horizons_s"]]:
            lab, ok, dom = labels(jd, h)
            cls_lab = kinds[np.where(dom >= 0, dom, len(kinds) - 1)]
            keep = ok.copy()
            keep[:20] = False  # first 2 s excluded (tracker start-up), as in paper 2
            pc = jd["geom"]["path_class"][: lab.shape[0]]
            for tk, ue in conds:
                q = predict_job(jd, tk, ue, h, K, tcfg, prior if tk == "real" else None, rng, est)
                variants = [(tk, q)]
                if tk == "real":
                    qc = np.where(pc == 0, isotonic_apply(calib[(ue, h, "los")], q), isotonic_apply(calib[(ue, h, "nlos")], q))
                    variants.append(("realcal", qc))
                for name, qv in variants:
                    key = f"{name}|{ue}|{h:g}"
                    for split, m in (("all", keep), ("los", keep & (pc == 0)), ("nlos", keep & (pc > 0))):
                        pooled.setdefault(f"{key}|{split}", []).append((qv[m], lab[m], cls_lab[m]))
                        per_seed.setdefault(f"{key}|{split}", {}).setdefault(job[0], []).append((qv[m], lab[m]))
        print(f"{job}: {time.perf_counter() - t0:.0f} s", flush=True)
    res = {"definition": __doc__, "coverage_prior": prior, "calibration": {f"{k[0]}|{k[1]:g}|{k[2]}": v for k, v in calib.items()}, "config": vc,
           "pooled": {}, "seed_level": {}, "paired": {}}
    for key, parts in pooled.items():
        q = np.concatenate([x[0] for x in parts])
        lab = np.concatenate([x[1] for x in parts])
        cl = np.concatenate([x[2] for x in parts])
        res["pooled"][key] = metrics(q, lab, cl)
        br, au = [], []
        for sd, pr in sorted(per_seed[key].items()):
            qq = np.concatenate([x[0] for x in pr])
            ll = np.concatenate([x[1] for x in pr])
            pb = 1 - qq
            br.append(float(np.mean((pb - ll) ** 2)))
            au.append(auc(pb, ll))
        res["seed_level"][key] = {"brier": seed_summary(br), "auc": seed_summary(au)}
    for h in [float(x) for x in vc["horizons_s"]]:
        for ue in ("true", "est"):
            for split in ("all", "los", "nlos"):
                a_ = res["seed_level"][f"perfect|{ue}|{h:g}|{split}"]
                b_ = res["seed_level"][f"real|{ue}|{h:g}|{split}"]
                res["paired"][f"real vs perfect | {ue} | {h:g} | {split} | brier"] = paired(b_["brier"]["per_seed"], a_["brier"]["per_seed"])
                res["paired"][f"real vs perfect | {ue} | {h:g} | {split} | auc"] = paired(b_["auc"]["per_seed"], a_["auc"]["per_seed"])
                c_ = res["seed_level"][f"realcal|{ue}|{h:g}|{split}"]
                res["paired"][f"realcal vs real | {ue} | {h:g} | {split} | brier"] = paired(c_["brier"]["per_seed"], b_["brier"]["per_seed"])
    res["wall_s"] = time.perf_counter() - clock
    (OUT / "visibility.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    for key, m in res["pooled"].items():
        sl = res["seed_level"][key]
        print(f"{key:28s} n {m['n']:7d} blocked {m['n_blocked']:6d}  Brier {m['brier']:.4f} (clim {m['brier_climatology']:.4f}, seeds {sl['brier']['mean']:.4f})  "
              f"AUC {m['auc']:.3f} (seeds {sl['auc']['mean']:.3f})  by class " + " ".join(f"{c}:{v['auc']:.2f}({v['n_blocked']})" for c, v in m["by_class"].items()))


if __name__ == "__main__":
    main()
