"""TVT T4 stage 2: blockage-aware map-aided multipath tracker on all runs of a seed set.

All trackers (jobs x UEs) run in lockstep over the 600 epochs so that the visibility of their
sources is evaluated in one batched GPU call per epoch. Inputs: multipath components
(scripts/tvt_t4_measure.py), the map (sim/tvt/sources.py faces), the walkable area
(sim/positioning/estimator_v2.walk_map), blocker tracks (perfect: scenario truth; real: paper-1
map tracker replay with covariance, sim/tvt/tracks.py).
Visibility conditions (identical tracker parameters):
  none          q = the constant prior P(visible) per LoS/NLoS (tuning seeds) -> no visibility prediction (ablation i)
  oracle        q in {0, 1} from the true UE, true blockers and the true map (ablation ii)
  pred_real     q from K samples of the predicted UE state and the REAL track posteriors, recalibrated with
                the isotonic map of T3 (horizon 0, "est" mode, fitted on the tuning seeds) (iii, the method)
  pred_perfect  q from K samples of the predicted UE state and perfect tracks (bound on (iii))
Map error: the tracker's faces are displaced by e_f ~ N(0, sigma_map^2) (vertical faces; drawn per
run, seeded), the tracker estimates the offsets with prior sigma_map.
Writes results/TVT/T4/track/<set>/<cond>_map<sigma>/<mount>_<density>_<seed>.npz (xy, P_xy, flags).
Run: python scripts/tvt_t4_track.py --set development --cond pred_real [--sigma-map 0] [--params results/TVT/T4/tuned.json]
     [--calibration results/TVT/T3/visibility.json]   (frozen copies for T7: configs/tvt_frozen/tracker.json,
     configs/tvt_frozen/visibility_calibration.json; written by scripts/tvt_freeze_params.py)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CFG = "bw400_tdoa_s1_p2_b"
OUT = ROOT / "results" / "TVT" / "T4"


def real_tracks(job, raw, det_dir: Path | None = None, cache_tag: str = "") -> dict:
    """Cached real-track posteriors of one job (sizes by the paper-1 rule, box resting on the ground).

    ``det_dir``: directory with the radar detections (default: the job's cache directory); ``cache_tag`` keeps
    the cached posteriors of such variants apart (e.g. residual self-interference, T6)."""
    from sim.scenes.config import load_yaml
    from sim.tvt.tracks import pack_posteriors, replay_with_covariance
    from xapp.tracks import load_detections

    f = OUT / "tracks" / f"{cache_tag}{job[1]}_{job[2]}_{job[0]}.npz"
    if f.exists():
        with np.load(f) as g:
            return {k: g[k] for k in g.files}
    d = det_dir or ROOT / "results" / "cache" / job[1] / job[2] / f"seed_{job[0]}"
    spec = raw["sensing_radar"]
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    tc = load_yaml(ROOT / "configs" / "tvt.yaml")["visibility"]
    det = cfg["sensing"]["budgets"][int(tc["radar_budget"])]
    frames = replay_with_covariance(load_detections(d), train=int(det["train"]), pfa=float(det["pfa"]), eps_m=float(det["eps_m"]),
                                    ghost_association_m=float(det["ghost_association_m"]), walls=[float(v) for v in spec["wall_y_m"]], spec=spec,
                                    lanes=list(raw["lanes"]), sidewalks=list(raw["sidewalks"]), tuned=cfg["sensing"]["map_tracker"])
    sizes = {k: (float(v["length_m"]), float(v["width_m"]), float(v["height_m"])) for k, v in raw["blocker_kinds"].items()}
    pk = pack_posteriors(frames[:600], sizes)
    pk["mean"][..., 2] = pk["size"][..., 2] / 2.0
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **pk)
    return pk


def run(jobs, set_name, cond, sigma_map, params, raw, p2cfg, tcfg, *, save=True, tag=None, log=True, cfg: str = CFG, tracks_fn=None,
        faces_path: Path | None = None, calibration_path: str | Path | None = None) -> dict:
    import torch  # noqa: F401

    from sim.positioning.estimator_v2 import walk_map
    from sim.scenes.traffic import prepare_scenario
    from sim.tvt.sources import faces_from_geometry, source_planes, street_faces
    from sim.tvt.tracker import MultipathTracker, TrackerParams
    from sim.tvt.tracks import truth_posteriors
    from sim.tvt.visibility import sample_blockers, visibility

    from tvt_t3_visibility import isotonic_apply

    wl = 299_792_458.0 / float(raw["carrier_hz"])
    t3 = json.loads(Path(calibration_path or ROOT / "results" / "TVT" / "T3" / "visibility.json").read_text())
    cal = {k: t3["calibration"][f"est|0|{k}"] for k in ("los", "nlos")}
    faces = street_faces(faces_from_geometry(faces_path))
    vc = tcfg["visibility"]
    K = int(vc["samples"])
    thr = float(vc["threshold_db"])
    prm = TrackerParams(**{**{k: v for k, v in params.items() if not k.startswith("_")}, "sigma_map_m": float(sigma_map)})
    q_prior = params.get("_q_prior", {"los": 0.9, "nlos": 0.7})
    prm_fields = {k: v for k, v in asdict(prm).items()}
    trackers, data = [], []
    true_N, true_A, true_V, _ = source_planes(faces, __import__("sim.tvt.sources", fromlist=["source_list"]).source_list(len(faces)))
    for job in jobs:
        f = np.load(OUT / "meas" / set_name / cfg / f"{job[1]}_{job[2]}_{job[0]}.npz")
        comp = {k[5:]: f[k] for k in f.files if k.startswith("comp_")}
        sc = prepare_scenario(raw, seed=job[0], mount=job[1], density=job[2], duration_s=60.0, dt_s=0.1)
        if "walk_rects" in p2cfg:  # general walkable area (second deployment)
            wmap = {"rects": [list(map(float, r)) for r in p2cfg["walk_rects"]], "edges": p2cfg.get("walk_edges", {})}
        else:
            wmap = walk_map(raw, sc, p2cfg)
        truth = truth_posteriors(sc, np.arange(600) * 0.1)
        rk = (tracks_fn or real_tracks)(job, raw) if cond == "pred_real" else None
        m_id = ["lamppost", "facade"].index(job[1]) if job[1] in ("lamppost", "facade") else 2 + sum(map(ord, job[1]))  # canyon draws unchanged
        rng_map = np.random.default_rng([job[0], m_id, ["low", "high"].index(job[2]), 7])
        e = rng_map.standard_normal(len(faces)) * float(sigma_map)
        e = np.where(np.array([fc["axis"] for fc in faces]) == 2, 0.0, e)
        for u in range(comp["valid"].shape[1]):
            tr = MultipathTracker(faces, f["oru"][0], float(f["period_ns"]), wmap, prm, map_error=e)
            trackers.append(tr)
            data.append({"job": job, "u": u, "comp": comp, "ue": f["ue"][:, u], "oru": f["oru"], "truth": truth, "real": rk,
                         "rng": np.random.default_rng([job[0], u, 11]), "xy": np.full((600, 2), np.nan), "Pxy": np.full((600, 2, 2), np.nan),
                         "reinit": np.zeros(600, bool), "n_src": np.zeros((600, 2), int), "los_assoc": np.zeros(600, int)})
    T = 600
    qsum, qcnt = {"los": 0.0, "nlos": 0.0}, {"los": 0, "nlos": 0}
    clock = time.perf_counter()
    t_vis = t_upd = 0.0
    for t in range(T):
        live = []
        for i, (tr, d) in enumerate(zip(trackers, data)):
            meas = [{k: d["comp"][k][t, d["u"], c] for k in ("tau_ns", "uy", "uz", "var_tau", "var_uy", "var_uz", "valid")} for c in range(tr.C)]
            d["meas"] = meas
            if tr.st is None:
                p0 = tr.global_search(meas)
                if p0 is not None:
                    tr.init(p0)
                    d["xy"][t] = tr.st.x[:2]
                    d["Pxy"][t] = tr.st.P[:2, :2]
                continue
            tr.oru = d["oru"][t]
            tr.predict(0.1)
            idx = tr.valid_sources(tr.st.x)
            d["idx"] = idx
            live.append(i)
        # visibility, batched over the live trackers
        t0 = time.perf_counter()
        qs = {}
        if live:
            if cond == "none":
                for i in live:
                    tr = trackers[i]
                    qs[i] = [np.where(np.array([len(tr.sources[s]) == 0 for s in data[i]["idx"][c]], bool), q_prior["los"], q_prior["nlos"])
                             for c in range(tr.C)]
            else:
                S = max(1, max(len(data[i]["idx"][c]) for i in live for c in range(trackers[i].C)))
                L = len(live)
                C = trackers[live[0]].C
                Nb = np.zeros((L, 1, C, S, 2, 3))
                Ab = np.zeros((L, 1, C, S, 2, 3))
                Vb = np.zeros((L, 1, C, S, 2), bool)
                pv = np.zeros((L, 1, C, S), bool)
                oru = np.zeros((L, C, 3))
                kk = 1 if cond == "oracle" else K
                ue_s = np.zeros((kk, L, 1, 3))
                bmax = max((data[i]["real"]["valid"].shape[1] if cond == "pred_real" else data[i]["truth"]["mean"].shape[1]) for i in live)
                bpos = np.zeros((kk, L, bmax, 3))
                bsz = np.ones((L, bmax, 3))
                bval = np.zeros((L, bmax), bool)
                for li, i in enumerate(live):
                    tr, d = trackers[i], data[i]
                    oru[li] = d["oru"][t]
                    if cond == "oracle":
                        N_, A_, V_ = true_N, true_A, true_V
                    else:
                        N_, A_, V_ = tr._planes(tr.st.x)
                    for c in range(C):
                        ii = d["idx"][c]
                        Nb[li, 0, c, :len(ii)] = N_[ii]
                        Ab[li, 0, c, :len(ii)] = A_[ii]
                        Vb[li, 0, c, :len(ii)] = V_[ii]
                        pv[li, 0, c, :len(ii)] = True
                    if cond == "oracle":
                        ue_s[0, li, 0] = d["ue"][t]
                        tb = d["truth"]
                        nb = tb["mean"].shape[1]
                        bpos[0, li, :nb] = tb["mean"][t, :, :3]
                        bsz[li, :nb] = tb["size"][t]
                        bval[li, :nb] = True
                    else:
                        ue_s[:, li, 0] = tr.predicted_samples(K, d["rng"])
                        src = d["real"] if cond == "pred_real" else d["truth"]
                        nb = src["mean"].shape[1]
                        bpos[:, li, :nb] = sample_blockers(src["mean"][t], src["cov"][t], 0.0, K, d["rng"])
                        bsz[li, :nb] = src["size"][t]
                        bval[li, :nb] = src["valid"][t]
                q = visibility(oru, ue_s, Nb, Ab, Vb, pv, bpos, bsz, bval, wl, thr)  # [L, 1, C, S]

                for li, i in enumerate(live):
                    tr = trackers[i]
                    out = []
                    for c in range(C):
                        ii = data[i]["idx"][c]
                        qq = q[li, 0, c, :len(ii)]
                        if cond == "pred_real":
                            los = np.array([len(tr.sources[s]) == 0 for s in ii], bool)
                            qq = np.where(los, isotonic_apply(cal["los"], qq), isotonic_apply(cal["nlos"], qq))
                        out.append(qq)
                    qs[i] = out
        for i in live:
            tr = trackers[i]
            for c in range(tr.C):
                for s_, qv in zip(data[i]["idx"][c], qs[i][c]):
                    k_ = "los" if len(tr.sources[s_]) == 0 else "nlos"
                    qsum[k_] += float(qv)
                    qcnt[k_] += 1
        t_vis += time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in live:
            tr, d = trackers[i], data[i]
            los_mask = [np.array([len(tr.sources[s]) == 0 for s in d["idx"][c]], bool) for c in range(tr.C)]
            info = tr.step(d["meas"], qs[i], d["idx"], los_mask)
            d["xy"][t] = tr.st.x[:2]
            d["Pxy"][t] = tr.st.P[:2, :2]
            d["reinit"][t] = info["reinit"]
            if info["reinit"]:
                d.setdefault("reasons", []).append((t, info["reinit_reason"]))
            d["n_src"][t] = [len(x) for x in d["idx"]]
            d["los_assoc"][t] = info["los_assoc"]
        t_upd += time.perf_counter() - t0
        if log and t % 100 == 0:
            errs = [np.linalg.norm(d["xy"][t] - d["ue"][t, :2]) for d in data if np.isfinite(d["xy"][t]).all()]
            print(f"  t={t}: median error {np.median(errs) if errs else float('nan'):.3f} m, vis {t_vis:.0f} s, update {t_upd:.0f} s", flush=True)
    if log:
        for d in data:
            print("  reinit", d["job"], d["u"], d.get("reasons", [])[:12], flush=True)
    n_steps = sum(int(np.isfinite(d["xy"]).all(-1).sum()) for d in data)
    timing = {"visibility_s_per_tracker_epoch": t_vis / max(n_steps, 1), "update_s_per_tracker_epoch": t_upd / max(n_steps, 1), "wall_s": time.perf_counter() - clock}
    res = {}
    for d in data:
        key = d["job"]
        res.setdefault(key, {"xy": np.full((600, 2, 2), np.nan), "P": np.full((600, 2, 2, 2), np.nan), "ue": np.zeros((600, 2, 3)),
                             "reinit": np.zeros((600, 2), bool), "n_src": np.zeros((600, 2, 2), int), "los_assoc": np.zeros((600, 2), int)})
        r = res[key]
        r["xy"][:, d["u"]] = d["xy"]
        r["P"][:, d["u"]] = d["Pxy"]
        r["ue"][:, d["u"]] = np.concatenate([d["ue"][:, :2], np.full((600, 1), 1.5)], -1)
        r["reinit"][:, d["u"]] = d["reinit"]
        r["n_src"][:, d["u"]] = d["n_src"]
        r["los_assoc"][:, d["u"]] = d["los_assoc"]
    if save:
        od = OUT / "track" / set_name / (tag or (f"{cond}_map{float(sigma_map):g}" + ("" if cfg == CFG else f"_{cfg}")))
        od.mkdir(parents=True, exist_ok=True)
        for job, r in res.items():
            np.savez(od / f"{job[1]}_{job[2]}_{job[0]}.npz", **r)
        (od / "meta.json").write_text(json.dumps({"cond": cond, "sigma_map": sigma_map, "params": prm_fields, "q_prior": q_prior, "timing": timing,
                                                  "jobs": [list(j) for j in jobs]}, indent=1) + "\n")
    return {"res": res, "timing": timing, "q_mean": {k: qsum[k] / max(qcnt[k], 1) for k in qsum}}


def main() -> None:
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="development", choices=["tuning", "development"])
    ap.add_argument("--cond", default="pred_real", choices=["none", "oracle", "pred_real", "pred_perfect"])
    ap.add_argument("--sigma-map", type=float, default=0.0)
    ap.add_argument("--params", default=str(OUT / "tuned.json"))
    ap.add_argument("--calibration", default=None, help="isotonic visibility calibration (default results/TVT/T3/visibility.json)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cfg", default=CFG, help="measurement configuration (scripts/tvt_t4_measure.py --cfg)")
    ap.add_argument("--scenario", default="configs/m2_scenario.yaml")
    ap.add_argument("--p2cfg", default="configs/p2.yaml")
    ap.add_argument("--faces", default=None, help="building-box geometry json (default: paper2/tables/geometry.json)")
    ap.add_argument("--mounts", nargs="*", default=["lamppost", "facade"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--si-inr", type=float, default=None, help="real tracks from the residual-SI detections of T6")
    a = ap.parse_args()
    raw = load_yaml(ROOT / a.scenario)
    p2cfg = load_yaml(ROOT / a.p2cfg)
    tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
    params = json.loads(Path(a.params).read_text())["params"] if Path(a.params).exists() else {}
    seeds = check(load()[a.set])
    jobs = [(s, m, d) for s in seeds for m in a.mounts for d in ("low", "high")]
    if a.limit:
        jobs = jobs[: a.limit]
    tfn = None
    if a.si_inr is not None:
        def tfn(job, raw_):
            return real_tracks(job, raw_, det_dir=ROOT / "results" / "TVT" / "T6" / "si" / f"inr{a.si_inr:g}" / job[1] / job[2] / f"seed_{job[0]}",
                               cache_tag=f"si{a.si_inr:g}_")
    out = run(jobs, a.set, a.cond, a.sigma_map, params, raw, p2cfg, tcfg, cfg=a.cfg, tag=a.tag, faces_path=ROOT / a.faces if a.faces else None, tracks_fn=tfn,
              calibration_path=a.calibration)
    errs = np.concatenate([np.linalg.norm(r["xy"][20:] - r["ue"][20:, :, :2], axis=-1).ravel() for r in out["res"].values()])
    print(f"{a.cond} map {a.sigma_map}: median {np.nanmedian(errs):.3f} m, p90 {np.nanpercentile(errs, 90):.3f} m, timing {out['timing']}")


if __name__ == "__main__":
    main()
