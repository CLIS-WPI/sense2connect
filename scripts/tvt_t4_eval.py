"""TVT T4: evaluation of the multipath tracker on the development seeds (J1, J2).

Conditions (results/TVT/T4/track/development/<cond>_map<sigma>): visibility none / oracle /
pred_real (the method) / pred_perfect at sigma_map 0, and pred_real at sigma_map in configs/tvt.yaml
map.sigma_map_m (finite values). References: paper-2 estimator A (main configuration, results/P2/
est_A/dev), the single-epoch PEB of T1 (map-aided sigma_map 0 and LoS-only, same configuration).
Per seed (4 runs x 2 UEs, epochs after 2 s): median, p90, RMSE, share > 0.1 m; windows as in paper 2
(scripts/p2_diag_onset.py): [-1, -0.5) s and [-0.5, 0) s before the onset of a 10 dB LoS event of the
serving cell, during the event, elsewhere (paper-1 events, m3_timeline_meta.json). Paired seed-level
tests (exact Wilcoxon, bootstrap CIs): pred_real vs none (J1), oracle vs none, pred_perfect vs
pred_real, pred_real vs paper-2 A (J2), and the remaining gap: tracker median / map-aided PEB median.
Covariance consistency: share of epochs whose error lies inside the 95 % ellipse of the tracker
covariance. Writes results/TVT/T4/eval.json. Run: python scripts/tvt_t4_eval.py
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

WINDOWS = ("[-1,-0.5) s", "[-0.5,0) s", "during", "elsewhere")
CFG = "bw400_tdoa_s1_p2_b"


def window_labels(meta: dict, u: int, T: int = 600) -> np.ndarray:
    t = np.arange(T) * 0.1
    lab = np.full(T, "elsewhere", dtype=object)
    evs = [e for e in meta["events"] if e["ue"] == u]
    for e in evs:
        st = e["start_s"]
        lab[(t >= st - 1.0) & (t < st - 0.5) & (lab == "elsewhere")] = "[-1,-0.5) s"
        lab[(t >= st - 0.5) & (t < st) & ((lab == "elsewhere") | (lab == "[-1,-0.5) s"))] = "[-0.5,0) s"
    for e in evs:
        lab[(t >= e["start_s"]) & (t <= e["end_s"])] = "during"
    return lab


def main() -> None:
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load
    from sim.tvt.stats import paired, seed_summary

    tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
    seeds = check(load()["development"])
    jobs = [(s, m, d) for s in seeds for m in ("lamppost", "facade") for d in ("low", "high")]
    base = ROOT / "results" / "TVT" / "T4" / "track" / "development"
    conds = ["none_map0", "oracle_map0", "pred_real_map0", "pred_perfect_map0"] + [f"pred_real_map{float(s):g}" for s in tcfg["map"]["sigma_map_m"]
                                                                                   if np.isfinite(float(s)) and float(s) > 0]
    conds = [c for c in conds if (base / c).exists()]
    data = {}
    for c in conds:
        data[c] = {}
        for j in jobs:
            r = np.load(base / c / f"{j[1]}_{j[2]}_{j[0]}.npz")
            e = np.linalg.norm(r["xy"] - r["ue"][..., :2], axis=-1)
            d = r["xy"] - r["ue"][..., :2]
            P = r["P"]
            ok = np.isfinite(d).all(-1) & np.isfinite(P).all((-1, -2))
            nees = np.full(e.shape, np.nan)
            Pi = np.linalg.inv(np.where(ok[..., None, None], P, np.eye(2)))
            nees = np.where(ok, np.einsum("tui,tuij,tuj->tu", d, Pi, d), np.nan)
            data[c][j] = {"err": np.where(np.isfinite(e), e, 1e3), "nees": nees}
    data["paper2_A"] = {}
    for j in jobs:
        r = np.load(ROOT / "results" / "P2" / "est_A" / "dev" / CFG / f"{j[1]}_{j[2]}_{j[0]}_track.npz")
        ue = np.load(ROOT / "results" / "TVT" / "T4" / "meas" / "development" / CFG / f"{j[1]}_{j[2]}_{j[0]}.npz")["ue"]
        e = np.linalg.norm(r["xy_ekf"] - ue[..., :2], axis=-1)
        data["paper2_A"][j] = {"err": np.where(np.isfinite(e), e, 1e3)}
    for name, key in (("PEB_map0", "mapsweep__map_0"), ("PEB_los", "mapsweep__los")):
        data[name] = {}
        for j in jobs:
            g = np.load(ROOT / "results" / "TVT" / "T1" / "bound" / f"{j[1]}_{j[2]}_{j[0]}.npz")
            data[name][j] = {"err": g[key]}
    metas = {j: json.loads((ROOT / "results" / "cache" / j[1] / j[2] / f"seed_{j[0]}" / "m3_timeline_meta.json").read_text()) for j in jobs}
    res = {"definition": __doc__, "conditions": {}, "paired": {}}
    per_seed = {}
    for c, dd in data.items():
        cell = {}
        for w in ("all",) + WINDOWS:
            vals = {k: [] for k in ("median", "p90", "rmse", "share_gt_0.1", "n")}
            for s in seeds:
                v = []
                for j in jobs:
                    if j[0] != s:
                        continue
                    for u in range(2):
                        lab = window_labels(metas[j], u)
                        sel = np.arange(600) >= 20
                        if w != "all":
                            sel &= lab == w
                        v.append(dd[j]["err"][sel, u])
                v = np.concatenate(v)
                vals["n"].append(int(v.size))
                vals["median"].append(float(np.median(v)) if v.size else np.nan)
                vals["p90"].append(float(np.percentile(v, 90)) if v.size else np.nan)
                vals["rmse"].append(float(np.sqrt(np.mean(np.minimum(v, 1e3) ** 2))) if v.size else np.nan)
                vals["share_gt_0.1"].append(float(np.mean(v > 0.1)) if v.size else np.nan)
            cell[w] = {k: seed_summary(v) if k != "n" else v for k, v in vals.items()}
            per_seed[(c, w)] = vals
        if "nees" in next(iter(dd.values())):
            allnees = np.concatenate([dd[j]["nees"][20:].ravel() for j in jobs])
            cell["share_inside_95pct_ellipse"] = float(np.nanmean(allnees < 5.991))
        res["conditions"][c] = cell
    pairs = [("pred_real_map0", "none_map0"), ("oracle_map0", "none_map0"), ("pred_perfect_map0", "pred_real_map0"), ("oracle_map0", "pred_real_map0"),
             ("pred_real_map0", "paper2_A"), ("none_map0", "paper2_A"), ("oracle_map0", "paper2_A")]
    for a_, b_ in pairs:
        if a_ not in data or b_ not in data:
            continue
        for w in ("all",) + WINDOWS:
            for k in ("median", "p90", "rmse"):
                res["paired"][f"{a_} vs {b_} | {w} | {k}"] = paired(per_seed[(a_, w)][k], per_seed[(b_, w)][k])
    for c in data:
        if c.startswith("PEB"):
            continue
        for w in ("all",) + WINDOWS:
            ratio = np.array(per_seed[(c, w)]["median"]) / np.array(per_seed[("PEB_map0", w)]["median"])
            res["conditions"][c][w]["median_over_PEB_map0"] = seed_summary(ratio)
    (ROOT / "results" / "TVT" / "T4" / "eval.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("condition | window | median [m] | p90 [m] | RMSE [m] | >0.1 m | median/PEB_map")
    for c, cell in res["conditions"].items():
        for w in ("all",) + WINDOWS:
            x = cell[w]
            r_ = x.get("median_over_PEB_map0", {"mean": float("nan")})["mean"]
            print(f"{c:22s} {w:12s} {x['median']['mean']:.4f} {x['p90']['mean']:.3f} {x['rmse']['mean']:.2f} {x['share_gt_0.1']['mean']:.3f} {r_:.1f}")
        if "share_inside_95pct_ellipse" in cell:
            print(f"   inside 95 % ellipse: {cell['share_inside_95pct_ellipse']:.3f}")
    for k, v in res["paired"].items():
        if "| median" in k:
            print(f"{k:60s} diff {v['mean_diff']:+.4f} [{v['ci95_boot'][0]:+.4f}, {v['ci95_boot'][1]:+.4f}] p {v['wilcoxon_p_two_sided']:.3g}")


if __name__ == "__main__":
    main()
