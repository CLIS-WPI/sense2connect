"""TVT T6: path-visibility prediction under residual radar self-interference (development seeds).

As scripts/tvt_t3_visibility.py (same sources, labels, K, horizons 0 and 1 s, UE from the paper-2
estimator with sigma_ue), real tracks replayed from the detections recomputed at INR in {0, 10, 20} dB
(scripts/tvt_t6_si.py; INR 0 = nominal detections recomputed in the frozen tree). Raw and recalibrated
(T3 isotonic maps, fitted on the tuning seeds with nominal detections) Brier score and AUC, seed level.
Also the number of confirmed tracks and the radar detection count per frame.
Writes results/TVT/T6/si_visibility.json. Run: python scripts/tvt_t6_si_eval.py
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


def main() -> None:
    import tvt_t3_visibility as V3
    import tvt_t4_track as T4
    from sim.scenes.config import load_yaml
    from sim.tvt.seeds import check, load
    from sim.tvt.stats import paired, seed_summary

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    tcfg = load_yaml(ROOT / "configs" / "tvt.yaml")
    t3 = json.loads((ROOT / "results" / "TVT" / "T3" / "visibility.json").read_text())
    K = int(tcfg["visibility"]["samples"])
    seeds = check(load()["development"])
    jobs = [(s, m, d) for s in seeds for m in ("lamppost", "facade") for d in ("low", "high")]
    inrs = sorted(float(p.name[3:]) for p in (ROOT / "results" / "TVT" / "T6" / "si").glob("inr*"))
    res = {"definition": __doc__, "inr_db": inrs, "cells": {}, "paired": {}}
    per = {}
    for job in jobs:
        jd = V3.job_data(job, raw, p2cfg, tcfg, real=False)
        est = np.load(__import__("sim.tvt.panels", fromlist=["est_track_path"]).est_track_path("development", "bw400_tdoa_s1_p2_b", job))["xy_ekf"]
        est = np.where(np.isfinite(est), est, jd["truth"]["ue"][..., :2])
        for inr in inrs:
            d = __import__("sim.tvt.panels", fromlist=["detections_dir"]).detections_dir(job[1], job[2], job[0], inr)
            rk = T4.real_tracks(job, raw, det_dir=d, cache_tag=f"si{inr:g}_")
            jd["real"] = rk
            n_tracks = float(rk["valid"].sum(1).mean())
            for h in (0.0, 1.0):
                lab, ok, _ = V3.labels(jd, h)
                keep = ok.copy()
                keep[:20] = False
                pc = jd["geom"]["path_class"][: lab.shape[0]]
                q = V3.predict_job(jd, "real", "est", h, K, tcfg, t3["coverage_prior"], np.random.default_rng([job[0], int(inr), int(h * 10)]), est)
                qc = np.where(pc == 0, V3.isotonic_apply(t3["calibration"][f"est|{h:g}|los"], q), V3.isotonic_apply(t3["calibration"][f"est|{h:g}|nlos"], q))
                for name, qq in (("raw", q), ("cal", qc)):
                    per.setdefault((inr, h, name), {}).setdefault(job[0], []).append((qq[keep], lab[keep]))
            per.setdefault((inr, "tracks"), {}).setdefault(job[0], []).append(n_tracks)
        print(job, flush=True)
    for key, by_seed in per.items():
        if key[1] == "tracks":
            res["cells"][f"inr{key[0]:g}|tracks_per_epoch"] = seed_summary([np.mean(v) for _, v in sorted(by_seed.items())])
            continue
        br, au = [], []
        for _, parts in sorted(by_seed.items()):
            q = np.concatenate([x[0] for x in parts])
            lab = np.concatenate([x[1] for x in parts])
            br.append(float(np.mean((1 - q - lab) ** 2)))
            au.append(V3.auc(1 - q, lab))
        res["cells"][f"inr{key[0]:g}|h{key[1]:g}|{key[2]}"] = {"brier": seed_summary(br), "auc": seed_summary(au)}
    for inr in inrs:
        if inr == 0:
            continue
        for h in (0.0, 1.0):
            a_ = res["cells"][f"inr{inr:g}|h{h:g}|cal"]
            b_ = res["cells"][f"inr0|h{h:g}|cal"]
            res["paired"][f"inr{inr:g} vs 0 | h{h:g} | auc"] = paired(a_["auc"]["per_seed"], b_["auc"]["per_seed"])
            res["paired"][f"inr{inr:g} vs 0 | h{h:g} | brier"] = paired(a_["brier"]["per_seed"], b_["brier"]["per_seed"])
    (ROOT / "results" / "TVT" / "T6" / "si_visibility.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    for k, v in res["cells"].items():
        print(k, {m: round(x["mean"], 4) for m, x in v.items()} if "tracks" not in k else round(v["mean"], 2))


if __name__ == "__main__":
    main()
