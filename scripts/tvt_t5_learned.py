"""TVT T5: train the learned LoS-blockage predictor (sim/tvt/learned.py) on the TRAINING seeds only.

Training jobs: seeds 5001-5040 x 2 mounts x 2 densities (traced by scripts/tvt_training_traces.py);
4 of the 40 seeds (5037-5040) are held back for early stopping (validation), none of the tuning,
development or held-out seeds is used. Inputs per report (0.1 s), UE and cell: the real radar
track posteriors of O-RU 0 (sim/tvt/tracks.py replay; same as the planner inputs) and a UE position
estimate. The TVT tracker does not run on the training seeds (no SRS caches), so the UE estimate is
the true position plus a Gaussian error whose per-axis standard deviation is the robust spread
(1.4826 x median absolute deviation) of the TVT tracker error on the TUNING seeds; at evaluation the
TVT tracker output is used. Labels: model-B LoS loss >= 10 dB of the paper-1 10 ms timeline at the
look-aheads TAUS. Writes results/TVT/T5/learned.pt and results/TVT/T5/learned.json.
Run: python scripts/tvt_t5_learned.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "results" / "TVT" / "T5"


def tracker_sigma() -> float:
    errs = []
    for f in sorted((ROOT / "results" / "TVT" / "T4" / "track" / "tuning" / "pred_real_map0").glob("*.npz")):
        r = np.load(f)
        e = (r["xy"][20:] - r["ue"][20:, :, :2]).reshape(-1)
        errs.append(e[np.isfinite(e)])
    e = np.concatenate(errs)
    return float(1.4826 * np.median(np.abs(e - np.median(e))))


def main() -> None:
    import torch

    import run_m3 as R
    import tvt_t4_track as T4
    from sim.scenes.config import load_yaml
    from sim.tvt.learned import MLP, N_FEAT, TAUS, features, labels
    from sim.tvt.seeds import check, load
    from xapp.timeline import actor_tracks  # noqa: F401  (build uses it)

    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    seeds = check(load()["training"])
    val_seeds = seeds[-4:]
    jobs = [(s, m, d) for s in seeds for m in R.MOUNTS for d in R.DENSITIES]
    sigma = tracker_sigma()
    print(f"UE error sigma per axis (TVT tracker, tuning seeds, robust): {sigma:.3f} m", flush=True)
    built = R.build(jobs, raw, cfg, 4)
    from review_b5_ablation import truth_tracks

    X, Y, V = [], [], []
    for job in jobs:
        tr = truth_tracks(raw, job, 600)
        rk = T4.real_tracks(job, raw)
        rng = np.random.default_rng([job[0], ["lamppost", "facade"].index(job[1]), ["low", "high"].index(job[2]), 3])
        ue = tr["ue"].copy()
        ue[..., :2] += sigma * rng.standard_normal(ue[..., :2].shape)
        los = built[job]["data"]["los_loss_db"]
        for r in range(20, 600 - int(TAUS[-1] / 0.1) - 1):
            X.append(features(tr["oru"], ue[r], tr["ue_vel"][r], rk, r).reshape(-1, N_FEAT))
            Y.append(labels(los, r, ue.shape[1], tr["oru"].shape[0]).reshape(-1, len(TAUS)))
            V.append(np.full(X[-1].shape[0], job[0] in val_seeds))
        print(f"features {job} ({time.perf_counter() - clock:.0f} s)", flush=True)
    X, Y, V = np.concatenate(X), np.concatenate(Y), np.concatenate(V)
    m = MLP(N_FEAT, len(TAUS))
    hist = m.fit(X[~V], Y[~V], X[V], Y[V])
    p = m.predict_proba(X[V])
    from tvt_t3_visibility import auc

    aucs = [auc(p[:, k], Y[V][:, k] > 0.5) for k in range(len(TAUS))]
    torch.save(m, OUT / "learned.pt")
    out = {"definition": __doc__, "train_seeds": [s for s in seeds if s not in val_seeds], "val_seeds": val_seeds, "ue_sigma_m": sigma,
           "n_train": int((~V).sum()), "n_val": int(V.sum()), "positive_rate": Y.mean(0).tolist(), "val_bce": hist["val_bce"], "val_auc_per_tau": aucs,
           "taus_s": TAUS.tolist(), "wall_s": time.perf_counter() - clock}
    (OUT / "learned.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "definition"}, indent=1))


if __name__ == "__main__":
    main()
