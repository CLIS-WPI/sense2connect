"""ADDED AFTER EXTERNAL REVIEW (B6 diagnostic): where does the robust planner lose against A5?

Diagnostic only. With the robust planner's tuned parameters per margin
(results/M5/review_b6.json: T_c, H, detector budget), three variants are
evaluated on the evaluation seeds and compared with A5 by the paired test:
- A5 + overhead: A5 alone, but charged the sensing overhead (no advance, no veto);
- advance only: the robust planner without vetoes;
- veto only: the robust planner without advances.
Same track replay, predictions and tables as scripts/review_b6_robust.py.

Writes results/M5/review_b6_diag.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_m3 as R  # noqa: E402
from review_b6_robust import _replay, run_robust, tables  # noqa: E402
from run_m5_paired import paired  # noqa: E402
from run_m5_planner import per_job_outage  # noqa: E402
from sim.comm.linkbudget import sensing_overhead, snr_ref_db  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402


def main() -> None:
    from sim.scenes.motion import states_at
    from sim.scenes.traffic import prepare_scenario
    from xapp.predict_torch import pack_tracks, predict_torch, ue_fixes

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    b6 = json.loads((ROOT / "results" / "M5" / "review_b6.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    ovh = sensing_overhead(rw["sensing"]["symbols_fraction"], rw["sensing"]["duty_cycle"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, eval_jobs, rw, bw, info["extra_loss_db"])
    b, pr = rw["budget"], rw["predict"]
    tau = int(round(float(rw["e2"]["tau_ho_s"]) / R.DT_COMM))
    d = int(round(float(rw["e2"]["loop_delay_s"]) / R.DT_COMM))
    rs = int(round(R.DT_SENSE / R.DT_COMM))
    wl = 299792458.0 / float(raw["carrier_hz"])
    with mp.get_context("spawn").Pool(4) as pool:
        tracks = dict(pool.map(_replay, [(job,) for job in eval_jobs], chunksize=1))
    taus = built[eval_jobs[0]]["data"]["taus"]
    geo = {}
    for job in eval_jobs:
        sc = prepare_scenario(raw, seed=job[0], mount=job[1], density=job[2], duration_s=60.0, dt_s=0.1)
        n_r = int(built[job]["data"]["pred_2"].shape[0])
        t_r = (np.arange(n_r) * rs) * R.DT_COMM
        ue = np.array([[states_at(sc, float(t))[x["name"]]["position_m"] for x in sc["ues"]] for t in t_r])
        uev = np.array([[states_at(sc, float(t))[x["name"]]["velocity_mps"] for x in sc["ues"]] for t in t_r])
        sizes = {k: (float(v["length_m"]), float(v["width_m"]), float(v["height_m"])) for k, v in sc["blocker_kinds"].items()}
        geo[job] = {"fix": ue_fixes(ue, uev, float(pr["ue_sigma_m"]), job[0] * 17 + 3), "vel": uev,
                    "oru": np.array([o["position_m"] for o in sc["orus"]], dtype=np.float64), "sizes": sizes}
    preds_cache: dict[tuple, dict] = {}
    out = {"definition": __doc__, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        tuned = b6["margins"][mi]["tuned"]
        key = (tuned["budget"], tuned["T_c_s"])
        if key not in preds_cache:
            preds_cache[key] = {}
            for job in eval_jobs:
                rows = [[t for t in fr if t["age_s"] >= tuned["T_c_s"] - 1e-9] for fr in tracks[job][tuned["budget"]]]
                pk = pack_tracks(rows, {"bus": geo[job]["sizes"]["bus"], "pedestrian": geo[job]["sizes"]["pedestrian"]})
                preds_cache[key][job] = predict_torch(pk, geo[job]["fix"], geo[job]["vel"], geo[job]["oru"], taus, wl)
        loss = info["extra_loss_db"][mi]
        unb = {job: snr_ref_db(built[job]["data"]["unblocked_power"], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - loss for job in eval_jobs}
        snr_e = {j: snr_all[j][mi] for j in eval_jobs}
        or_e = R.stateless_rows(eval_jobs, snr_e, built, "oracle", bw, rate_req)[1]
        a5p = pl["margins"][mi]["a5"]["params"]
        base = lambda ix: tables(ix, preds_cache[key], unb, taus, tuned["H_s"], d, rs, tau, bw, rate_req, ovh)  # noqa: E731
        variants = {
            "A5 + overhead": lambda ix: tuple(np.zeros_like(t) for t in base(ix)),
            "advance only": lambda ix: (lambda a, v: (a, np.zeros_like(v)))(*base(ix)),
            "veto only": lambda ix: (lambda a, v: (np.zeros_like(a), v))(*base(ix)),
            "advance + veto (check)": base,
        }
        res = {"label": lab, "tuned": tuned}
        for name, fn in variants.items():
            rows = run_robust(eval_jobs, a5p, snr_e, built, rw, bw, rate_req, or_e, info["snr_req_db"], fn, ovh)
            pj = per_job_outage(rows)
            res[name] = {"outage_mean": float(np.mean(list(pj.values()))), "vs_a5": paired(pj, pl["margins"][mi]["a5"]["per_job"])}
        out["margins"].append(res)
        print(f"{lab}: " + " | ".join(f"{n} {res[n]['outage_mean']:.3f} ({res[n]['vs_a5']['mean_diff']:+.3f})" for n in variants), flush=True)
    dest = ROOT / "results" / "M5" / "review_b6_diag.json"
    dest.write_text(json.dumps(out, indent=1, default=R._json) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
