"""Causal foresight bound: A3 outage decomposition with an unblocked counterfactual.

Tuned A3 (main run, tau_HO = 20 ms) on the evaluation seeds, every margin.
Every 10 ms step in outage at the service rate is assigned to one class
(precedence top to bottom):
- both cells unusable (blocked SNR of both cells below the service rate);
- handover interruption;
- wrong cell, blockage-caused: A3's serving cell is unusable, the other
  cell is usable, and A3's serving cell WOULD be usable with its unblocked
  power (no blockers);
- wrong cell, distance-caused: A3's serving cell is unusable even
  unblocked, the other cell is usable.
New foresight bound = blockage-caused wrong-cell time (any loss level,
whichever cell A3 is on). Blockage-caused steps are split by the class of
the dominant LoS blocker of A3's cell at that step ("no LoS blocker" when
the loss is on other paths only).

The old definition (wrong-cell time inside 10 dB events of the fixed,
strongest-unblocked cell, by event class) is recomputed on the same lanes
for comparison; both splits of the wrong-cell time must sum to the same
total. The A3-oracle gap (A3 outage minus instantaneous-oracle outage =
wrong cell + interruption, pooled) is split into foresight-recoverable
(blockage-caused), distance-caused and interruption shares.

Writes results/M5/foresight.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_m3 as R  # noqa: E402
from sim.comm.linkbudget import snr_ref_db  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from sim.scenes.traffic import prepare_scenario  # noqa: E402
from xapp.schemes import simulate  # noqa: E402

CLASS_OF = {"bus": "bus/truck", "truck": "bus/truck", "pedestrian": "pedestrian", "car": "car"}
NEW_CLASSES = ("bus/truck", "pedestrian", "car", "no LoS blocker")
OLD_CLASSES = ("bus/truck", "pedestrian", "car")


def usable(snr_db: np.ndarray, bw: float, rate_req: float) -> np.ndarray:
    return bw * np.minimum(np.log2(1.0 + 10.0 ** (snr_db / 10.0)), MAX_NR_SE) >= rate_req


def blocker_kinds(raw: dict, job: tuple) -> list[str]:
    seed, mount, density = job
    sc = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    return [str(s["kind"]) for s in list(sc["vehicles"]) + list(sc["pedestrians"])]


def decompose_margin(mi, jobs, built, snr_all, info, rw, raw, bw, rate_req, a3p, tau_ho, kinds) -> dict[str, Any]:
    b = rw["budget"]
    snr_m = {j: snr_all[j][mi] for j in jobs}
    lanes, index = R.make_lanes(jobs, [a3p], snr_m, built, "a3", {}, rw, None, tau_ho_s=tau_ho)
    sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
    loss = info["extra_loss_db"][mi]
    n_t = lanes.snr_db.shape[1]
    minutes = n_t * R.DT_COMM / 60.0
    use_b = usable(lanes.snr_db, bw, rate_req)  # [L, T, 2]
    per_job: dict[tuple, dict[str, float]] = {}
    for lane, (_, job, u) in enumerate(index):
        d = built[job]["data"]
        snr_unb = snr_ref_db(d["unblocked_power"][:, u, :], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - loss
        use_u = usable(snr_unb, bw, rate_req)  # [T, 2]
        srv = sim["serving"][lane].astype(np.int64)
        k = np.arange(n_t)
        out = sim["outage_req"][lane]
        both = ~use_b[lane].any(-1)
        intr = sim["interrupted"][lane]
        c = out & both
        bb = out & ~both & intr
        wrong = out & ~both & ~intr
        srv_unblocked_ok = use_u[k, srv]
        blk = wrong & srv_unblocked_ok
        dist = wrong & ~srv_unblocked_ok
        # dominant LoS blocker of A3's cell
        dom = d["los_blocker"][k, u, srv]
        cls = np.array(["no LoS blocker" if x < 0 else CLASS_OF.get(kinds[job][x], "other") for x in dom])
        # old definition: wrong cell inside 10 dB events of the fixed cell, by event class
        old_in = {c_: np.zeros(n_t, dtype=bool) for c_ in ("all",) + OLD_CLASSES}
        for ev in built[job]["events"]:
            if ev["ue"] != u:
                continue
            old_in["all"][ev["start_k"] : ev["end_k"] + 1] = True
            if ev["class"] in old_in:
                old_in[ev["class"]][ev["start_k"] : ev["end_k"] + 1] = True
        acc = per_job.setdefault(job, {})
        add = lambda key, mask: acc.__setitem__(key, acc.get(key, 0.0) + float(mask.sum()))  # noqa: E731
        add("outage", out)
        add("c_both_unusable", c)
        add("b_interruption", bb)
        add("wrong_total", wrong)
        add("new_blockage_caused", blk)
        add("new_distance_caused", dist)
        for c_ in NEW_CLASSES:
            add(f"new_blockage_caused|{c_}", blk & (cls == c_))
        add("old_in_events", wrong & old_in["all"])
        add("old_outside_events", wrong & ~old_in["all"])
        for c_ in OLD_CLASSES:
            add(f"old_in_events|{c_}", wrong & old_in[c_])
        acc["n_ue"] = acc.get("n_ue", 0) + 1
    keys = [k_ for k_ in next(iter(per_job.values())) if k_ != "n_ue"]
    res: dict[str, Any] = {}
    for key in keys:
        vals = [v[key] * R.DT_COMM / (minutes * v["n_ue"]) for v in per_job.values()]
        shares = [v[key] / v["outage"] for v in per_job.values() if v["outage"] > 0]
        tot_out = sum(v["outage"] for v in per_job.values())
        res[key] = {
            "s_per_ue_min": R.ci95(vals),
            "share_of_a3_outage_per_job": R.ci95(shares),
            "share_of_a3_outage_pooled": (sum(v[key] for v in per_job.values()) / tot_out) if tot_out else None,
        }
    tot = {k_: sum(v[k_] for v in per_job.values()) for k_ in ("wrong_total", "new_blockage_caused", "new_distance_caused", "old_in_events", "old_outside_events", "outage", "c_both_unusable", "b_interruption")}
    gap = tot["outage"] - tot["c_both_unusable"]  # A3 - instantaneous oracle (oracle outage = both cells unusable), pooled steps
    res["a3_oracle_gap_pooled"] = {
        "gap_s_per_ue_min_pooled": gap * R.DT_COMM / (minutes * sum(v["n_ue"] for v in per_job.values())),
        "foresight_recoverable_share": (tot["new_blockage_caused"] / gap) if gap > 0 else None,
        "distance_caused_share": (tot["new_distance_caused"] / gap) if gap > 0 else None,
        "interruption_share": (tot["b_interruption"] / gap) if gap > 0 else None,
    }
    res["checks"] = {
        "new_split_sums": tot["new_blockage_caused"] + tot["new_distance_caused"] == tot["wrong_total"],
        "old_split_sums": tot["old_in_events"] + tot["old_outside_events"] == tot["wrong_total"],
        "classes_sum_to_outage": tot["c_both_unusable"] + tot["b_interruption"] + tot["wrong_total"] == tot["outage"],
    }
    return res


def main() -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    main_metrics = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    if abs(info["margin_ref_db"] - main_metrics["budget"]["margin_ref_db"]) > 1e-9:
        raise SystemExit("reference margin differs from the main run")
    snr_all = R.all_snr(built, eval_jobs, rw, bw, info["extra_loss_db"])
    kinds = {job: blocker_kinds(raw, job) for job in eval_jobs}
    tau = float(rw["e2"]["tau_ho_s"])
    out = {"definition": __doc__, "tau_ho_s": tau, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        res = decompose_margin(mi, eval_jobs, built, snr_all, info, rw, raw, bw, rate_req, main_metrics["tuned"][mi]["a3"]["params"], tau, kinds)
        if not all(res["checks"].values()):
            raise SystemExit(f"{lab}: decomposition checks failed {res['checks']}")
        out["margins"].append({"label": lab, "margin_db": info["points_db"][mi], **res})
        print(
            f"{lab}: A3 {res['outage']['s_per_ue_min']['mean']:.3f} | old bound {res['old_in_events']['s_per_ue_min']['mean']:.3f} "
            f"({100 * (res['old_in_events']['share_of_a3_outage_pooled'] or 0):.1f} %) | new bound {res['new_blockage_caused']['s_per_ue_min']['mean']:.3f} "
            f"({100 * (res['new_blockage_caused']['share_of_a3_outage_pooled'] or 0):.1f} %) | distance {res['new_distance_caused']['s_per_ue_min']['mean']:.3f}",
            flush=True,
        )
    dest = ROOT / "results" / "M5" / "foresight.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=R._json) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
