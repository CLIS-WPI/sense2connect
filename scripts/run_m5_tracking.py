"""M2 tracking numbers for the paper, recomputed from the stage-validated detection caches.

Evaluation seeds, budget 4 FA/CPI, blind clutter, detector parameters from
the picks in results/M2/metrics.json, replay helpers of
scripts/run_m2_followup.py, map tracker parameters = MAP_TUNED of
scripts/run_m2_final.py (the follow-up's tuning choice; checked against
results/M5/leadtime.json when present).
- "noghost": budget-4 none/blind pick; unconstrained EKF and map tracker
  -> MapBus / MapPed (bus/truck and pedestrian track Pd) and MapCross
  (bus/truck cross-lane velocity RMSE [m/s]).
- "image": budget-4 image/blind pick (image-method ghost handling) =
  final M2 configuration with the map tracker -> BusPd, PedPd, FA per CPI,
  lead 0.5 s per class and mount; with the unconstrained EKF -> GhostGain
  (bus/truck track Pd, none -> image, both unconstrained, as in M2).
Pooled over mounts unless a per-mount value is stored.

Writes results/M5/tracking.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LEADS = (0.1, 0.3, 0.5, 1.0)


def _pick(budget: float, method: str) -> dict[str, Any]:
    metrics = json.loads((ROOT / "results" / "M2" / "metrics.json").read_text(encoding="utf-8"))
    for p in metrics["picks"]:
        if float(p["budget"]) == budget and p["method"] == method and p["clutter"] == "blind" and p.get("point"):
            return dict(p["point"]["params"])
    raise SystemExit(f"no budget-{budget} {method}/blind pick in results/M2/metrics.json")


def _job(item: tuple) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    import run_m2_final as FIN
    import run_m2_followup as F
    from fig_leadtime import load_case
    from sim.scenes.config import load_yaml
    from sim.sensing.metrics import empty_score, scaled_gates

    mount, density, seed = item
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    spec = raw["sensing_radar"]
    gates = scaled_gates(1.0)
    case = load_case(mount, density, seed)
    out: dict[str, Any] = {"mount": mount, "stage_check": case["stage_check"]}
    for variant, method in (("noghost", "none"), ("image", "image")):
        params = _pick(4.0, method)
        params["clutter"] = "blind"
        params["ghost"] = method == "image"
        for name in ("unconstrained", "map"):
            tracker = (F._unconstrained(params, spec, case["radar_position_m"]) if name == "unconstrained"
                       else F._map_tracker(params, spec, case["radar_position_m"], list(raw["lanes"]), list(raw["sidewalks"]), FIN.MAP_TUNED))
            score, tracks = F._replay_tracker(case, tracker, params, spec, raw["blocker_kinds"], gates)
            total = empty_score()
            F._add(total, score)
            out[f"{variant}|{name}"] = {
                "score": total,
                "velocity": F._velocity(case, tracks, case["radar_position_m"], gates),
                "leads": F._lead_counts(case, tracks, gates),
            }
    return out


def main() -> None:
    import run_m2_final as FIN
    import run_m2_followup as F
    from sim.scenes.config import load_yaml
    from sim.sensing.metrics import CLASSES, empty_score, summarize

    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    jobs = [(m, d, int(s)) for s in seeds["evaluation"] for m in ("lamppost", "facade") for d in ("low", "high")]
    lt = ROOT / "results" / "M5" / "leadtime.json"
    if lt.exists():
        tuned = json.loads(lt.read_text())["tuned_map"]
        if any(float(tuned[k]) != float(FIN.MAP_TUNED[k]) for k in FIN.MAP_TUNED):
            raise SystemExit(f"recomputed map tuning {tuned} differs from MAP_TUNED {FIN.MAP_TUNED}")
    with mp.get_context("spawn").Pool(4) as pool:
        parts = pool.map(_job, jobs, chunksize=1)
    res: dict[str, Any] = {"map_tracker": FIN.MAP_TUNED, "stage_checks": sorted({p["stage_check"] for p in parts}), "variants": {}}
    for key in ("noghost|unconstrained", "noghost|map", "image|unconstrained", "image|map"):
        total = empty_score()
        vel = {c: {"n": 0, "radial": 0.0, "cross_range": 0.0, "along": 0.0, "cross_lane": 0.0} for c in CLASSES}
        leads: dict[str, Any] = {}
        for part in parts:
            F._add(total, part[key]["score"])
            F._accumulate(vel, part[key]["velocity"])
            for cls, row in part[key]["leads"].items():
                e = leads.setdefault(part["mount"], {}).setdefault(cls, {"n": 0, "hits": {str(L): 0 for L in LEADS}})
                e["n"] += row["n"]
                for L in LEADS:
                    e["hits"][str(L)] += row["hits"][str(L)]
        summ = summarize(total)
        res["variants"][key] = {
            "track_pd": {c: summ["classes"][c]["track_pd"] for c in CLASSES},
            "false_alarms_per_cpi": summ["false_alarms_per_cpi"],
            "cross_lane_rmse_mps": {c: (float(np.sqrt(vel[c]["cross_lane"] / vel[c]["n"])) if vel[c]["n"] else None) for c in CLASSES},
            "leads": leads,
            "summary": summ,  # full sim.sensing.metrics.summarize output (per-class Pd, position/velocity RMSE; false tracks per CPI)
        }
        print(key, {c: round(v or 0, 3) for c, v in res["variants"][key]["track_pd"].items()}, "FA", round(summ["false_alarms_per_cpi"], 2),
              "xlane", {c: v and round(v, 2) for c, v in res["variants"][key]["cross_lane_rmse_mps"].items()}, flush=True)
    dest = ROOT / "results" / "M5" / "tracking.json"
    dest.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
