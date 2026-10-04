"""ADDED AFTER EXTERNAL REVIEW (C8): main results split by traffic density.

From the per-job outages already computed (no re-simulation): A3 (wide) and
A5, genie-planner (H = 0.5 s), sensing-planner (tuned H), true-LoS-loss
planner (sensing-planner's tuned H) from results/M5/planner.json; robust
planner (B6) from results/M5/review_b6.json if present; any-step,
grid-aligned and instantaneous cost-aware oracles from
results/M5/grid_oracle.json. Per margin and density (low / high, 20
evaluation jobs each): mean and 95 % CI over jobs (1.96 sd / sqrt(n)), and
the paired A5 -> any-step oracle reduction (mean per-job difference, t-based
95 % CI) with its share of A5's outage.

Writes results/M5/review_c8.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results" / "M5"


def ci(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "ci": float(1.96 * a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0, "n": int(a.size)}


def main() -> None:
    pl = json.loads((R5 / "planner.json").read_text())
    go = {r["label"]: r for r in json.loads((R5 / "grid_oracle.json").read_text())["margins"]}
    b6f = R5 / "review_b6.json"
    b6 = {r["label"]: r for r in json.loads(b6f.read_text())["margins"]} if b6f.exists() else {}
    out = {"definition": __doc__, "margins": []}
    for r in pl["margins"]:
        lab = r["label"]
        h = f"{r['sensing_planner_tuned_H']:.1f}"
        series = {
            "A3 (wide)": r["a3_wide"]["per_job"],
            "A5": r["a5"]["per_job"],
            "genie-planner H=0.5": r["genie_planner"]["0.5"]["per_job"],
            f"sensing-planner H={h}": r["sensing_planner"][h]["per_job"],
            f"true-LoS-loss planner H={h}": r["diag_trueloss_planner"][h]["per_job"],
            "cost-aware oracle (any step)": go[lab]["any"]["per_job"],
            "cost-aware oracle (grid-aligned)": go[lab]["grid"]["per_job"],
            "instantaneous oracle": go[lab]["inst"]["per_job"],
        }
        if lab in b6:
            series["robust planner (B6)"] = b6[lab]["per_job"]
        res = {"label": lab, "density": {}}
        for dens in ("low", "high"):
            jobs = sorted(j for j in r["a5"]["per_job"] if j.endswith("/" + dens))
            block = {name: ci([pj[j] for j in jobs]) for name, pj in series.items()}
            dlt = np.array([series["A5"][j] - series["cost-aware oracle (any step)"][j] for j in jobs])
            half = float(stats.t.ppf(0.975, dlt.size - 1) * dlt.std(ddof=1) / math.sqrt(dlt.size))
            a5m = block["A5"]["mean"]
            block["A5 -> any-step oracle"] = {"mean_reduction": float(dlt.mean()), "ci95_half_t": half,
                                               "share_of_a5": (float(dlt.mean()) / a5m) if a5m > 0 else None}
            res["density"][dens] = block
        out["margins"].append(res)
        lo, hi = res["density"]["low"], res["density"]["high"]
        print(f"{lab}: A5 low {lo['A5']['mean']:.3f} high {hi['A5']['mean']:.3f} | oracle low {lo['cost-aware oracle (any step)']['mean']:.3f} "
              f"high {hi['cost-aware oracle (any step)']['mean']:.3f} | reduction share low {lo['A5 -> any-step oracle']['share_of_a5']} high {hi['A5 -> any-step oracle']['share_of_a5']}", flush=True)
    dest = R5 / "review_c8.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
