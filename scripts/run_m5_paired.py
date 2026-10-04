"""Paired statistics on the evaluation jobs (statistics only, no method change).

For every margin, two comparisons against A5 (the best reactive scheme):
- true-LoS-loss planner (diagnostic) at the sensing-planner's tuned H;
- sensing-planner at its tuned H (and budget).
Per evaluation job j, d_j = outage(scheme) - outage(A5) [s/UE-min], each job's
outage being the mean over its two UEs (results/M5/planner.json, per_job).
Reported: mean of d_j with a t-based 95 % CI (n = 40 jobs, n-1 dof), and the
two-sided Wilcoxon signed-rank p-value (scipy.stats.wilcoxon, zero
differences dropped; the number of non-zero pairs is recorded).
Negative d = lower outage than A5.

Writes results/M5/paired.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def paired(a: dict[str, float], b: dict[str, float]) -> dict:
    jobs = sorted(a)
    if sorted(b) != jobs:
        raise SystemExit("paired comparison needs the same jobs")
    d = np.array([a[j] - b[j] for j in jobs])
    n = len(d)
    mean = float(d.mean())
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    nz = int(np.count_nonzero(d))
    p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if nz > 0 else float("nan")
    return {"n_jobs": n, "n_nonzero": nz, "mean_diff": mean, "ci95_half_t": half, "ci95": [mean - half, mean + half],
            "wilcoxon_p_two_sided": p, "n_scheme_lower": int((d < 0).sum()), "n_scheme_higher": int((d > 0).sum())}


def main() -> None:
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    out = {"definition": __doc__, "margins": []}
    for r in pl["margins"]:
        h = f"{r['sensing_planner_tuned_H']:.1f}"
        a5 = r["a5"]["per_job"]
        res = {
            "label": r["label"],
            "margin_db": r["margin_db"],
            "H_s": float(h),
            "trueloss_vs_a5": paired(r["diag_trueloss_planner"][h]["per_job"], a5),
            "sensing_vs_a5": paired(r["sensing_planner"][h]["per_job"], a5),
        }
        out["margins"].append(res)
        t, s = res["trueloss_vs_a5"], res["sensing_vs_a5"]
        print(f"{r['label']}: true-loss - A5 {t['mean_diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['wilcoxon_p_two_sided']:.3g} | "
              f"sensing - A5 {s['mean_diff']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] p={s['wilcoxon_p_two_sided']:.3g}", flush=True)
    dest = ROOT / "results" / "M5" / "paired.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
