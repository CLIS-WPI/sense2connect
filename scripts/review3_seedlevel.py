"""ADDED AFTER THE THIRD EXTERNAL REVIEW: seed-level statistics (re-analysis only, no new simulation).

The statistical unit is the SEED. The four runs of a seed (2 O-RU mounts x
2 traffic densities) are not independent: they share the seed's traffic
random stream (identical vehicles and pedestrians for both mounts at one
density; the low- and high-density fleets start from the same stream), the
radar noise seed (seed * 100003 + frame), the PathSolver seed and the UE
position-fix draws (seed * 17 + 3); the UE trajectories are the same in
every run. Only the error-injection draws (per run and condition) and the
uncertainty-aware planner's posterior samples are drawn per run.

For every paired comparison scheme - reference (per-job outages
[s/UE-min] from the results of the frozen pipeline):
- seed level: per seed, the mean over its 4 runs of the per-run
  difference; mean over the n = 10 seeds; exact two-sided Wilcoxon
  signed-rank p on the 10 seed differences (scipy, zero differences
  dropped); 95 % CI = percentile interval of a cluster bootstrap over
  seeds (10,000 resamples of the 10 seeds with replacement, mean of the
  resampled seed differences; generator seed 20261004);
- run level (as before, kept for comparison): scripts/run_m5_paired.paired
  on the 40 runs.
Break-even (sweeps): the sigma at which the mean paired difference to A5
(= seed-level mean) first reaches 0 from below, linear interpolation
between evaluated sigmas (sigma = 0 is perfect tracks); significant-
advantage limit: the largest evaluated sigma at which the planner is
still significantly better than A5 at seed level (mean < 0 and Wilcoxon
p < 0.05).

Reads results/M5/{planner,grid_oracle,review_b6}.json,
results/M5/review2/{sweeps,uaplanner}.json, results/M5/review3/eval.json.
Writes results/M5/review3/seedlevel.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_m5_paired import paired  # noqa: E402

R5 = ROOT / "results" / "M5"
N_BOOT = 10_000
BOOT_SEED = 20261004


def seed_paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    """Seed-level paired comparison of per-job outages a - b (job keys 'seed/mount/density')."""
    if sorted(a) != sorted(b):
        raise SystemExit("seed-level comparison needs the same jobs")
    per: dict[str, list[float]] = {}
    for k in a:
        per.setdefault(k.split("/")[0], []).append(a[k] - b[k])
    seeds = sorted(per)
    if len({len(v) for v in per.values()}) != 1:
        raise SystemExit("unequal number of runs per seed")
    d = np.array([np.mean(per[s]) for s in seeds])
    n = len(d)
    nz = int(np.count_nonzero(d))
    if nz:
        res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        p = float(res.pvalue)
    else:
        p = float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_seeds": n, "runs_per_seed": len(per[seeds[0]]), "mean_diff": float(d.mean()), "ci95_boot": [float(lo), float(hi)],
            "wilcoxon_p_two_sided": p, "n_seeds_lower": int((d < 0).sum()), "n_seeds_higher": int((d > 0).sum()),
            "per_seed": {s: float(v) for s, v in zip(seeds, d)}}


def both(a: dict, b: dict) -> dict:
    return {"seed": seed_paired(a, b), "run": paired(a, b)}


def crossover(xs: list[float], ys: list[float]) -> float | None:
    if ys[0] >= 0:
        return None
    for i in range(1, len(xs)):
        if ys[i] >= 0:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (-ys[i - 1]) / (ys[i] - ys[i - 1])
    return math.inf


def main() -> None:
    pl = {m["label"]: m for m in json.loads((R5 / "planner.json").read_text())["margins"]}
    go = {m["label"]: m for m in json.loads((R5 / "grid_oracle.json").read_text())["margins"]}
    b6 = {m["label"]: m for m in json.loads((R5 / "review_b6.json").read_text())["margins"]}
    sw = json.loads((R5 / "review2" / "sweeps.json").read_text())
    ua = {m["label"]: m for m in json.loads((R5 / "review2" / "uaplanner.json").read_text())["margins"]}
    ev = json.loads((R5 / "review3" / "eval.json").read_text())
    oh = {m["label"]: m for m in ev["a5_overhead"]}
    labels = list(pl)
    comp: dict[str, dict[str, Any]] = {}

    def add(name: str, lab: str, a: dict, b: dict) -> None:
        comp.setdefault(name, {})[lab] = both(a, b)

    for lab in labels:
        p = pl[lab]
        a5 = p["a5"]["per_job"]
        h = f"{p['sensing_planner_tuned_H']:.1f}"
        add("sensing-planner - A5", lab, p["sensing_planner"][h]["per_job"], a5)
        add("true-LoS-loss planner - A5", lab, p["diag_trueloss_planner"][h]["per_job"], a5)
        add("genie-planner (H 0.5 s) - A5", lab, p["genie_planner"]["0.5"]["per_job"], a5)
        add("A3 (wide) - A5", lab, p["a3_wide"]["per_job"], a5)
        add("cost-aware oracle (any step) - A5", lab, go[lab]["any"]["per_job"], a5)
        add("instantaneous oracle - A5", lab, go[lab]["inst"]["per_job"], a5)
        add("guarded planner (B6) - A5", lab, b6[lab]["per_job"], a5)
        add("uncertainty-aware planner - A5", lab, ua[lab]["per_job"], a5)
        add("A5 + sensing overhead - A5", lab, oh[lab]["per_job"], a5)
        add("uncertainty-aware planner - (A5 + sensing overhead)", lab, ua[lab]["per_job"], oh[lab]["per_job"])
        for cname, c in list(sw["conditions"].items()) + list(ev["conditions"].items()):
            m = {x["label"]: x for x in c["margins"]}[lab]
            add(f"sweep: {cname} - A5", lab, m["per_job"], a5)
    # break-even series on the merged grids
    def series(prefix_fn, sigmas):
        xs = [0.0] + sorted(set(sigmas))
        names = ["perfect | ue 0.0"] + [prefix_fn(s) for s in xs[1:]]
        return xs, names

    def fmt_sigma(s: float) -> str:
        return f"{s:g}"

    ue_sig = [float(s) for s in sw["sigmas"]] + [float(s) for s in ev["ue_sigmas"]]
    pos_sig = [float(s) for s in sw["sigmas"]] + [float(s) for s in ev["pos_sigmas"]]
    vel_sig = [float(s) for s in sw["sigmas"]]
    be: dict[str, Any] = {}
    for key, fn, sig in (("UE position (perfect tracks)", lambda s: f"perfect | ue {fmt_sigma(s) if fmt_sigma(s) != '1' else '1.0'}", ue_sig),
                         ("blocker position (realistic), UE exact", lambda s: f"R pos {fmt_sigma(s) if fmt_sigma(s) != '1' else '1.0'} | ue 0.0", pos_sig),
                         ("blocker velocity (realistic), UE exact", lambda s: f"R vel {fmt_sigma(s) if fmt_sigma(s) != '1' else '1.0'} | ue 0.0", vel_sig)):
        xs, names = series(fn, sig)
        be[key] = {"sigmas": xs, "conditions": names, "margins": {}}
        for lab in labels:
            pts = [comp[f"sweep: {n} - A5"][lab]["seed"] for n in names]
            ys = [q["mean_diff"] for q in pts]
            cx = crossover(xs, ys)
            sig_ok = [x for x, q in zip(xs, pts) if q["mean_diff"] < 0 and q["wilcoxon_p_two_sided"] < 0.05]
            be[key]["margins"][lab] = {"mean_diff": ys, "p": [q["wilcoxon_p_two_sided"] for q in pts], "ci95_boot": [q["ci95_boot"] for q in pts],
                                       "mean_break_even": (None if cx is None else ("inf" if math.isinf(cx) else cx)),
                                       "significant_advantage_limit": (max(sig_ok) if sig_ok else None)}
    out = {"definition": __doc__, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "comparisons": comp, "break_even": be}
    dest = R5 / "review3" / "seedlevel.json"
    dest.write_text(json.dumps(out) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
