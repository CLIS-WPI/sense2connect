"""Second external review, Block 2 report (results/M5/review2/block2.md).

Reads results/M5/review2/{calibration,sweeps,uaplanner}.json and
results/M5/paired.json. Significance = two-sided Wilcoxon signed-rank p
(< 0.05); effect size = mean paired difference with its t-based 95 % CI.
A "disagreement" is flagged when the CI excludes 0 but p >= 0.05, or the
CI includes 0 but p < 0.05.

Crossover (break-even) of a sweep at one margin: the sigma at which the
mean paired difference to A5 first reaches 0 from below, by linear
interpolation between grid points; "none" if the planner is not below A5
at sigma = 0; "> max" if it stays below up to the largest sigma.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "results" / "M5" / "review2"
MARGINS = ("10 dB", "15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference")
SHORT = {"3GPP short-range reference": "Ref", "v1 radio (high margin)": "v1"}


def short(lab: str) -> str:
    return SHORT.get(lab, lab)


def flag(v: dict) -> str:
    ci_ex = v["ci95"][0] > 0 or v["ci95"][1] < 0
    sig = v["wilcoxon_p_two_sided"] < 0.05
    return " (disagree)" if ci_ex != sig else ""


def cell(v: dict, nd: int = 3) -> str:
    return f"{v['mean_diff']:+.{nd}f} [{v['ci95'][0]:+.{nd}f}, {v['ci95'][1]:+.{nd}f}], p={v['wilcoxon_p_two_sided']:.2g}{flag(v)}"


def by_label(res: dict) -> dict[str, dict]:
    return {m["label"]: m for m in res["margins"]}


def crossover(xs: list[float], ys: list[float]) -> str:
    if ys[0] >= 0:
        return "none (not below A5 at 0)"
    for i in range(1, len(xs)):
        if ys[i] >= 0:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return f"{x0 + (x1 - x0) * (-y0) / (y1 - y0):.2f}"
    return f"> {xs[-1]}"


def main() -> None:
    cal = json.loads((R2 / "calibration.json").read_text())
    sw = json.loads((R2 / "sweeps.json").read_text())
    cond, sig = sw["conditions"], list(sw["sigmas"])
    ua = json.loads((R2 / "uaplanner.json").read_text()) if (R2 / "uaplanner.json").exists() else None
    paired = {m["label"]: m for m in json.loads((ROOT / "results" / "M5" / "paired.json").read_text())["margins"]}
    L: list[str] = ["# Second external review, Block 2: new analyses (development seeds 1001-1010)", "",
                    "All runs on the development seeds (40 jobs); no tuning except the uncertainty-aware planner (tuning seeds 101-105).",
                    "Significance: two-sided Wilcoxon signed-rank p; effect size: mean paired difference to A5 [s/UE-min] with t-based 95 % CI. "
                    "'(disagree)' marks entries where the two disagree.", ""]
    # ---------------------------------------------------------------- calibration
    L += ["## Calibrated tracker-error statistics (real map tracker, budget 4, development seeds)", "",
          "| Class | track Pd | acquisition delay med/mean [CPI] | outage med/mean [CPI] | matched run med/mean [CPI] | free-mode share |", "|---|---|---|---|---|---|"]
    for c, o in cal["outages"].items():
        L.append(f"| {c} | {o['track_pd']:.3f} | {o['median']['acquisition']:.0f} / {o['mean']['acquisition']:.1f} | {o['median']['outage']:.0f} / {o['mean']['outage']:.1f} | "
                 f"{o['median']['matched_run']:.0f} / {o['mean']['matched_run']:.1f} | {cal['errors'][c]['free_mode_share']:.3f} |")
    L += ["", "State error of matched tracks on the predictor's (pinned) state; x = along every lane / sidewalk. One CPI = 0.1 s.", "",
          "| Class | comp. | mean | std | lag-1 phi | corr. time [s] | ACF lag 5 (measured / phi^5) |", "|---|---|---|---|---|---|---|"]
    for c, e in cal["errors"].items():
        for comp in ("x", "y", "vx", "vy"):
            ct = e["corr_time_s"][comp]
            a5, p5 = e["acf_lag5_vs_phi5"][comp]
            L.append(f"| {c} | {comp} | {e['mean'][comp]:+.2f} | {e['std'][comp]:.2f} | {e['phi'][comp]:.2f} | {ct:.2f} | {a5:.2f} / {p5:.2f} |" if ct else
                     f"| {c} | {comp} | {e['mean'][comp]:+.2f} | {e['std'][comp]:.2f} | {e['phi'][comp]:.2f} | - | {a5:.2f} / {p5:.2f} |")
    ft = cal["false_tracks"]
    L += ["", f"False tracks (unmatched confirmed tracks): {sum(ft['count_per_cpi'].values()):.2f} per CPI, fragment share {ft['fragment_share']:.3f}.", "",
          "| Mount | type | count / CPI | births / CPI | lifetime median / mean [CPI] | episodes |", "|---|---|---|---|---|---|"]
    for m, v in ft["per_mount"].items():
        for t, s in v["stats"].items():
            L.append(f"| {m} | {t} | {s['count_per_cpi']:.2f} | {s['birth_rate_per_cpi']:.3f} | {s['median_lifetime_cpi']:.0f} / {s['mean_lifetime_cpi']:.1f} | {s['episodes']} |")
    val = cond.get("R table all", {}).get("validation")
    if val:
        L += ["", "Generator check ('R table all' on the development jobs): tracked share of in-range reports "
              + ", ".join(f"{c} {val[c]['tracked_in_range'] / val[c]['in_range']:.3f} (Pd {cal['outages'][c]['track_pd']:.3f})" for c in cal["outages"])
              + f"; false tracks {val['false']['count'] / val['false']['reports']:.2f} per report."]
    # ---------------------------------------------------------------- sweeps
    L += ["", "## 2a Accuracy break-even sweeps (perfect tracks + one error; H fixed per margin)", "",
          "H per margin: " + ", ".join(f"{short(k)} {v}" for k, v in sw["H_fixed"].items()) + " s.", ""]
    series = [("UE position only (white, as the main runs)", lambda s: f"perfect | ue {s}", "perfect | ue 0.0")]
    for model, mname in (("R", "realistic"), ("M", "memoryless")):
        for kind, kname in (("pos", "blocker position"), ("vel", "blocker velocity"), ("both", "blocker position + velocity")):
            for u in (0.0, 1.0):
                series.append((f"{mname}: {kname}, UE error {u:g} m", (lambda s, model=model, kind=kind, u=u: f"{model} {kind} {s} | ue {u}"), f"perfect | ue {u}"))
    L += ["Break-even sigma (m or m/s) where the planner's advantage over A5 vanishes (mean difference reaches 0):", "",
          "| Sweep | " + " | ".join(short(m) for m in MARGINS) + " |", "|---|" + "---|" * len(MARGINS)]
    for name, fn, base in series:
        names = [base] + [fn(s) for s in sig]
        if not all(n in cond for n in names):
            continue
        row = []
        for lab in MARGINS:
            ys = [by_label(cond[n])[lab]["vs_a5"]["mean_diff"] for n in names]
            row.append(crossover([0.0] + sig, ys))
        L.append(f"| {name} | " + " | ".join(row) + " |")
    L += ["", "Paired difference to A5 at 20 dB and the reference (mean [95 % CI], Wilcoxon p):", ""]
    for name, fn, base in series:
        names = [base] + [fn(s) for s in sig]
        if not all(n in cond for n in names):
            continue
        L += [f"**{name}**", "", "| sigma | 20 dB | Ref |", "|---|---|---|"]
        for s, n in zip([0.0] + sig, names):
            bl = by_label(cond[n])
            L.append(f"| {s:g} | {cell(bl['20 dB']['vs_a5'])} | {cell(bl['3GPP short-range reference']['vs_a5'])} |")
        L.append("")
    # ---------------------------------------------------------------- table II
    L += ["## 2d Table II (error budget) with 'Perfect tracks' and 'All + UE error (1 m)'", ""]
    for model, title, cols in (
        ("R", "Realistic model (primary)", [("Perfect tracks", "perfect | ue 0.0"), ("Misses", "R table miss"), ("False tracks", "R table false"),
                                             ("Pos./vel. error", "R table noise"), ("Size rule", "R table size"), ("All", "R table all"),
                                             ("All + UE error (1 m)", "R table all + ue 1")]),
        ("M", "Memoryless model (limiting case; = first-review B5 at fixed H)", [("Perfect tracks", "perfect | ue 0.0"), ("Misses", "M table miss"),
                                                                                ("False tracks", "M table false"), ("Pos./vel. error", "M table noise"),
                                                                                ("All", "M table all"), ("All + UE error (1 m)", "M table all + ue 1")])):
        L += [f"**{title}**", "", "| Column | 10 dB | Ref |", "|---|---|---|"]
        for cname, key in cols:
            if key not in cond:
                continue
            bl = by_label(cond[key])
            L.append(f"| {cname} | {cell(bl['10 dB']['vs_a5'])} | {cell(bl['3GPP short-range reference']['vs_a5'])} |")
        L.append(f"| Real sensing-planner (own tuned H) | {cell(paired['10 dB']['sensing_vs_a5'])} | {cell(paired['3GPP short-range reference']['sensing_vs_a5'])} |")
        L.append("")
    L += ["Realistic Table II at every margin (mean difference, Wilcoxon p):", "",
          "| Column | " + " | ".join(short(m) for m in MARGINS) + " |", "|---|" + "---|" * len(MARGINS)]
    for cname, key in (("Perfect", "perfect | ue 0.0"), ("Misses", "R table miss"), ("False", "R table false"), ("Pos./vel.", "R table noise"),
                       ("Size rule", "R table size"), ("All", "R table all"), ("All + UE 1 m", "R table all + ue 1")):
        if key in cond:
            bl = by_label(cond[key])
            L.append(f"| {cname} | " + " | ".join(f"{bl[m]['vs_a5']['mean_diff']:+.3f} ({bl[m]['vs_a5']['wilcoxon_p_two_sided']:.2g})" for m in MARGINS) + " |")
    L.append("| Real sensing-planner | " + " | ".join(f"{paired[m]['sensing_vs_a5']['mean_diff']:+.3f} ({paired[m]['sensing_vs_a5']['wilcoxon_p_two_sided']:.2g})" for m in MARGINS) + " |")
    # ---------------------------------------------------------------- UA planner
    if ua:
        L += ["", "## 2b Uncertainty-aware planner (A5 default; tuned K, alpha, theta on tuning seeds)", "",
              f"Zero-noise check vs the cached sensing-planner predictions: {ua['zero_noise_check_db']:.1e} dB.", "",
              "| Margin | K | alpha | theta [s] | H [s] | budget | outage | UA - A5 [95 % CI], p | HO/UE-min |", "|---|---|---|---|---|---|---|---|---|"]
        for m in ua["margins"]:
            t = m["tuned"]
            L.append(f"| {short(m['label'])} | {t['K']} | {t['alpha']:g} | {t['theta_s']:g} | {t['H_s']:g} | {t['budget']} | {m['eval']['outage_req_s_per_min']['mean']:.3f} | "
                     f"{cell(m['vs_a5'], 4)} | {m['eval']['ho_per_min']['mean']:.2f} |")
    (R2 / "block2.md").write_text("\n".join(L) + "\n")
    print(f"wrote {R2 / 'block2.md'}")


if __name__ == "__main__":
    main()
