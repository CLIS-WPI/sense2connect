"""Third external review report (results/M5/review3/report.md) from results/M5/review3/seedlevel.json.

Significance at seed level: exact two-sided Wilcoxon on the 10 seed
differences (p < 0.05); at run level (as before): Wilcoxon on 40 runs.
A claim "flips" when the sign-and-significance class (<*, <, >, >*) differs.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results" / "M5"
SHORT = {"3GPP short-range reference": "Ref", "v1 radio (high margin)": "v1"}


def cls(v: dict) -> str:
    return ("<" if v["mean_diff"] < 0 else ">") + ("*" if v["wilcoxon_p_two_sided"] < 0.05 else "")


def main() -> None:
    sl = json.loads((R5 / "review3" / "seedlevel.json").read_text())
    comp, be = sl["comparisons"], sl["break_even"]
    labels = list(next(iter(comp.values())))
    L = ["# Third external review: seed-level statistics, dense break-even, A5 + overhead (held-out seeds 2001-2010)", "",
         "Frozen code (v1.1-freeze); re-analysis of held-out results plus frozen-code evaluations (scripts/review3_eval.py).", "",
         "## 1. Statistical unit", "", sl["definition"].split("For every paired")[0].strip(), "",
         "Seed level: mean over 10 seeds of the per-seed mean difference; exact two-sided Wilcoxon on n = 10 (smallest attainable p = 2/1024 = 0.002); "
         "95 % CI from a cluster bootstrap over seeds (10,000 resamples). Classes: '<' / '>' = below / above the reference, '*' = p < 0.05.", "",
         "### Main comparisons (seed level: mean [bootstrap 95 % CI], p, seeds lower/higher; run level class)", ""]
    main_cmp = ["true-LoS-loss planner - A5", "sensing-planner - A5", "uncertainty-aware planner - A5", "A5 + sensing overhead - A5",
                "uncertainty-aware planner - (A5 + sensing overhead)", "guarded planner (B6) - A5", "genie-planner (H 0.5 s) - A5",
                "cost-aware oracle (any step) - A5", "A3 (wide) - A5"]
    for name in main_cmp:
        L += [f"**{name}**", "", "| Margin | seed level | run level |", "|---|---|---|"]
        for lab in labels:
            s, r = comp[name][lab]["seed"], comp[name][lab]["run"]
            flip = " **FLIP**" if cls(s) != cls(r) else ""
            L.append(f"| {SHORT.get(lab, lab)} | {s['mean_diff']:+.4f} [{s['ci95_boot'][0]:+.4f}, {s['ci95_boot'][1]:+.4f}], p={s['wilcoxon_p_two_sided']:.3g}, "
                     f"{s['n_seeds_lower']}/{s['n_seeds_higher']} ({cls(s)}) | p={r['wilcoxon_p_two_sided']:.2g} ({cls(r)}){flip} |")
        L.append("")
    tl = comp["true-LoS-loss planner - A5"]
    sel = ["15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference"]
    pmax = max(tl[lab]["seed"]["wilcoxon_p_two_sided"] for lab in sel)
    L += [f"True-LoS-loss planner from 15 dB (15-30 dB + Ref, 5 margins): max seed-level p = {pmax:.4g}; Bonferroni x5 = {5 * pmax:.4g} "
          + ("(< 0.05: holds)" if 5 * pmax < 0.05 else "(>= 0.05: does NOT hold)"), ""]
    L += ["### Every flip (run level -> seed level), all comparisons and sweep conditions", ""]
    n_flip = 0
    for name, per in comp.items():
        fl = [f"{SHORT.get(lab, lab)} {cls(per[lab]['run'])}->{cls(per[lab]['seed'])}" for lab in labels if cls(per[lab]["seed"]) != cls(per[lab]["run"])]
        if fl:
            n_flip += len(fl)
            L.append(f"- {name}: " + ", ".join(fl))
    L += ["", f"{n_flip} flips in total.", "", "### Table II (realistic model) at seed level", "",
          "| Column | 10 dB | Ref |", "|---|---|---|"]
    for col, cname in (("Perfect tracks", "perfect | ue 0.0"), ("Misses", "R table miss"), ("False tracks", "R table false"), ("Pos./vel.", "R table noise"),
                       ("Size rule", "R table size"), ("All", "R table all"), ("All + UE 1 m", "R table all + ue 1")):
        c = comp[f"sweep: {cname} - A5"]
        cells = []
        for lab in ("10 dB", "3GPP short-range reference"):
            s, r = c[lab]["seed"], c[lab]["run"]
            cells.append(f"{s['mean_diff']:+.3f} [{s['ci95_boot'][0]:+.3f}, {s['ci95_boot'][1]:+.3f}] p={s['wilcoxon_p_two_sided']:.3g} "
                         f"(run p={r['wilcoxon_p_two_sided']:.2g})" + (" FLIP" if cls(s) != cls(r) else ""))
        L.append(f"| {col} | {cells[0]} | {cells[1]} |")
    s = comp["sensing-planner - A5"]
    L.append(f"| Real sensing-planner | {s['10 dB']['seed']['mean_diff']:+.3f} p={s['10 dB']['seed']['wilcoxon_p_two_sided']:.3g} | "
             f"{s['3GPP short-range reference']['seed']['mean_diff']:+.3f} p={s['3GPP short-range reference']['seed']['wilcoxon_p_two_sided']:.3g} |")
    L += ["", "## 2. Break-even", "",
          "Definition used so far (Blocks 2-3): mean break-even = sigma at which the run-level mean paired difference to A5 (= the seed-level mean, "
          "4 runs per seed) first reaches 0 from below, linear interpolation on the grid 0, 0.1, 0.25, 0.5, 1.0; sigma = standard deviation of the "
          "injected error per horizontal axis (UE: both axes, white; realistic blocker model: along-line axis only, AR(1)). Now on the merged dense grid.", ""]
    for key, b in be.items():
        L += [f"**{key}** (sigmas {', '.join(f'{x:g}' for x in b['sigmas'])})", "", "| Margin | mean break-even | significant-advantage limit | seed-level mean diff per sigma (p) |", "|---|---|---|---|"]
        for lab in labels:
            m = b["margins"][lab]
            mb = m["mean_break_even"]
            mbs = "none" if mb is None else (">max" if mb == "inf" else f"{mb:.3f}")
            sg = "--" if m["significant_advantage_limit"] is None else f"{m['significant_advantage_limit']:g}"
            pts = " ".join(f"{y:+.3f}({p:.2g})" for y, p in zip(m["mean_diff"], m["p"]))
            L.append(f"| {SHORT.get(lab, lab)} | {mbs} | {sg} | {pts} |")
        L.append("")
    (R5 / "review3" / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {R5 / 'review3' / 'report.md'}")


if __name__ == "__main__":
    main()
