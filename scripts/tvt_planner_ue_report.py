"""TVT fourth round, human item 4: planner with PERFECT blocker tracks, UE position from the T4 tracker vs from truth,
vs A5, per margin and per blocker class (back-to-back panels, development seeds 1001-1010; primary = failure-aware
signaling, wrap-masked: the 1 s after each UE wrap excluded; ideal signaling as supplement).

Reads only result files (no run): results/TVT/J3_diagnosis/diagnosis_<model>.json (scripts/tvt_j3_diagnosis.py part c:
seed-level outage per scheme, per blocker class of the 10 dB event covering the outage step, paired exact Wilcoxon over
the 10 development seeds, bootstrap 95 % CI of the seed-level mean difference) and its tracker-error part b.
Schemes: planner_true_perfect (UE = truth), planner_tvt_perfect (UE = T4 tracker, pred_real, sigma_map 0), A5;
parameters tuned on the tuning seeds for each signaling model (results/TVT/T5/handover_<model>.json).
Run: python scripts/tvt_planner_ue_report.py > results/TVT/J3_diagnosis/planner_ue_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J3 = ROOT / "results" / "TVT" / "J3_diagnosis"
MARGINS = ("15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference")
CLASSES = ("bus/truck", "pedestrian", "car", "no LoS blocker", "none")
TRUE, TVT = "planner_true_perfect", "planner_tvt_perfect"


def pv(v: dict) -> str:
    """Seed-level mean difference [95 % CI] (exact Wilcoxon p; seeds lower / higher)."""
    lo, hi = v["ci95_boot"]
    return f"{v['mean_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] (p {v['wilcoxon_p_two_sided']:.2g}; {v['n_lower']}/{v['n_higher']})"


def report(model: str, primary: bool) -> None:
    f = J3 / f"diagnosis_{model}.json"
    d = json.loads(f.read_text())
    c = d["c"]
    title = "PRIMARY: failure-aware signaling" if primary else "Supplement: ideal signaling"
    print(f"## {title} (wrap-masked) [s/UE-min]\n")
    print("Seed-level mean over the 10 development seeds (4 runs x 2 UEs per seed); differences: mean [bootstrap 95 % CI] "
          "(exact two-sided Wilcoxon p over 10 seeds; number of seeds lower / higher). Smallest possible p = 0.002.\n")
    print("| margin | A5 | truth UE | tracker UE | truth UE - A5 | tracker UE - A5 | tracker UE - truth UE |")
    print("|---|---|---|---|---|---|---|")
    for lab in MARGINS:
        cell = c.get(lab)
        if not cell or TRUE not in cell or TVT not in cell:
            continue
        print(f"| {lab} | {cell['A5']['outage']:.3f} | {cell[TRUE]['outage']:.3f} | {cell[TVT]['outage']:.3f} | {pv(cell[TRUE]['gap_vs_A5'])} | "
              f"{pv(cell[TVT]['gap_vs_A5'])} | {pv(cell[TVT]['vs_true_perfect'])} |")
    print("\nWithin the 1 s after a UE wrap (excluded from the primary metric above; tracker re-acquisition after the scenario "
          "teleports the UE): tracker UE - truth UE\n")
    print("| margin | A5 | truth UE | tracker UE | tracker UE - truth UE |")
    print("|---|---|---|---|---|")
    for lab in MARGINS:
        cell = c.get(lab)
        if not cell or TRUE not in cell or TVT not in cell:
            continue
        print(f"| {lab} | {cell['A5']['outage_1s_after_ue_wrap']:.3f} | {cell[TRUE]['outage_1s_after_ue_wrap']:.3f} | "
              f"{cell[TVT]['outage_1s_after_ue_wrap']:.3f} | {pv(cell[TVT]['vs_true_perfect_1s_after_ue_wrap'])} |")
    print("\n### Per blocker class\n")
    print("Outage steps attributed to the class of the 10 dB blockage event of the UE covering the step (onset - 2 s to the "
          "end; serving cell first, then the other cell; 'none' = no event). Class outages add up to the total.\n")
    for lab in MARGINS:
        cell = c.get(lab)
        if not cell or TRUE not in cell or TVT not in cell:
            continue
        print(f"**{lab}**\n")
        print("| blocker class | A5 | truth UE | tracker UE | truth UE - A5 | tracker UE - A5 | tracker UE - truth UE |")
        print("|---|---|---|---|---|---|---|")
        for k in CLASSES:
            a5 = cell["A5"]["by_class"][k]
            if max(a5, cell[TRUE]["by_class"][k], cell[TVT]["by_class"][k]) == 0.0:
                print(f"| {k} | 0 | 0 | 0 | - | - | - |")
                continue
            print(f"| {k} | {a5:.3f} | {cell[TRUE]['by_class'][k]:.3f} | {cell[TVT]['by_class'][k]:.3f} | {pv(cell[TRUE]['gap_by_class'][k])} | "
                  f"{pv(cell[TVT]['gap_by_class'][k])} | {pv(cell[TVT]['vs_true_perfect_by_class'][k])} |")
        print()
    w = d["b_tracker_error"]["windows"]
    rows = [(k.split(" | ", 1)[1], v) for k, v in w.items() if k.startswith("tvt | ") and "(20 dB)" not in k or k.startswith(f"tvt | switch decisions of {TVT}")]
    if rows:
        print("Tracker UE error [m, horizontal] (J3 diagnosis part b; seed-level mean [bootstrap 95 % CI]):\n")
        print("| window | median | p90 |")
        print("|---|---|---|")
        for name, v in rows:
            print(f"| {name} | {v['median']['mean']:.3f} [{v['median']['ci95_boot'][0]:.3f}, {v['median']['ci95_boot'][1]:.3f}] | "
                  f"{v['p90']['mean']:.3f} [{v['p90']['ci95_boot'][0]:.3f}, {v['p90']['ci95_boot'][1]:.3f}] |")
        print()
    print(f"Source: {f.relative_to(ROOT)}\n")


def main() -> None:
    print("# Planner with perfect blocker tracks: UE position from the tracker vs from truth, vs A5 (back-to-back panels, "
          "development seeds)\n")
    print(__doc__.split("\n\n")[0].replace("\n", " ") + "\n")
    report("failure_aware", True)
    report("ideal", False)


if __name__ == "__main__":
    main()
