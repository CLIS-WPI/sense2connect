"""Markdown tables for the TVT milestone reports, generated from the result files (no number typed by hand).

Run: python3 scripts/tvt_report_tables.py t4|t5|t6  > table.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ("all", "[-1,-0.5) s", "[-0.5,0) s", "during", "elsewhere")


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def t4() -> None:
    d = json.loads((ROOT / "results" / "TVT" / "T4" / "eval.json").read_text())
    print("Seed level (per seed over 4 runs x 2 UEs, epochs after 2 s): mean over the 10 development seeds [bootstrap 95 % CI].")
    print()
    print("| condition | window | median [m] | p90 [m] | RMSE [m] | share > 0.1 m | median / map-aided PEB |")
    print("|---|---|---|---|---|---|---|")
    for c, cell in d["conditions"].items():
        for w in WINDOWS:
            x = cell[w]
            r = x.get("median_over_PEB_map0")
            print(f"| {c} | {w} | {x['median']['mean']:.4f} [{x['median']['ci95_boot'][0]:.4f}, {x['median']['ci95_boot'][1]:.4f}] | "
                  f"{x['p90']['mean']:.3f} | {x['rmse']['mean']:.2f} | {x['share_gt_0.1']['mean']:.3f} | {'-' if r is None else f'{r['mean']:.1f}'} |")
    print()
    print("Covariance consistency (share of epochs inside the tracker's 95 % ellipse): " + ", ".join(
        f"{c} {cell['share_inside_95pct_ellipse']:.2f}" for c, cell in d["conditions"].items() if "share_inside_95pct_ellipse" in cell))
    print()
    print("Paired seed-level tests (exact Wilcoxon over 10 seeds; negative = first condition better):")
    print()
    print("| comparison | window | metric | mean diff [m] | 95 % CI | p |")
    print("|---|---|---|---|---|---|")
    for k, v in d["paired"].items():
        a, w, m = [x.strip() for x in k.split("|")]
        print(f"| {a} | {w} | {m} | {v['mean_diff']:+.4f} | [{v['ci95_boot'][0]:+.4f}, {v['ci95_boot'][1]:+.4f}] | {v['wilcoxon_p_two_sided']:.3g} |")


def t5(name: str = "handover.json") -> None:
    d = json.loads((ROOT / "results" / "TVT" / "T5" / name).read_text())
    print(f"Service model {d['model']}, 3GPP reference margin {d['margin_ref_db']:.1f} dB. Outage at 400 Mbit/s [s/UE-min], mean over 40 runs x 2 UEs;"
          " vs A5: seed-level mean difference, exact Wilcoxon p.")
    print()
    for lab, schemes in d["schemes"].items():
        print(f"**{lab}**")
        print()
        print("| scheme | outage | HO/min | ping-pong | vs A5: diff [CI] (p) | tuned parameters |")
        print("|---|---|---|---|---|---|")
        for sch, v in schemes.items():
            if " vs " in sch:
                continue
            va = v.get("vs_A5")
            vs = "-" if va is None else f"{va['mean_diff']:+.3f} [{va['ci95_boot'][0]:+.3f}, {va['ci95_boot'][1]:+.3f}] ({va['wilcoxon_p_two_sided']:.2g})"
            print(f"| {sch} | {v['outage']:.3f} | {v['ho_per_min']:.2f} | {v['ping_pong']:.2f} | {vs} | {json.dumps(v['params'])} |")
        extra = [(k, v) for k, v in schemes.items() if " vs " in k]
        if extra:
            print()
            for k, v in extra:
                print(f"- {k}: {v['mean_diff']:+.3f} [{v['ci95_boot'][0]:+.3f}, {v['ci95_boot'][1]:+.3f}] s/UE-min, p = {v['wilcoxon_p_two_sided']:.2g}")
        print()


if __name__ == "__main__":
    {"t4": t4, "t5": t5}[sys.argv[1]](*sys.argv[2:])
