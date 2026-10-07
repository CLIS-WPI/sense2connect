"""TVT signaling round (human decision before the freeze): markdown tables of T5 and the handover parts of T6
under "ideal" and "failure_aware" signaling, primary metrics wrap-masked, from the result files only.

Inputs: results/TVT/T5/handover_<model>.json and the T6 files written by scripts/tvt_signaling_runs.sh.
Run: python scripts/tvt_signaling_summary.py > results/TVT/T6/signaling_tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T5 = ROOT / "results" / "TVT" / "T5"
MODELS = ("ideal", "failure_aware")


def load(name: str) -> dict | None:
    f = T5 / name
    return json.loads(f.read_text()) if f.exists() else None


def pv(v: dict | None) -> str:
    return "-" if v is None else f"{v['mean_diff']:+.3f} ({v['wilcoxon_p_two_sided']:.2g})"


def main_tables() -> None:
    for m in MODELS:
        d = load(f"handover_{m}.json")
        if d is None:
            continue
        print(f"### T5 under {m} signaling (re-tuned on the tuning seeds; primary = wrap-masked)\n")
        print(f"Service model {d['model']}, reference margin {d['margin_ref_db']:.1f} dB; outage at 400 Mbit/s [s/UE-min], mean over 40 runs x 2 UEs; "
              "vs A5: seed-level mean difference (exact Wilcoxon p).\n")
        for lab, cell in d["schemes"].items():
            print(f"**{lab}**\n")
            print("| scheme | outage | unmasked | HO/min | RLF/min | HOF rate | vs A5 (p) | vs A5 unmasked | parameters |")
            print("|---|---|---|---|---|---|---|---|---|")
            seeds = sorted(cell["A5"]["per_seed"])
            from sim.tvt.stats import paired

            for sch, v in cell.items():
                if " vs " in sch:
                    continue
                um = None
                if sch != "A5" and "outage_unmasked" in v["per_seed"][seeds[0]]:
                    um = paired([v["per_seed"][s]["outage_unmasked"] for s in seeds], [cell["A5"]["per_seed"][s]["outage_unmasked"] for s in seeds])
                print(f"| {sch} | {v['outage']:.3f} | {v.get('outage_unmasked', float('nan')):.3f} | {v['ho_per_min']:.2f} | {v.get('rlf_per_min', 0):.3f} | "
                      f"{v.get('hof_rate', 0):.3f} | {pv(v.get('vs_A5'))} | {pv(um)} | {json.dumps(v['params'])} |")
            extra = [(k, v) for k, v in cell.items() if " vs " in k]
            if extra:
                print()
                for k, v in extra:
                    print(f"- {k}: {pv(v)}")
            print()


def compact(title: str, files: list[tuple[str, str]]) -> None:
    """Per sweep point and margin: A5 outage, schemes significantly better / worse than A5 (p < 0.05)."""
    print(f"### {title}\n")
    print("| setting | margin | A5 | CHO vs A5 | best scheme (outage) | significantly better than A5 | significantly worse than A5 |")
    print("|---|---|---|---|---|---|---|")
    for name, fn in files:
        d = load(fn)
        if d is None:
            print(f"| {name} | (missing {fn}) | | | | | |")
            continue
        for lab, cell in d["schemes"].items():
            sch = {k: v for k, v in cell.items() if " vs " not in k}
            best = min(sch, key=lambda k: sch[k]["outage"])
            better = [f"{k} {v['vs_A5']['mean_diff']:+.3f}" for k, v in sch.items() if k != "A5" and v.get("vs_A5") and v["vs_A5"]["wilcoxon_p_two_sided"] < 0.05
                      and v["vs_A5"]["mean_diff"] < 0]
            worse = [k for k, v in sch.items() if k != "A5" and v.get("vs_A5") and v["vs_A5"]["wilcoxon_p_two_sided"] < 0.05 and v["vs_A5"]["mean_diff"] > 0]
            print(f"| {name} | {lab} | {sch['A5']['outage']:.3f} | {pv(sch.get('CHO', {}).get('vs_A5'))} | {best} ({sch[best]['outage']:.3f}) | "
                  f"{', '.join(better) or '-'} | {', '.join(worse) or '-'} |")
    print()


def main() -> None:
    sys.path.insert(0, str(ROOT))
    main_tables()
    for m in MODELS:
        compact(f"T6 E2 loop delay ({m})", [(f"E2 {e} ms", f"handover_e2_{e}ms_{m}.json") for e in (0, 10, 50, 100)] + [("E2 20 ms (main)", f"handover_{m}.json")])
        compact(f"T6 sensing overhead ({m})", [(f"overhead x{o}", f"handover_ovh_{o}_{m}.json") for o in ("0", "0.5", "2")]
                + [("overhead x0 re-tuned", f"handover_ovh0_tuned_{m}.json")])
        compact(f"T6 residual self-interference ({m})", [(f"INR {i} dB", f"handover_si_{i}_{m}.json") for i in (0, 10, 20, 30, 40)])
        compact(f"T6 UE speed / O-RU height ({m})", [(v, f"handover_{v}_{m}.json") for v in ("ueslow", "uefast", "h3", "h10")])
        compact(f"T6 second deployment, intersection ({m})", [("canyon parameters", f"handover_intersection_fixed_{m}.json"),
                                                                ("re-tuned on intersection tuning seeds", f"handover_intersection_tuned_{m}.json")])
    compact("T6 failure-model sensitivity (failure_aware, parameters of the primary failure-aware tuning)",
            [("primary (T310 1 s, Qout -8 dB, tau_RE 1.83 s)", "handover_failure_aware.json")]
            + [(f"T310 {t} s", f"handover_fs_t310_{t}_failure_aware.json") for t in ("0.2", "0.5")]
            + [(f"Qout/Qin {q:+d} dB", f"handover_fs_q{q}_failure_aware.json") for q in (-3, 3)]
            + [("tau_RE 0.23 s", "handover_fs_taure_0.23_failure_aware.json")])


if __name__ == "__main__":
    main()
