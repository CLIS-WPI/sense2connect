"""TVT third round: what changed from the isotropic single UPA (results/TVT_iso) to the back-to-back TR 38.901 panels
(results/TVT), development seeds; markdown tables from the result files only.
Run: python scripts/tvt_panels_compare.py > results/TVT/panels_vs_iso.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ISO, PAN = ROOT / "results" / "TVT_iso", ROOT / "results" / "TVT"
MARGINS = ("15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference")
REAL = ("trigger_tvt", "trigger_learned", "planner_p2", "planner_tvt", "risk_tvt", "riskneutral_tvt")


def load(base: Path, rel: str):
    f = base / rel
    return json.loads(f.read_text()) if f.exists() else None


def pv(v) -> str:
    return "-" if not v else f"{v['mean_diff']:+.3f} ({v['wilcoxon_p_two_sided']:.2g})"


def m(v, scale=1.0, fmt="{:.3f}") -> str:
    return "-" if v is None else fmt.format(v * scale)


def t1() -> None:
    a, b = load(ISO, "T1/bound_summary.json"), load(PAN, "T1/bound_summary.json")
    if not (a and b):
        return
    print("### T1 position error bound (median over epochs, mean over seeds) [mm]\n")
    print("| configuration | isotropic UPA | back-to-back panels | isotropic, blocked epochs | panels, blocked epochs |")
    print("|---|---|---|---|---|")
    for k in ("mapsweep|los", "mapsweep|map_0", "mapsweep|map_0.1", "mapsweep|map_1", "mapsweep|map_inf"):
        if k in a["configs"] and k in b["configs"]:
            ca, cb = a["configs"][k], b["configs"][k]
            print(f"| {k.split('|')[1]} | {m(ca['all']['mean'], 1e3, '{:.1f}')} | {m(cb['all']['mean'], 1e3, '{:.1f}')} | "
                  f"{m(ca['blocked']['mean'], 1e3, '{:.1f}')} | {m(cb['blocked']['mean'], 1e3, '{:.1f}')} |")
    print()


def t3() -> None:
    a, b = load(ISO, "T3/visibility.json"), load(PAN, "T3/visibility.json")
    if not (a and b):
        return
    print("### T3 visibility prediction (pooled, development)\n")
    print("| tracks / UE / horizon / paths | AUC isotropic | AUC panels | Brier isotropic | Brier panels |")
    print("|---|---|---|---|---|")
    for k in a["pooled"]:
        if k in b["pooled"] and k.endswith("|all") and k.split("|")[0] in ("perfect", "real", "realcal"):
            x, y = a["pooled"][k], b["pooled"][k]
            print(f"| {k} | {m(x.get('auc'))} | {m(y.get('auc'))} | {m(x.get('brier'))} | {m(y.get('brier'))} |")
    print()


def t4() -> None:
    a, b = load(ISO, "T4/eval.json"), load(PAN, "T4/eval.json")
    if not (a and b):
        return
    print("### T4 UE position error (median / p90 over epochs, mean over seeds) [m]\n")
    print("| condition | state | isotropic median | panels median | isotropic p90 | panels p90 |")
    print("|---|---|---|---|---|---|")
    for c in a["conditions"]:
        if c not in b["conditions"]:
            continue
        for st in a["conditions"][c]:
            x, y = a["conditions"][c][st], b["conditions"][c].get(st)
            if not isinstance(x, dict) or "median" not in x or not y:
                continue
            print(f"| {c} | {st} | {m(x['median']['mean'], fmt='{:.4f}')} | {m(y['median']['mean'], fmt='{:.4f}')} | "
                  f"{m(x.get('p90', {}).get('mean'))} | {m(y.get('p90', {}).get('mean'))} |")
    print()


def radar() -> None:
    """Detections per frame (blind CFAR, tracker budget train 4): paper-1 radar vs the two panels."""
    from sim.tvt.panels import detections_dir

    rows = []
    for f in sorted((PAN / "radar" / "det").glob("*/*/seed_*/detections_1024.json"))[:8]:
        mount, dens, sd = f.parts[-4], f.parts[-3], f.parts[-2]
        p1 = ROOT / "results" / "cache" / mount / dens / sd / "detections_1024.json"
        if not p1.exists():
            continue
        dp, d1 = json.loads(f.read_text()), json.loads(p1.read_text())
        key = next((k for k in dp["frames"][0]["detections"] if k.startswith("blind:4:")), None)
        if key is None or key not in d1["frames"][0]["detections"]:
            continue
        rows.append((f"{mount} {dens} {sd}", np.mean([len(fr["detections"][key]) for fr in d1["frames"]]),
                     np.mean([len(fr["detections"][key]) for fr in dp["frames"]]), dp.get("detections_per_panel_blind_train4")))
    del detections_dir
    if rows:
        print("### Radar detections per frame (blind clutter removal, CFAR train 4; first jobs)\n")
        print("| job | paper-1 radar (tilted, isotropic) | panels (union) | per panel (+x, -x), all frames |")
        print("|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]} |")
        print()


def t5(model: str) -> None:
    a, b = load(ISO, f"T5/handover_{model}.json"), load(PAN, f"T5/handover_{model}.json")
    if not (a and b):
        return
    print(f"### T5 under {model} signaling: outage [s/UE-min] and difference to A5 (exact Wilcoxon p), primary (wrap-masked)\n")
    print("| margin | scheme | isotropic | panels | isotropic vs A5 | panels vs A5 | panels parameters |")
    print("|---|---|---|---|---|---|---|")
    for lab in MARGINS:
        if lab not in a["schemes"] or lab not in b["schemes"]:
            continue
        for sch in ("A5", "CHO", "A3") + REAL + ("planner_tvt_perfect", "planner_true_perfect"):
            x, y = a["schemes"][lab].get(sch), b["schemes"][lab].get(sch)
            if not y:
                continue
            print(f"| {lab} | {sch} | {m(x['outage'] if x else None)} | {m(y['outage'])} | {pv(x.get('vs_A5') if x else None)} | {pv(y.get('vs_A5'))} | "
                  f"{json.dumps(y.get('params'))} |")
    print(f"\nReference margin: isotropic {a.get('margin_ref_db', float('nan')):.1f} dB, panels {b.get('margin_ref_db', float('nan')):.1f} dB.\n")


def t6(model: str) -> None:
    files = [("E2 0 ms", "e2_0ms"), ("E2 100 ms", "e2_100ms"), ("overhead x0", "ovh_0"), ("overhead x2", "ovh_2"), ("INR 40 dB", "si_40"),
             ("slow UEs", "ueslow"), ("fast UEs", "uefast"), ("O-RU 3 m", "h3"), ("O-RU 10 m", "h10"), ("intersection, re-tuned", "intersection_tuned")]
    print(f"### T6 under {model} signaling: best real-input scheme and exact-position perfect-track planner vs A5, isotropic -> panels\n")
    print("| setting | margin | A5 iso -> panels | best real-input vs A5 iso | panels | planner_true_perfect vs A5 iso | panels |")
    print("|---|---|---|---|---|---|---|")
    for name, tag in files:
        a, b = load(ISO, f"T5/handover_{tag}_{model}.json"), load(PAN, f"T5/handover_{tag}_{model}.json")
        if not (a and b):
            continue
        for lab in MARGINS:
            if lab not in b["schemes"]:
                continue

            def best(cell):
                r = [k for k in REAL if k in cell]
                if not r:
                    return None, None
                k = min(r, key=lambda z: cell[z]["outage"])
                return k, cell[k].get("vs_A5")

            ka, va = best(a["schemes"].get(lab, {}))
            kb, vb = best(b["schemes"][lab])
            print(f"| {name} | {lab} | {m(a['schemes'].get(lab, {}).get('A5', {}).get('outage'))} -> {m(b['schemes'][lab]['A5']['outage'])} | "
                  f"{ka} {pv(va)} | {kb} {pv(vb)} | {pv(a['schemes'].get(lab, {}).get('planner_true_perfect', {}).get('vs_A5'))} | "
                  f"{pv(b['schemes'][lab].get('planner_true_perfect', {}).get('vs_A5'))} |")
    print()


def main() -> None:
    sys.path.insert(0, str(ROOT))
    print("## Isotropic single UPA (results/TVT_iso) vs back-to-back TR 38.901 panels (results/TVT), development seeds\n")
    t1()
    radar()
    t3()
    t4()
    for model in ("failure_aware", "ideal"):
        t5(model)
    for model in ("failure_aware", "ideal"):
        t6(model)


if __name__ == "__main__":
    main()
