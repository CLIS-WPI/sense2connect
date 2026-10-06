"""TVT T6: sensitivity / generalisation summary (development seeds), markdown tables from the result files.

Inputs (each optional; missing parts are reported as missing):
- E2 loop delay and sensing overhead: results/TVT/T5/handover_e2_<ms>ms.json, handover_ovh_<scale>.json (fixed T5 params);
- residual self-interference: results/TVT/T6/si_visibility.json, results/TVT/T5/handover_si_<inr>.json;
- calibration / sync / bandwidth: results/TVT/T4/track/development/<cond>_map0_<cfg> vs paper-2 estimator A
  (results/P2/est_A/dev/<cfg>);
- UE speed and O-RU height variants: T4 tracks of the alias mounts, paper-2 estimator A on the variants
  (results/TVT/T6/est_A), results/TVT/T5/handover_<variant>.json;
- traffic density: the main T4 / T5 results split by density;
- second deployment: results/TVT/T1/bound_summary_intersection.json, T4 tracks of mount corner,
  results/TVT/T5/handover_intersection_{fixed,tuned}.json;
- blockage-model check: results/TVT/T6/blockage_check.json.
Seed level as everywhere (per seed over its runs and UEs, exact Wilcoxon, bootstrap CI).
Writes results/TVT/T6/summary.json and prints markdown. Run: python scripts/tvt_t6_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

T5 = ROOT / "results" / "TVT" / "T5"
MARGINS = ("15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference")
KEY_SCHEMES = ("A5", "planner_tvt", "risk_tvt", "riskneutral_tvt", "trigger_tvt", "planner_tvt_perfect", "risk_tvt_perfect", "planner_true_perfect")


def handover_table(files: list[tuple[str, Path]], title: str, out: dict) -> None:
    rows = []
    for name, f in files:
        if not f.exists():
            rows.append((name, None))
            continue
        rows.append((name, json.loads(f.read_text())))
    out[title] = {}
    print(f"### {title}\n")
    print("Outage [s/UE-min] per margin (mean over runs; in brackets: vs A5 seed-level difference and exact Wilcoxon p).\n")
    for lab in MARGINS:
        print(f"**{lab}**\n")
        print("| setting | " + " | ".join(KEY_SCHEMES) + " |")
        print("|---|" + "---|" * len(KEY_SCHEMES))
        for name, d in rows:
            if d is None or lab not in d["schemes"]:
                print(f"| {name} | " + " | ".join("missing" for _ in KEY_SCHEMES) + " |")
                continue
            cells = []
            for sch in KEY_SCHEMES:
                v = d["schemes"][lab].get(sch)
                if v is None:
                    cells.append("-")
                    continue
                va = v.get("vs_A5")
                cells.append(f"{v['outage']:.3f}" + ("" if va is None else f" ({va['mean_diff']:+.3f}, p {va['wilcoxon_p_two_sided']:.2g})"))
                out[title].setdefault(name, {}).setdefault(lab, {})[sch] = {"outage": v["outage"], "vs_A5": va}
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()


def tracker_errors(set_dir: Path, jobs) -> dict | None:
    from sim.tvt.stats import seed_summary

    if not set_dir.exists():
        return None
    per = {}
    for j in jobs:
        f = set_dir / f"{j[1]}_{j[2]}_{j[0]}.npz"
        if not f.exists():
            return None
        r = np.load(f)
        e = np.linalg.norm(r["xy"][20:] - r["ue"][20:, :, :2], axis=-1).ravel()
        per.setdefault(j[0], []).append(np.where(np.isfinite(e), e, 1e3))
    med = [float(np.median(np.concatenate(v))) for _, v in sorted(per.items())]
    p90 = [float(np.percentile(np.concatenate(v), 90)) for _, v in sorted(per.items())]
    return {"median": seed_summary(med), "p90": seed_summary(p90)}


def p2_errors(base: Path, jobs, meas_dir: Path) -> dict | None:
    from sim.tvt.stats import seed_summary

    per = {}
    for j in jobs:
        f = base / f"{j[1]}_{j[2]}_{j[0]}_track.npz"
        m = meas_dir / f"{j[1]}_{j[2]}_{j[0]}.npz"
        if not f.exists() or not m.exists():
            return None
        xy = np.load(f)["xy_ekf"]
        ue = np.load(m)["ue"]
        e = np.linalg.norm(xy[20:] - ue[20:, :, :2], axis=-1).ravel()
        per.setdefault(j[0], []).append(np.where(np.isfinite(e), e, 1e3))
    med = [float(np.median(np.concatenate(v))) for _, v in sorted(per.items())]
    p90 = [float(np.percentile(np.concatenate(v), 90)) for _, v in sorted(per.items())]
    return {"median": seed_summary(med), "p90": seed_summary(p90)}


def fmt(x) -> str:
    if x is None:
        return "missing"
    return f"{x['median']['mean']:.4f} / {x['p90']['mean']:.3f}"


def main() -> None:
    from sim.tvt.seeds import load
    from sim.tvt.stats import paired

    out = {"definition": __doc__}
    seeds = load()["development"]
    canyon = [(s, m, d) for s in seeds for m in ("lamppost", "facade") for d in ("low", "high")]
    trk = ROOT / "results" / "TVT" / "T4" / "track" / "development"
    meas = ROOT / "results" / "TVT" / "T4" / "meas" / "development"

    # --- E2 delay / overhead / SI handover
    handover_table([(f"E2 {e} ms", T5 / f"handover_e2_{e}ms.json") for e in (0, 10, 50, 100)] + [("E2 20 ms (main)", T5 / "handover.json")], "E2 loop delay", out)
    handover_table([(f"overhead x{o}", T5 / f"handover_ovh_{o}.json") for o in ("0", "0.5", "2")] + [("overhead x1 (main)", T5 / "handover.json")], "Sensing overhead", out)
    handover_table([(f"INR {i} dB", T5 / f"handover_si_{i}.json") for i in (0, 10, 20)], "Residual radar self-interference (handover)", out)

    # --- SI visibility
    sv = ROOT / "results" / "TVT" / "T6" / "si_visibility.json"
    if sv.exists():
        d = json.loads(sv.read_text())
        out["si_visibility"] = d["cells"]
        print("### Residual radar self-interference (visibility, real tracks, UE from paper-2 estimator)\n")
        print("| INR [dB] | horizon [s] | Brier raw | Brier recal. | AUC recal. | confirmed tracks / epoch |")
        print("|---|---|---|---|---|---|")
        for inr in d["inr_db"]:
            for h in (0, 1):
                r = d["cells"].get(f"inr{inr:g}|h{h}|raw")
                c = d["cells"].get(f"inr{inr:g}|h{h}|cal")
                t = d["cells"].get(f"inr{inr:g}|tracks_per_epoch")
                if r and c:
                    print(f"| {inr:g} | {h} | {r['brier']['mean']:.4f} | {c['brier']['mean']:.4f} | {c['auc']['mean']:.3f} | {t['mean']:.2f} |")
        print()
        for k, v in d["paired"].items():
            print(f"- {k}: {v['mean_diff']:+.4f} [{v['ci95_boot'][0]:+.4f}, {v['ci95_boot'][1]:+.4f}], p = {v['wilcoxon_p_two_sided']:.2g}")
        print()

    # --- calibration / sync / bandwidth (tracker vs paper-2 estimator A)
    print("### Calibration, synchronisation and bandwidth (tracker vs paper-2 estimator A; median / p90 [m])\n")
    print("| configuration | tracker, no visibility | tracker, predicted visibility | paper-2 estimator A |")
    print("|---|---|---|---|")
    out["calibration"] = {}
    for cfg in ("bw400_tdoa_s1_p2_b", "bw400_tdoa_s0_p0_b", "bw400_tdoa_s1_p5_b", "bw400_tdoa_s3_p2_b", "bw100_tdoa_s1_p2_b"):
        suf = "" if cfg == "bw400_tdoa_s1_p2_b" else f"_{cfg}"
        a = tracker_errors(trk / f"none_map0{suf}", canyon)
        b = tracker_errors(trk / f"pred_real_map0{suf}", canyon)
        c = p2_errors(ROOT / "results" / "P2" / "est_A" / "dev" / cfg, canyon, meas / cfg)
        out["calibration"][cfg] = {"none": a, "pred_real": b, "paper2_A": c}
        if b and c:
            out["calibration"][cfg]["pred_real_vs_paper2_median"] = paired(b["median"]["per_seed"], c["median"]["per_seed"])
        print(f"| {cfg} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    print()

    # --- variants
    print("### UE speed and O-RU height (tracker vs paper-2 estimator A; median / p90 [m])\n")
    print("| variant | tracker, no visibility | tracker, predicted visibility | paper-2 estimator A |")
    print("|---|---|---|---|")
    out["variants"] = {}
    for v, mounts in (("ueslow", ("lamppost_ueslow", "facade_ueslow")), ("uefast", ("lamppost_uefast", "facade_uefast")), ("h3", ("lamppost_h3",)),
                      ("h10", ("lamppost_h10",))):
        jobs = [(s, m, d) for s in seeds for m in mounts for d in ("low", "high")]
        a = tracker_errors(trk / "none_map0", jobs)
        b = tracker_errors(trk / "pred_real_map0", jobs)
        c = p2_errors(ROOT / "results" / "TVT" / "T6" / "est_A" / "bw400_tdoa_s1_p2_b", jobs, meas / "bw400_tdoa_s1_p2_b")
        out["variants"][v] = {"none": a, "pred_real": b, "paper2_A": c}
        print(f"| {v} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    lp = [(s, "lamppost", d) for s in seeds for d in ("low", "high")]
    print(f"| reference lamppost 5 m, paper speeds | {fmt(tracker_errors(trk / 'none_map0', lp))} | {fmt(tracker_errors(trk / 'pred_real_map0', lp))} | "
          f"{fmt(p2_errors(ROOT / 'results' / 'P2' / 'est_A' / 'dev' / 'bw400_tdoa_s1_p2_b', lp, meas / 'bw400_tdoa_s1_p2_b'))} |")
    print()
    handover_table([(v, T5 / f"handover_{v}.json") for v in ("ueslow", "uefast", "h3", "h10")] + [("main (canyon)", T5 / "handover.json")], "UE speed / O-RU height (handover)", out)

    # --- density
    print("### Traffic density (main runs split by density; tracker median / p90 [m])\n")
    print("| density | tracker, no visibility | tracker, predicted visibility | paper-2 estimator A |")
    print("|---|---|---|---|")
    out["density"] = {}
    for dens in ("low", "high"):
        jobs = [(s, m, dens) for s in seeds for m in ("lamppost", "facade")]
        a, b = tracker_errors(trk / "none_map0", jobs), tracker_errors(trk / "pred_real_map0", jobs)
        c = p2_errors(ROOT / "results" / "P2" / "est_A" / "dev" / "bw400_tdoa_s1_p2_b", jobs, meas / "bw400_tdoa_s1_p2_b")
        out["density"][dens] = {"none": a, "pred_real": b, "paper2_A": c}
        print(f"| {dens} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    print()
    hf = T5 / "handover.json"
    if hf.exists():
        d = json.loads(hf.read_text())
        print("Handover outage by density (mean over the runs of that density) [s/UE-min]:\n")
        print("| margin | density | " + " | ".join(KEY_SCHEMES) + " |")
        print("|---|---|" + "---|" * len(KEY_SCHEMES))
        for lab in MARGINS:
            for dens in ("low", "high"):
                cells = []
                for sch in KEY_SCHEMES:
                    v = d["schemes"][lab].get(sch)
                    if v is None:
                        cells.append("-")
                        continue
                    rows = [r for r in v["rows"] if r["job"][2] == dens]
                    cells.append(f"{np.mean([r['outage_req_s_per_min'] for r in rows]):.3f}")
                print(f"| {lab} | {dens} | " + " | ".join(cells) + " |")
        print()

    # --- second deployment
    print("### Second deployment: four-way intersection (mount corner)\n")
    bi = ROOT / "results" / "TVT" / "T1" / "bound_summary_intersection.json"
    xj = [(s, "corner", d) for s in seeds for d in ("low", "high")]
    if bi.exists():
        b = json.loads(bi.read_text())["configs"]
        print(f"- PEB (single epoch, main configuration): LoS-only {b['mapsweep|los']['all']['mean']:.4f} m, map-aided {b['mapsweep|map_0']['all']['mean']:.4f} m "
              f"(blocked epochs {b['mapsweep|los']['blocked']['mean']:.3f} / {b['mapsweep|map_0']['blocked']['mean']:.4f} m)")
    a, bb = tracker_errors(trk / "none_map0", xj), tracker_errors(trk / "pred_real_map0", xj)
    out["intersection"] = {"tracker_none": a, "tracker_pred_real": bb}
    print(f"- tracker median / p90 [m]: no visibility {fmt(a)}, predicted visibility {fmt(bb)}; paper-2 estimator: not applicable (single-street map)")
    if a and bb:
        out["intersection"]["pred_real_vs_none_median"] = paired(bb["median"]["per_seed"], a["median"]["per_seed"])
        v = out["intersection"]["pred_real_vs_none_median"]
        print(f"- predicted vs no visibility, median: {v['mean_diff']:+.4f} m [{v['ci95_boot'][0]:+.4f}, {v['ci95_boot'][1]:+.4f}], p = {v['wilcoxon_p_two_sided']:.2g}")
    print()
    handover_table([("canyon parameters", T5 / "handover_intersection_fixed.json"), ("re-tuned on intersection tuning seeds", T5 / "handover_intersection_tuned.json")],
                   "Second deployment (handover)", out)

    # --- blockage model
    bc = ROOT / "results" / "TVT" / "T6" / "blockage_check.json"
    if bc.exists():
        d = json.loads(bc.read_text())
        print("### Blockage model: 3GPP model B vs Sionna RT first-order diffraction (box blocker crossing the LoS)\n")
        print("| case | positions with model-B loss >= 10 dB | agreement on >= 10 dB | median abs. diff. where blocked [dB] | max loss Sionna / model B [dB] |")
        print("|---|---|---|---|---|")
        for k, v in d["cases"].items():
            md = v["median_abs_diff_blocked_db"]
            print(f"| {k} | {v['n_blocked_modelB_ge10']} | {v['agreement_blocked_ge10db']:.2f} | {'-' if md is None else f'{md:.1f}'} | "
                  f"{v['max_loss_diffraction_db']:.1f} / {v['max_loss_model_b_db']:.1f} |")
        print()
    (ROOT / "results" / "TVT" / "T6" / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
