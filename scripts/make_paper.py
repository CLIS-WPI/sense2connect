"""Regenerate every paper figure, the table and paper/numbers.tex, then compile the paper.

Steps:
1. Run the figure/table scripts (each one command, from saved results):
   fig_blockage_table.py, fig_leadtime.py, fig_onset.py,
   fig_outage_margin.py, fig_headroom.py.
2. Build the number catalog from results (results/M5/numbers_catalog.json):
   every value the paper may quote, under a descriptive key, as a plain
   string with its precision fixed here.
3. Read the macro names (every ``\\num...`` token) from the header comment of
   paper/main.tex, map each to a catalog key (MACRO_MAP, default: same name)
   and write paper/numbers.tex (``\\newcommand{\\Name}{value}``). Every
   macro in main.tex without a mapping, and every mapped key missing from
   the catalog, is reported; numbers.tex then defines only the rest.
4. Compile paper/main.tex with latexmk if it is installed; otherwise print
   the command.

Run in the container: ``python scripts/make_paper.py [--leadtime-recompute]``.
By default fig_leadtime.py re-plots from results/M5/leadtime.json; with
``--leadtime-recompute`` it replays the trackers (about 10–15 min, 4 workers).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"

FIGURE_SCRIPTS = ["fig_blockage_table.py", "fig_leadtime.py", "fig_onset.py", "fig_outage_margin.py", "fig_headroom.py"]

# main.tex macro name -> catalog key. Filled in once paper/main.tex (with its
# macro list) is in the repository; unmapped macros are reported.
MACRO_MAP: dict[str, str] = {}  # empty: catalog keys are the main.tex macro names

MACRO_NAME = re.compile(r"\\(num[A-Za-z]+)")


def run_scripts(leadtime_recompute: bool) -> None:
    for name in FIGURE_SCRIPTS:
        cmd = [sys.executable, str(ROOT / "scripts" / name)]
        if name == "fig_leadtime.py" and not leadtime_recompute and (RESULTS / "M5" / "leadtime.json").exists():
            cmd.append("--plot-only")
        print("$", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}\\%"


def _rng(values: list[float], fmt: str, pct: bool = False) -> str:
    lo, hi = min(values), max(values)
    a, b = (fmt.format(100.0 * lo), fmt.format(100.0 * hi)) if pct else (fmt.format(lo), fmt.format(hi))
    text = a if a == b else f"{a}--{b}"
    return text + ("\\%" if pct else "")


EVAL = "evaluation seeds 1001-1010 (configs/seeds.yaml), 40 jobs (2 mounts x 2 densities x 10 seeds), 2 UEs each"
CHAR_CFG = "configs/m2_scenario.yaml two-O-RU config (8x8, 1024 SC), comm traces at 0.1 s, serving cell = strongest unblocked, 10 dB LoS events, min gap 0.5 s"
M3_CFG = "M3 rework: 3GPP TR 38.802 budget, 400 Mbit/s service rate, model B every 10 ms, tau_HO 20 ms, E2 delay 20 ms; per-margin tuning on tuning seeds 101-105"


def catalog() -> dict[str, dict[str, Any]]:
    """One entry per main.tex macro: value (main.tex format) and its source."""
    out: dict[str, dict[str, Any]] = {}

    def put(name, value, *, script, config, aggregation, raw, seeds=EVAL, note=""):
        out[name] = {"value": value, "script": script, "config": config, "seeds": seeds, "aggregation": aggregation, "raw": raw, "note": note}

    R5 = RESULTS / "M5"
    m3 = json.loads((RESULTS / "M3" / "metrics.json").read_text())
    gen = json.loads((RESULTS / "M3" / "genie.json").read_text())
    hyb = json.loads((RESULTS / "M3" / "hybrid.json").read_text())
    char = json.loads((R5 / "characterization.json").read_text())
    tab = json.loads((R5 / "blockage_table.json").read_text())["evaluation"]
    trk = json.loads((R5 / "tracking.json").read_text())
    fs = json.loads((R5 / "foresight.json").read_text())
    labels = m3["budget"]["labels"]
    i10, iref = labels.index("10 dB"), labels.index("3GPP short-range reference")
    tau = f"{gen['tau_ho_default_s']:.3f}"
    mounts = ("lamppost", "facade")

    # --- characterization (fig_blockage_table.py)
    sb = "scripts/fig_blockage_table.py -> results/M5/characterization.json, blockage_table.json"
    ov = char["overall"]
    put("numEventsPerMin", f"{ov['events_per_ue_min']:.1f}", script=sb, config=CHAR_CFG, aggregation="pooled over all jobs and both mounts", raw=ov["events_per_ue_min"])
    pcm = char["per_class_per_mount"]
    for macro, cls in (("numBusShare", "bus/truck"), ("numPedShare", "pedestrian")):
        shares = [pcm[m][cls]["n_events"] / sum(pcm[m][c]["n_events"] for c in pcm[m]) for m in mounts]
        put(macro, _rng(shares, "{:.0f}", pct=True), script=sb, config=CHAR_CFG, aggregation="min-max over mounts of the per-mount pooled share", raw=dict(zip(mounts, shares)))
    sp = [-float(char["same_cell_surviving_path"][m]["all"]["p10_p50_p90_db"]["0.5"]) for m in mounts]
    put("numSameCellLoss", _rng(sp, "{:.0f}"), script=sb, config=CHAR_CFG + "; M1.5 part-1 definition (best_alt_db) on the serving cell",
        aggregation="min-max over mounts of the per-mount median over events (loss in dB, sign dropped)", raw=dict(zip(mounts, sp)))
    for macro, cls in (("numOtherCellBus", "bus/truck"), ("numOtherCellPed", "pedestrian")):
        shares = [tab["other_oru"][m]["classes"][cls]["unblocked"] / tab["other_oru"][m]["classes"][cls]["n_events"] for m in mounts]
        put(macro, _rng(shares, "{:.0f}", pct=True), script=sb, config=CHAR_CFG + "; other cell clear = LoS loss < 3 dB on every 10 dB snapshot",
            aggregation="min-max over mounts of the per-mount pooled share", raw=dict(zip(mounts, shares)))
    for macro, cls in (("numOnsetBus", "bus/truck"), ("numOnsetPed", "pedestrian")):
        v = m3["event_stats"]["evaluation"][cls]["onset_10_90_s"]["p50"]
        put(macro, f"{v:.2f}", script="scripts/run_m3.py -> results/M3/metrics.json (event_stats)", config="10 ms analytic model B, fixed cell = strongest unblocked, 10 dB LoS events, min gap 0.5 s",
            aggregation="median over all evaluation events of the class (pooled over mounts)", raw=v)

    # --- tracking (run_m5_tracking.py, fig_leadtime.py)
    st = "scripts/run_m5_tracking.py -> results/M5/tracking.json (detections via stage-validated cache, replay helpers of run_m2_followup.py)"
    fin = trk["variants"]["image|map"]
    fin_cfg = "final M2 config: detector budget 4 FA/CPI, blind clutter, image-method ghost handling (budget-4 image/blind pick), map-constrained tracker (MAP_TUNED)"
    put("numBusPd", f"{fin['track_pd']['bus/truck']:.2f}", script=st, config=fin_cfg, aggregation="pooled over all evaluation jobs (track hits / ground-truth samples)", raw=fin["track_pd"]["bus/truck"])
    put("numPedPd", f"{fin['track_pd']['pedestrian']:.2f}", script=st, config=fin_cfg, aggregation="pooled over all evaluation jobs", raw=fin["track_pd"]["pedestrian"])
    put("numFA", f"{fin['false_alarms_per_cpi']:.1f}", script=st, config=fin_cfg, aggregation="pooled: false alarms / CPIs over all evaluation jobs", raw=fin["false_alarms_per_cpi"])
    for macro, cls in (("numLeadBus", "bus/truck"), ("numLeadPed", "pedestrian")):
        shares = [fin["leads"][m][cls]["hits"]["0.5"] / fin["leads"][m][cls]["n"] for m in mounts]
        put(macro, _rng(shares, "{:.0f}", pct=True), script=st, config=fin_cfg + "; 10 dB LoS events on oru-0 (M2 event definition), confirmed track in the class gate 0.5 s before onset",
            aggregation="min-max over mounts of the per-mount pooled share", raw=dict(zip(mounts, shares)))
    ung, mpg = trk["variants"]["noghost|unconstrained"], trk["variants"]["noghost|map"]
    ng_cfg = "detector budget 4 FA/CPI, blind clutter, NO ghost handling (budget-4 none/blind pick: Pfa 1e-5, train 4, eps 3 m); same detections for both trackers; unconstrained EKF (association 6 m, coast 4, q 1.0) -> map-constrained tracker (MAP_TUNED)"
    put("numMapBus", f"{ung['track_pd']['bus/truck']:.2f} to {mpg['track_pd']['bus/truck']:.2f}", script=st, config=ng_cfg, aggregation="pooled over all evaluation jobs",
        raw=[ung["track_pd"]["bus/truck"], mpg["track_pd"]["bus/truck"]])
    put("numMapPed", f"{ung['track_pd']['pedestrian']:.2f} to {mpg['track_pd']['pedestrian']:.2f}", script=st, config=ng_cfg, aggregation="pooled over all evaluation jobs",
        raw=[ung["track_pd"]["pedestrian"], mpg["track_pd"]["pedestrian"]])
    put("numMapCross", f"{ung['cross_lane_rmse_mps']['bus/truck']:.1f} to {mpg['cross_lane_rmse_mps']['bus/truck']:.1f}", script=st, config=ng_cfg + "; bus/truck cross-lane velocity RMSE [m/s], tracks older than 1 s",
        aggregation="pooled RMSE over all matched track samples", raw=[ung["cross_lane_rmse_mps"]["bus/truck"], mpg["cross_lane_rmse_mps"]["bus/truck"]])
    ui = trk["variants"]["image|unconstrained"]
    put("numGhostGain", f"{ung['track_pd']['bus/truck']:.2f} to {ui['track_pd']['bus/truck']:.2f}", script=st,
        config="UNCONSTRAINED EKF tracker (as in M2); detector budget 4 FA/CPI, blind clutter: no-ghost pick (Pfa 1e-5, eps 3 m) -> image-method pick (Pfa 1e-3, eps 3 m, ghost association 6 m)",
        aggregation="pooled over all evaluation jobs and both mounts", raw=[ung["track_pd"]["bus/truck"], ui["track_pd"]["bus/truck"]],
        note="the earlier default 0.80 to 0.86 was the lamppost-only value from the M2 report; this value pools both mounts")

    # --- link budget
    sm3 = "scripts/run_m3.py -> results/M3/metrics.json"
    put("numRefMargin", f"{m3['budget']['margin_ref_db']:.1f}", script=sm3, config=M3_CFG, seeds="tuning seeds 101-105 (median over tuning jobs, UEs, 10 ms steps)",
        aggregation="median best-cell unblocked SNR minus SNR_req", raw=m3["budget"]["margin_ref_db"])
    put("numSNRreq", f"{m3['budget']['snr_req_db']:.1f}", script=sm3, config="Shannon SNR for 400 Mbit/s on 122.88 MHz, no overhead", seeds="n/a", aggregation="n/a", raw=m3["budget"]["snr_req_db"])
    put("numOverhead", f"{100 * m3['overhead_xapp']:.1f}\\%", script=sm3, config="(1/14 sensing symbols per slot) x CPI duty cycle 0.32", seeds="n/a", aggregation="n/a", raw=m3["overhead_xapp"])

    # --- outage (s/UE-min)
    def outage(macro, value, scheme, lab, extra=""):
        put(macro, f"{value:.2f}", script=scheme[0], config=M3_CFG + f"; margin {lab}; {scheme[1]}" + extra,
            aggregation="mean over 40 evaluation jobs of the per-job outage at the service rate (s per UE-minute)", raw=value)

    a3 = ("scripts/run_m3.py -> results/M3/metrics.json", "A3 tuned on tuning seeds (24-point grid)")
    orc = ("scripts/run_m3.py -> results/M3/metrics.json", "oracle cell selection (max post-model-B power, no interruption)")
    gn = ("scripts/run_m3_genie.py -> results/M3/genie.json", "genie + A3, first-round policy, ground-truth prediction, NO sensing overhead")
    hy = ("scripts/run_m3_hybrid.py -> results/M3/hybrid.json", "hybrid A3 + xApp, jointly tuned (400-point grid)")
    xa = ("scripts/run_m3.py -> results/M3/metrics.json", "sensing xApp + A3 (A3 held off during the xApp hold), budget-2/4 detector as tuned")
    for mi, tag in ((i10, "Ten"), (iref, "Ref")):
        lab = labels[mi]
        ev = m3["evaluation"][mi]
        outage(f"numAthree{tag}", ev["a3"]["outage_req_s_per_min"]["mean"], a3, lab)
        outage(f"numOracle{tag}", ev["oracle"]["outage_req_s_per_min"]["mean"], orc, lab)
        outage(f"numGenie{tag}", gen["margins"][mi]["tau_ho"][tau]["genie_no_overhead"]["outage_req_s_per_min"]["mean"], gn, lab)
        outage(f"numHybrid{tag}", hyb["margins"][mi]["evaluation"]["hybrid_joint"]["outage_req_s_per_min"]["mean"], hy, lab)
    outage("numXappTen", m3["evaluation"][i10]["xapp"]["outage_req_s_per_min"]["mean"], xa, labels[i10])

    # --- handover matching at 10 dB
    g10 = gen["margins"][i10]["tau_ho"][tau]
    match_def = "; a handover matches when it leaves the fixed cell within [event start - 2 s, event end] of a 10 dB event of that UE"
    put("numGenieMatch", _pct(g10["genie_no_overhead"]["precision"]["mean"]), script=gn[0], config=M3_CFG + "; margin 10 dB; " + gn[1] + match_def,
        aggregation="mean over evaluation jobs of the per-job share of genie-triggered handovers that match", raw=g10["genie_no_overhead"]["precision"]["mean"])
    put("numAthreeMatch", _pct(m3["evaluation"][i10]["a3"]["precision"]["mean"]), script=a3[0], config=M3_CFG + "; margin 10 dB; " + a3[1] + match_def,
        aggregation="mean over evaluation jobs of the per-job share of A3 handovers that match", raw=m3["evaluation"][i10]["a3"]["precision"]["mean"])
    pre = "; bus/truck 10 dB events with a matching handover before event start (proactive recall, all events)"
    put("numGeniePreBus", _pct(g10["genie_no_overhead"]["events"]["bus/truck|all"]["proactive_recall"]["mean"]), script=gn[0], config=M3_CFG + "; margin 10 dB; " + gn[1] + pre,
        aggregation="mean over evaluation jobs with bus/truck events of the per-job share", raw=g10["genie_no_overhead"]["events"]["bus/truck|all"]["proactive_recall"]["mean"])
    put("numAthreePreBus", _pct(m3["evaluation"][i10]["a3"]["events"]["bus/truck|all"]["proactive_recall"]["mean"]), script=a3[0], config=M3_CFG + "; margin 10 dB; " + a3[1] + pre,
        aggregation="mean over evaluation jobs with bus/truck events of the per-job share", raw=m3["evaluation"][i10]["a3"]["events"]["bus/truck|all"]["proactive_recall"]["mean"])

    # --- causal foresight bound (run_m5_foresight.py)
    sf = "scripts/run_m5_foresight.py -> results/M5/foresight.json"
    fcfg = M3_CFG + "; tuned A3 (24-point grid), tau_HO 20 ms; causal foresight = A3's cell unusable, other cell usable, A3's cell usable with its unblocked power"
    rng_rows = [r for r in fs["margins"] if r["label"] in ("5 dB", "10 dB", "15 dB", "20 dB", "25 dB", "30 dB")]
    shares = [r["new_blockage_caused"]["share_of_a3_outage_pooled"] for r in rng_rows]
    put("numForesightRange", _rng(shares, "{:.0f}", pct=True), script=sf, config=fcfg + "; margins 5-30 dB",
        aggregation="POOLED share of A3 outage over evaluation jobs (sum of steps / sum of A3 outage steps); min-max over margins 5-30 dB", raw={r["label"]: r["new_blockage_caused"]["share_of_a3_outage_pooled"] for r in rng_rows})
    ref = next(r for r in fs["margins"] if r["label"] == "3GPP short-range reference")
    put("numForesightRef", _pct(ref["new_blockage_caused"]["share_of_a3_outage_pooled"]), script=sf, config=fcfg + "; 3GPP reference margin",
        aggregation="POOLED share of A3 outage over evaluation jobs", raw=ref["new_blockage_caused"]["share_of_a3_outage_pooled"])
    absvals = {r["label"]: r["new_blockage_caused"]["s_per_ue_min"]["mean"] for r in fs["margins"]}
    put("numForesightMaxAbs", f"{max(absvals.values()):.2f}", script=sf, config=fcfg + "; all margins",
        aggregation="max over margins of the mean over evaluation jobs of blockage-caused wrong-cell time [s/UE-min]", raw=absvals)

    # --- tau_HO = 0 gap closure (run_m3_genie.py)
    def closed(r):
        gc = r["gap_closure"]
        return 1.0 - gc["a3_gap_tau0"] / gc["a3_gap_tau_default"]
    tcfg = M3_CFG + "; A3 retuned at tau_HO = 0 on tuning seeds; gap = A3 - oracle outage (means over evaluation jobs)"
    groups = {"numTauZeroLow": ("0 dB", "5 dB", "10 dB"), "numTauZeroHigh": ("15 dB", "20 dB", "25 dB", "30 dB")}
    for macro, labs in groups.items():
        vals = {r["label"]: closed(r) for r in gen["margins"] if r["label"] in labs}
        put(macro, _rng(list(vals.values()), "{:.0f}", pct=True), script=gn[0], config=tcfg + f"; margins {labs[0]}-{labs[-1]}",
            aggregation="share of the A3-oracle gap closed, 1 - gap(0)/gap(20 ms), from means over jobs; min-max over margins", raw=vals)
    rref = next(r for r in gen["margins"] if r["label"] == "3GPP short-range reference")
    put("numTauZeroRef", _pct(closed(rref)), script=gn[0], config=tcfg + "; 3GPP reference margin", aggregation="1 - gap(0)/gap(20 ms), from means over jobs", raw=closed(rref))
    return out


def macros_in_main() -> list[tuple[str, str]]:
    """Macro names listed in the header comment of paper/main.tex (before \\documentclass), in order."""
    main = PAPER / "main.tex"
    if not main.exists():
        return []
    names: list[str] = []
    for line in main.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("\\documentclass"):
            break
        if line.lstrip().startswith("%"):
            for name in MACRO_NAME.findall(line):
                if name not in names:
                    names.append(name)
    return [(name, "") for name in names]


def write_numbers(cat: dict[str, dict[str, str]]) -> dict[str, Any]:
    wanted = macros_in_main()
    lines = ["% Generated by scripts/make_paper.py from results/. Do not edit."]
    defined, unmapped, missing = [], [], []
    extra = sorted(set(cat) - {n for n, _ in wanted})
    for name, desc in wanted:
        key = MACRO_MAP.get(name, name)
        if key is None:
            unmapped.append((name, desc))
            continue
        if key not in cat:
            missing.append((name, key))
            continue
        lines.append(f"\\newcommand{{\\{name}}}{{{cat[key]['value']}}}")
        defined.append(name)
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"main_tex_found": (PAPER / "main.tex").exists(), "macros_in_main": len(wanted), "defined": defined, "unmapped": unmapped,
            "mapped_key_missing": missing, "catalog_not_in_main": extra}


def compile_paper() -> None:
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    if not (PAPER / "main.tex").exists():
        print("paper/main.tex not found; skipping compilation")
        return
    if shutil.which("latexmk") is None:
        print("latexmk not available here; compile with:  cd paper && " + " ".join(cmd))
        return
    subprocess.run(cmd, cwd=PAPER, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtime-recompute", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    if not args.skip_figures:
        run_scripts(args.leadtime_recompute)
    cat = catalog()
    (RESULTS / "M5").mkdir(parents=True, exist_ok=True)
    (RESULTS / "M5" / "numbers_catalog.json").write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    status = write_numbers(cat)
    print(f"catalog: {len(cat)} numbers -> results/M5/numbers_catalog.json")
    if not status["main_tex_found"]:
        print("paper/main.tex not found: numbers.tex written without macros; no macro list to check")
    else:
        print(f"main.tex lists {status['macros_in_main']} macros; numbers.tex defines {len(status['defined'])}")
        for name, desc in status["unmapped"]:
            print(f"  NOT DEFINED (no mapping): \\{name}  -- {desc}")
        for name, key in status["mapped_key_missing"]:
            print(f"  NOT DEFINED (no catalog entry '{key}'): \\{name}")
        for name in status["catalog_not_in_main"]:
            print(f"  catalog entry not listed in main.tex: {name}")
    compile_paper()


if __name__ == "__main__":
    main()
