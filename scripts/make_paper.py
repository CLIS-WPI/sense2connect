"""Regenerate every paper figure, the table and paper/numbers.tex, then compile the paper.

Steps:
1. Run the figure/table scripts (each one command, from saved results):
   fig_blockage_table.py, fig_leadtime.py, fig_onset.py,
   fig_outage_margin.py, fig_headroom.py.
2. Build the number catalog from results (results/M5/numbers_catalog.json):
   every value the paper may quote, under a descriptive key, as a plain
   string with its precision fixed here.
3. Collect the macros paper/main.tex expects (names in its header comment,
   its \\providecommand defaults, and \\num macros used in the text), map
   each to a catalog key (MACRO_MAP, default: same name)
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

FIGURE_SCRIPTS = ["fig_blockage_table.py", "fig_leadtime.py", "fig_onset.py", "fig_outage_margin.py", "fig_value.py"]

# Macros added after the planner round that are not (yet) in the main.tex header list;
# numbers.tex defines them as well, and missing ones are reported.
EXTRA_MACROS = [
    "numAfiveTen", "numAfiveRef", "numCostOracleTen", "numCostOracleRef", "numGeniePlanTen", "numGeniePlanRef",
    "numSensePlanTen", "numSensePlanRef", "numTrueLossRef", "numTrueLossFrom", "numAthreeWideRef", "numValueRange", "numValuePedRange",
    "numValueBusRange", "numValuePedTen", "numValueBusTen", "numPlanHorizon", "numSensePlanHO", "numSensePlanPP", "numAfiveHO",
]

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
    # --- extra (not in main.tex): A3-oracle gap shares and the cost-aware oracle
    for r in fs["margins"]:
        tag = {"3GPP short-range reference": "ref", "v1 radio (high margin)": "v1"}.get(r["label"], r["label"].replace(" dB", "db"))
        gs = r["a3_oracle_gap_pooled"]
        for key in ("foresight_recoverable_share", "interruption_share", "distance_caused_share"):
            if gs[key] is not None:
                put(f"x.gap.{tag}.{key}", _pct(gs[key]), script=sf, config=fcfg + f"; margin {r['label']}",
                    aggregation="POOLED over evaluation jobs: steps / (A3 outage steps - oracle outage steps)", raw=gs[key])
    dpo = R5 / "dporacle.json"
    if dpo.exists():
        dp = json.loads(dpo.read_text())
        sd = "scripts/run_m5_dporacle.py -> results/M5/dporacle.json"
        for r in dp["margins"]:
            tag = {"3GPP short-range reference": "ref", "v1 radio (high margin)": "v1"}.get(r["label"], r["label"].replace(" dB", "db"))
            for t in ("0.020", "0.000"):
                blk = r[t]
                ttag = "tau20" if t == "0.020" else "tau0"
                for v in ("costaware_any", "costaware_epoch"):
                    put(f"x.{v}.{tag}.{ttag}.outage", f"{blk[v]['mean']:.2f}", script=sd,
                        config=M3_CFG + f"; margin {r['label']}; tau_HO {float(t) * 1e3:.0f} ms; Viterbi over the cell with perfect knowledge of both cells' blocked SNR, switch = tau_HO outage, "
                        + ("switches at any 10 ms step" if v == "costaware_any" else "switches only at 0.1 s epochs"),
                        aggregation="mean over 40 evaluation jobs [s/UE-min]", raw=blk[v]["mean"])
                    if blk[f"{v}_gap_closed_pooled"] is not None:
                        put(f"x.{v}.{tag}.{ttag}.gap_closed", _pct(blk[f"{v}_gap_closed_pooled"]), script=sd, config=M3_CFG + f"; margin {r['label']}; tau_HO {float(t) * 1e3:.0f} ms",
                            aggregation="POOLED: (A3 - cost-aware) / (A3 - instantaneous oracle) outage steps", raw=blk[f"{v}_gap_closed_pooled"])
    _planner_macros(put, R5, M3_CFG, EVAL)
    rref = next(r for r in gen["margins"] if r["label"] == "3GPP short-range reference")
    put("numTauZeroRef", _pct(closed(rref)), script=gn[0], config=tcfg + "; 3GPP reference margin", aggregation="1 - gap(0)/gap(20 ms), from means over jobs", raw=closed(rref))
    return out


def _planner_macros(put, R5: Path, M3_CFG: str, EVAL: str) -> None:
    """Macros of the planner round (added after the cost-aware-oracle result)."""
    pf = R5 / "planner.json"
    if not pf.exists():
        return
    pl = json.loads(pf.read_text())
    dp = json.loads((R5 / "dporacle.json").read_text())
    rows = {r["label"]: r for r in pl["margins"]}
    dpr = {r["label"]: r for r in dp["margins"]}
    sp = "scripts/run_m5_planner.py -> results/M5/planner.json"
    sd = "scripts/run_m5_dporacle.py -> results/M5/dporacle.json"
    cfg = M3_CFG + "; ADDED AFTER THE COST-AWARE-ORACLE RESULT"
    mean_agg = "mean over 40 evaluation jobs [s/UE-min]"
    tag_lab = {"Ten": "10 dB", "Ref": "3GPP short-range reference"}
    for tag, lab in tag_lab.items():
        r = rows[lab]
        a5 = r["a5"]
        put(f"numAfive{tag}", f"{a5['eval']['outage_req_s_per_min']['mean']:.2f}", script=sp, config=cfg + f"; margin {lab}; A5 tuned on tuning seeds (t1/t2/TTT = {a5['params']})",
            aggregation=mean_agg, raw=a5["eval"]["outage_req_s_per_min"]["mean"])
        ca = dpr[lab]["0.020"]["costaware_epoch"]
        put(f"numCostOracle{tag}", f"{ca['mean']:.2f}", script=sd, config=cfg + f"; margin {lab}; cost-aware oracle, switches at 0.1 s epochs, tau_HO 20 ms",
            aggregation=mean_agg, raw=ca["mean"])
        g = r["genie_planner"]["0.5"]["eval"]["outage_req_s_per_min"]["mean"]
        put(f"numGeniePlan{tag}", f"{g:.2f}", script=sp, config=cfg + f"; margin {lab}; genie-planner, H = 0.5 s, true future SNR, no overhead",
            aggregation=mean_agg, raw=g)
        h = f"{r['sensing_planner_tuned_H']:.1f}"
        sv = r["sensing_planner"][h]
        put(f"numSensePlan{tag}", f"{sv['eval']['outage_req_s_per_min']['mean']:.2f}", script=sp,
            config=cfg + f"; margin {lab}; sensing-planner, H = {h} s and detector budget {sv['budget']} tuned on tuning seeds, overhead charged",
            aggregation=mean_agg, raw=sv["eval"]["outage_req_s_per_min"]["mean"])
    rref = rows["3GPP short-range reference"]
    h = f"{rref['sensing_planner_tuned_H']:.1f}"
    tl = rref["diag_trueloss_planner"][h]["eval"]["outage_req_s_per_min"]["mean"]
    put("numTrueLossRef", f"{tl:.2f}", script=sp, config=cfg + f"; 3GPP reference; diagnostic planner with the TRUE LoS loss (unblocked SNR minus true LoS loss), overhead charged, H = {h} s (the sensing-planner's tuned H at this margin)",
        aggregation=mean_agg, raw=tl)
    beats = {}
    for lab, r in rows.items():
        hh = f"{r['sensing_planner_tuned_H']:.1f}"
        beats[lab] = (r["diag_trueloss_planner"][hh]["eval"]["outage_req_s_per_min"]["mean"], r["a5"]["eval"]["outage_req_s_per_min"]["mean"], r["margin_db"])
    winning = sorted(v[2] for v in beats.values() if v[0] < v[1])
    first = winning[0] if winning else None
    above_all = first is not None and all(v[0] < v[1] for v in beats.values() if v[2] >= first)
    overlaps = []
    for lab, r in rows.items():
        if first is not None and r["margin_db"] >= first:
            hh = f"{r['sensing_planner_tuned_H']:.1f}"
            t_, a_ = r["diag_trueloss_planner"][hh]["eval"]["outage_req_s_per_min"], r["a5"]["eval"]["outage_req_s_per_min"]
            overlaps.append(not (t_["mean"] + t_["ci"] < a_["mean"] - a_["ci"] or a_["mean"] + a_["ci"] < t_["mean"] - t_["ci"]))
    ci_note = ("; 95 % CIs overlap at every such margin (not significant, unpaired)" if overlaps and all(overlaps)
               else "; 95 % CIs separate at some such margins" if overlaps else "")
    put("numTrueLossFrom", "--" if first is None else f"{first:.0f}", script=sp,
        config=cfg + "; true-LoS-loss planner (sensing-planner's tuned H per margin) vs A5, comparison of means",
        aggregation="lowest margin [dB] where the mean outage is below A5's", raw={k: {"trueloss": v[0], "a5": v[1]} for k, v in beats.items()},
        note=("lower mean than A5 at every margin from this one up" if above_all else "does NOT have a lower mean at every higher margin") + ci_note)
    sel = [r for lab, r in rows.items() if lab not in ("0 dB", "v1 radio (high margin)")]  # 5-30 dB and the 3GPP reference
    val = [r["value_of_foresight"]["share_of_best_reactive_gap_pooled"] for r in sel]
    ped = [r["value_of_foresight"]["by_class_share_of_gap_pooled"]["pedestrian"] for r in sel]
    bus = [r["value_of_foresight"]["by_class_share_of_gap_pooled"]["bus/truck"] for r in sel]
    vagg = "POOLED over evaluation jobs: (best reactive - cost-aware) / (best reactive - instantaneous oracle) outage steps"
    vcfg = cfg + "; best reactive = A5 (all margins); cost-aware oracle at 0.1 s epochs, tau_HO 20 ms"
    put("numValueRange", _rng(val, "{:.0f}", pct=True), script=sp, config=vcfg + "; margins 5-30 dB and the 3GPP reference", aggregation=vagg + "; min-max over margins",
        raw={r["label"]: v for r, v in zip(sel, val)})
    cagg = "net closed steps by the dominant LoS blocker class of A5's cell, as a share of the best-reactive-to-oracle gap (pooled); min-max over margins"
    put("numValuePedRange", _rng(ped, "{:.0f}", pct=True), script=sp, config=vcfg + "; pedestrian; margins 5-30 dB and the 3GPP reference", aggregation=cagg,
        raw={r["label"]: v for r, v in zip(sel, ped)})
    put("numValueBusRange", _rng(bus, "{:.0f}", pct=True), script=sp, config=vcfg + "; bus/truck; margins 5-30 dB and the 3GPP reference", aggregation=cagg,
        raw={r["label"]: v for r, v in zip(sel, bus)})
    r10 = rows["10 dB"]
    put("numValuePedTen", _pct(r10["value_of_foresight"]["by_class_share_of_gap_pooled"]["pedestrian"]), script=sp, config=vcfg + "; pedestrian; margin 10 dB",
        aggregation="net closed steps (pedestrian) / best-reactive-to-oracle gap, pooled", raw=r10["value_of_foresight"]["by_class_share_of_gap_pooled"]["pedestrian"])
    put("numValueBusTen", _pct(r10["value_of_foresight"]["by_class_share_of_gap_pooled"]["bus/truck"]), script=sp, config=vcfg + "; bus/truck; margin 10 dB",
        aggregation="net closed steps (bus/truck) / best-reactive-to-oracle gap, pooled", raw=r10["value_of_foresight"]["by_class_share_of_gap_pooled"]["bus/truck"])
    rel = {}
    for hs in ("0.5", "1.0", "2.0", "3.0"):
        rel[hs] = {lab: (r["genie_planner"][hs]["eval"]["outage_req_s_per_min"]["mean"] - dpr[lab]["0.020"]["costaware_epoch"]["mean"]) / dpr[lab]["0.020"]["costaware_epoch"]["mean"]
                   for lab, r in rows.items()}
    absx = {hs: {lab: r["genie_planner"][hs]["eval"]["outage_req_s_per_min"]["mean"] - dpr[lab]["0.020"]["costaware_epoch"]["mean"] for lab, r in rows.items()}
            for hs in ("0.5", "1.0", "2.0", "3.0")}
    ok = [hs for hs in ("0.5", "1.0", "2.0", "3.0") if all(rel[hs][lab] <= 0.02 or absx[hs][lab] <= 0.01 for lab in rows)]
    put("numPlanHorizon", ok[0] if ok else "--", script=sp, config=cfg + "; genie-planner vs cost-aware oracle (0.1 s epochs, tau_HO 20 ms), all margins incl. the v1-radio point",
        aggregation="RULE: smallest H [s] such that at every margin the genie-planner's mean outage exceeds the cost-aware oracle's by at most 2 % (relative) OR at most 0.01 s/UE-min (absolute)",
        raw={"relative_excess": rel, "absolute_excess_s_per_ue_min": absx},
        note=("" if ok else "no H meets the rule at every margin"))
    rw_ = rows["3GPP short-range reference"]["a3_wide"]
    put("numAthreeWideRef", f"{rw_['eval']['outage_req_s_per_min']['mean']:.2f}", script=sp,
        config=cfg + f"; 3GPP reference; A3 on the equal-effort wide grid (offset/hysteresis up to 15 dB, TTT 40-640 ms), tuned on tuning seeds: {rw_['params']}; the A3 curve in fig_outage_margin.pdf",
        aggregation=mean_agg, raw=rw_["eval"]["outage_req_s_per_min"]["mean"])
    for lab, r in rows.items():
        if r["margin_db"] < 15.0:
            continue
        hh = f"{r['sensing_planner_tuned_H']:.1f}"
        t_, a_ = r["diag_trueloss_planner"][hh]["eval"]["outage_req_s_per_min"], r["a5"]["eval"]["outage_req_s_per_min"]
        overlap = not (t_["mean"] + t_["ci"] < a_["mean"] - a_["ci"] or a_["mean"] + a_["ci"] < t_["mean"] - t_["ci"])
        tag = {"3GPP short-range reference": "ref", "v1 radio (high margin)": "v1"}.get(lab, lab.replace(" dB", "db"))
        put(f"x.trueloss_vs_a5.{tag}.ci_overlap", "yes" if overlap else "no", script=sp,
            config=cfg + f"; margin {lab}; true-LoS-loss planner (H = {hh} s) vs A5; 95 % CI = 1.96 sd / sqrt(n) over 40 evaluation jobs",
            aggregation="do the two 95 % CIs of the mean outage overlap", raw={"trueloss": t_, "a5": a_})
    sv10 = r10["sensing_planner"][f"{r10['sensing_planner_tuned_H']:.1f}"]["eval"]
    put("numSensePlanHO", f"{sv10['ho_per_min']['mean']:.1f}", script=sp, config=cfg + "; margin 10 dB; sensing-planner (tuned H)",
        aggregation="mean over evaluation jobs of handovers per UE-minute", raw=sv10["ho_per_min"]["mean"], note="margin not specified in the request; 10 dB used")
    put("numSensePlanPP", f"{sv10['ping_pong']['mean']:.2f}", script=sp, config=cfg + "; margin 10 dB; sensing-planner (tuned H); ping-pong = share of handovers back to the previous cell within 1 s",
        aggregation="mean over evaluation jobs of the per-job ping-pong rate", raw=sv10["ping_pong"]["mean"], note="margin not specified in the request; 10 dB used")
    put("numAfiveHO", f"{r10['a5']['eval']['ho_per_min']['mean']:.1f}", script=sp, config=cfg + "; margin 10 dB; A5 (tuned)",
        aggregation="mean over evaluation jobs of handovers per UE-minute", raw=r10["a5"]["eval"]["ho_per_min"]["mean"], note="margin not specified in the request; 10 dB used")


def macros_in_main() -> list[tuple[str, str]]:
    """Macros paper/main.tex expects, in order: names listed in the header comment
    (before \\documentclass), \\providecommand defaults, and \\num macros used in the text."""
    main = PAPER / "main.tex"
    if not main.exists():
        return []
    text = main.read_text(encoding="utf-8")
    names: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("\\documentclass"):
            break
        if line.lstrip().startswith("%"):
            for name in MACRO_NAME.findall(line):
                if name not in names:
                    names.append(name)
    for name in re.findall(r"\\providecommand\{\\(num[A-Za-z]+)\}", text):
        if name not in names:
            names.append(name)
    for name in macros_used_in_main():
        if name not in names:
            names.append(name)
    return [(name, "") for name in names]


def write_numbers(cat: dict[str, dict[str, str]]) -> dict[str, Any]:
    wanted = macros_in_main()
    wanted += [(m, "extra") for m in EXTRA_MACROS if m not in {n for n, _ in wanted}]
    # keep every catalogued macro defined even if the current text does not use it
    wanted += [(m, "catalog") for m in cat if m.startswith("num") and m not in {n for n, _ in wanted}]
    lines = ["% Generated by scripts/make_paper.py from results/. Do not edit."]
    defined, unmapped, missing = [], [], []
    extra = sorted(k for k in set(cat) - {n for n, _ in wanted} if not k.startswith("x."))
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
        print(f"numbers.tex defines {len(status['defined'])} macros (everything main.tex declares or uses, the extras and every catalogued macro)")
        for name, desc in status["unmapped"]:
            print(f"  NOT DEFINED (no mapping): \\{name}  -- {desc}")
        for name, key in status["mapped_key_missing"]:
            print(f"  NOT DEFINED (no catalog entry '{key}'): \\{name}")
        unused = [n for n in status["defined"] if n not in set(macros_used_in_main())]
        if unused:
            print(f"  defined but not used in main.tex ({len(unused)}): {', '.join(unused)}")
    used = macros_used_in_main()
    defined_now = set(re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", (PAPER / "numbers.tex").read_text(encoding="utf-8")))
    undefined = sorted(set(used) - defined_now)
    print(f"main.tex uses {len(set(used))} \\num macros; not defined by numbers.tex: {', '.join(undefined) if undefined else 'none'}")
    compile_paper()


def macros_used_in_main() -> list[str]:
    """Every \\num... macro used anywhere in paper/main.tex outside comments and \\providecommand defaults."""
    main = PAPER / "main.tex"
    if not main.exists():
        return []
    names = []
    for line in main.read_text(encoding="utf-8").splitlines():
        code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        if "\\providecommand" in code:
            continue
        names += MACRO_NAME.findall(code)
    return names


if __name__ == "__main__":
    main()
