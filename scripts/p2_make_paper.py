"""Paper 2 numbers (P2-M5): results/P2/numbers_catalog.json and paper2/numbers2.tex.

Every number quoted by paper2/main.tex is a macro defined here from the result
files of the HELD-OUT evaluation (S2C_EVAL_SET=heldout; --set dev for a
development preview), catalogued with source script, configuration, seeds and
aggregation. Also checks that every \\p... macro used in paper2/main.tex is
defined and compiles the paper with the local Docker image texlive/texlive
when --compile is given (run on the host, not in the sionna container).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
P2 = ROOT / "results" / "P2"
PAPER2 = ROOT / "paper2"
VARIANT = "v1"
MACRO = re.compile(r"\\(p[A-Z][A-Za-z]*)")


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def pfmt(p: float) -> str:
    """p-value rounded UP to two significant digits."""
    if not math.isfinite(p):
        return "--"
    e = math.floor(math.log10(p))
    v = math.ceil(p * 10 ** (1 - e)) / 10 ** (1 - e)
    return f"{v:.2g}"


def catalog(tag: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    seeds = {"heldout": "held-out test seeds 2001-2010 (configs/seeds_heldout.yaml; pre-fix record)",
             "heldout2": "held-out test seeds 3001-3010 (configs/p2.yaml heldout2; after the estimator fix, tag p2-freeze2)"}.get(tag, "development seeds 1001-1010")

    def put(name, value, *, script, config, aggregation, raw=None):
        out[name] = {"value": value, "script": script, "config": config, "seeds": seeds + ", 40 runs (2 mounts x 2 densities), 2 UEs, 600 epochs",
                     "aggregation": aggregation, "raw": raw}

    peb = json.loads((P2 / f"peb_{tag}.json").read_text())
    est = json.loads((P2 / f"est_{tag}{'' if VARIANT == 'v1' else '_' + VARIANT}.json").read_text())
    elabel = (lambda n: f"estimator {n}") if VARIANT == "v1" else (lambda n: f"estimator {VARIANT} {n}")
    sp = "scripts/p2_peb.py + scripts/p2_peb_report.py -> results/P2/peb_%s.json" % tag
    se = "scripts/p2_estimate.py + scripts/p2_est_report.py -> results/P2/est_%s.json" % tag
    main = "main configuration: TDoA (unknown UE clock bias), sigma_sync 1 ns constant per run, sigma_phi 2 deg, blocked LoS biased"
    seedagg = "per-seed value over its runs, UEs and epochs; mean over the 10 seeds"
    put("pNumSeeds", str(len(peb["seeds"])), script=sp, config="", aggregation="number of seeds")
    put("pNumRuns", str(peb["n_runs"]), script=sp, config="", aggregation="number of runs")
    rows = {(r["info"], r["bw"], r["state"]): r for r in peb["shares_main_hw"]}
    for info, I in (("los", "Los"), ("map", "Map")):
        for bw, B in (("100", "Hundred"), ("200", "TwoHundred"), ("400", "FourHundred")):
            for state, S in (("LoS both", "Los"), ("blocked", "Blk"), ("all", "All")):
                r = rows[(info, bw, state)]
                put(f"pDec{I}{B}{S}", pct(r["share_le_0.1"]), script=sp, config=f"{main}; {info} information; {bw} MHz; epochs: {state}",
                    aggregation="share of epochs with PEB <= 0.1 m; " + seedagg, raw=r)
                put(f"pPeb{I}{B}{S}", f"{100 * r['median_peb_m']:.1f}", script=sp, config=f"{main}; {info}; {bw} MHz; {state}",
                    aggregation="median PEB [cm]: median over seeds of the per-seed median", raw=r)
    p1 = peb["P1"]
    put("pVerdictOneLos", p1["los"]["verdict"], script=sp, config="P1, LoS-only information", aggregation="pre-specified rule (scripts/p2_peb_report.py)", raw=p1["los"])
    put("pVerdictOneMap", p1["map"]["verdict"], script=sp, config="P1, map-aided information", aggregation="pre-specified rule", raw=p1["map"])
    lim = peb["limit_analysis_200MHz"]
    for info, I in (("los", "Los"), ("map", "Map")):
        for key, K in (("bandwidth 200 -> 400 MHz", "Bw"), ("SNR x10", "Snr"), ("sigma_sync -> 0", "Sync"), ("sigma_phi -> 0", "Cal")):
            v = lim[info][key]
            put(f"pLim{I}{K}", f"{v['ratio_to_base']:.2f}", script=sp, config=f"200 MHz, {main}; {info}; relax: {key}",
                aggregation="median over seeds of (per-seed median PEB relaxed / base); Wilcoxon p in raw", raw=v)
        put(f"pLim{I}Base", f"{100 * lim[info]['base_median_m']:.1f}", script=sp, config=f"200 MHz, {main}; {info}", aggregation="median PEB [cm]")
    hw = {(h["info"], h["timing"], h["sync_ns"], h["phi_deg"]): h for h in peb["hardware_sweep_400MHz_biased"]}
    for info, I in (("los", "Los"), ("map", "Map")):
        for (sy, ph), T in (((0.0, 0.0), "Ideal"), ((3.0, 5.0), "Worst"), ((0.0, 5.0), "CalFive"), ((3.0, 0.0), "SyncThree")):
            h = hw[(info, "tdoa", sy, ph)]
            put(f"pHw{I}{T}", pct(h["share_le_0.1"]), script=sp, config=f"400 MHz, TDoA, sigma_sync {sy} ns, sigma_phi {ph} deg, blocked LoS biased; {info}",
                aggregation="share of epochs with PEB <= 0.1 m; " + seedagg, raw=h)
    for key, K in (("los|biased|400", "LosB"), ("los|kept|400", "LosA"), ("map|biased|400", "MapB"), ("map|kept|400", "MapA")):
        v = peb["P2"][key]
        put(f"pRatio{K}", f"{v['median_ratio']:.1f}", script=sp, config=f"400 MHz, {main}; info|variant = {key}",
            aggregation="median over seeds of median PEB (blocked epochs) / median PEB (LoS-to-both epochs)", raw=v)
        put(f"pRatioP{K}", pfmt(v["p"]), script=sp, config=key, aggregation="exact Wilcoxon over 10 seeds (log medians)")
    put("pVerdictTwo", peb["P2"]["los|biased|400"]["verdict"], script=sp, config="P2 main: LoS-only, biased, 400 MHz", aggregation="pre-specified rule")
    put("pVerdictTwoKept", peb["P2"]["los|kept|400"]["verdict"], script=sp, config="P2, variant a", aggregation="pre-specified rule")
    v3 = peb["P3"]["biased|400"]
    put("pMapToLosBlk", f"{v3['median_ratio_map_to_los']:.2f}", script=sp, config="400 MHz, main, blocked epochs", aggregation="median over seeds of median PEB map / LoS-only", raw=v3)
    put("pMapToLosP", pfmt(v3["p"]), script=sp, config="P3", aggregation="exact Wilcoxon over seeds")
    put("pVerdictThree", v3["verdict"], script=sp, config="P3 main (biased)", aggregation="pre-specified rule")
    put("pVerdictThreeKept", peb["P3"]["kept|400"]["verdict"], script=sp, config="P3 (kept)", aggregation="pre-specified rule")
    c = est["configs"]
    for name, K in (("bw400_tdoa_s1_p2_b", "FourHundred"), ("bw200_tdoa_s1_p2_b", "TwoHundred"), ("bw100_tdoa_s1_p2_b", "Hundred"),
                    ("bw400_toa_s1_p2_b", "Toa"), ("bw400_aoa_s1_p2_b", "Aoa"), ("bw400_tdoa_s0_p0_b", "IdealHw"), ("bw400_tdoa_s3_p5_b", "WorstHw"),
                    ("bw400_tdoa_s1_p2_a", "Kept"), ("bw400_tdoa_s1e_p2_b", "SyncEpoch")):
        if name not in c:
            continue
        r = c[name]
        put(f"pEstMed{K}", f"{100 * r['median_m']:.0f}", script=se, config=name, aggregation="median EKF error [cm]; " + seedagg, raw={k: r[k] for k in r if k != "cfg"})
        put(f"pEstPninety{K}", f"{r['p90_m']:.2f}", script=se, config=name, aggregation="90th percentile EKF error [m]; " + seedagg)
        put(f"pEstLos{K}", f"{100 * r['median_los_m']:.0f}", script=se, config=name + "; LoS-to-both epochs", aggregation="median EKF error [cm]; " + seedagg)
        put(f"pEstBlk{K}", f"{100 * r['median_blocked_m']:.0f}", script=se, config=name + "; blocked epochs", aggregation="median EKF error [cm]; " + seedagg)
        put(f"pEstToPeb{K}", f"{r['err_to_peb_median']:.0f}", script=se, config=name, aggregation="median over seeds of the per-seed median error/PEB (LoS-only bound, same configuration)")
        put(f"pEstDec{K}", pct(r["share_le_0.1"]), script=se, config=name, aggregation="share of epochs with error <= 0.1 m; " + seedagg)
    if "P4" in est:
        put("pVerdictFour", est["P4"], script=se, config="P4 (parts 1 and 2, pre-specified)", aggregation="rule in scripts/p2_est_report.py",
            raw={"part1": est["P4_part1"], "part2": est["P4_part2"]})
    cl = est.get("closing", {})
    for cname, K in ((elabel("bw400_tdoa_s1_p2_b"), "Est"), ("PEB los-only", "PebLos"), ("PEB map-aided", "PebMap"), ("perfect", "Perf"), ("white 1 m", "White")):
        if cname not in cl:
            continue
        for lab, M in (("15 dB", "Fifteen"), ("20 dB", "Twenty"), ("3GPP short-range reference", "Ref")):
            v = cl[cname][lab]
            put(f"pClose{K}{M}", f"{v['mean_diff']:+.3f}" + ("$^\\dagger$" if not v["wilcoxon_p_two_sided"] < 0.05 else ""),
                script="scripts/p2_closing.py -> results/P2/closing_%s.json" % tag, config=f"paper-1 planner (v1.4.1, perfect blocker tracks, H fixed) with UE position '{cname}'; margin {lab}",
                aggregation="planner - A5 outage [s/UE-min], mean over seeds of per-seed means; dagger = exact Wilcoxon p >= 0.05 (pointwise, exploratory)", raw=v)
    # --- added after the freeze (numbers only, no analysis change): configuration constants and further result macros
    def seed_range(which: str) -> str:
        """First--last seed of a seed set from the configs actually used (tuning, development, dev, heldout, heldout2)."""
        from sim.scenes.config import load_yaml as _ly

        sets = {"tuning": _ly(ROOT / "configs" / "seeds.yaml")["tuning"], "development": _ly(ROOT / "configs" / "seeds.yaml")["evaluation"],
                "dev": _ly(ROOT / "configs" / "seeds.yaml")["evaluation"], "heldout": _ly(ROOT / "configs" / "seeds_heldout.yaml")["heldout"],
                "heldout2": _ly(ROOT / "configs" / "p2.yaml")["heldout2"]}[which]
        return f"{min(sets)}--{max(sets)}"

    from sim.scenes.config import load_yaml

    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    m2 = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    sc = "configs/p2.yaml, configs/m2_scenario.yaml"
    const = {
        "pCarrier": f"{float(m2['carrier_hz']) / 1e9:g}", "pTxPower": f"{float(p2cfg['link']['ue_tx_power_dbm']):g}", "pNoiseFig": f"{float(p2cfg['link']['oru_noise_figure_db']):g}",
        "pScs": f"{15 * 2 ** int(p2cfg['numerology']):g}", "pNumerology": str(int(p2cfg["numerology"])),
        "pBwA": "100", "pBwB": "200", "pBwC": "400",
        "pNscA": str(p2cfg["bandwidths"]["100"]["n_sc"]), "pNscB": str(p2cfg["bandwidths"]["200"]["n_sc"]), "pNscC": str(p2cfg["bandwidths"]["400"]["n_sc"]),
        "pArray": "\\ensuremath{" + f"{m2['array']['oru']['num_rows']}\\times{m2['array']['oru']['num_cols']}" + "}", "pNumEl": str(int(m2["array"]["oru"]["num_rows"]) * int(m2["array"]["oru"]["num_cols"])),
        "pEpoch": "0.1", "pBlockedDb": f"{float(p2cfg['blockage']['blocked_los_db']):g}", "pDecThr": f"{100 * float(p2cfg['thresholds_m']['decimeter']):g}",
        "pBreakEven": f"{100 * p2cfg['thresholds_m']['paper1_break_even'][0]:g}--{100 * p2cfg['thresholds_m']['paper1_break_even'][1]:g}",
        "pSyncList": "0, 0.3, 1 and 3", "pPhiList": "0, 2 and 5", "pSyncMain": "1", "pPhiMain": "2", "pBootN": "10{,}000",
        "pSeedsTune": seed_range("tuning"), "pSeedsDev": seed_range("development"), "pSeedsHeld": seed_range(tag), "pFdStep": "1", "pClockBias": "50", "pMaxDepth": "2", "pMaxPaths": "8",
        "pSensRange": f"{float(m2['sensing_radar']['sensing_range_m']):g}", "pRatioThr": "5", "pRatioPtwo": "2", "pUeHeight": "1.5", "pLitLow": "0.6", "pLitHigh": "0.8",
        "pSyncWorst": "3", "pPhiWorst": "5", "pMarginFifteen": "15", "pMarginTwenty": "20", "pBinM": "5", "pCi": "95",
        "pGridSize": str(int(np.prod([len(v) for v in __import__("p2_estimate").GRID.values()]))),
    }
    for k, v in const.items():
        put(k, v, script="scripts/p2_make_paper.py", config=sc, aggregation="configuration constant (added after the freeze; no analysis change)")
    be = [float(x) for x in p2cfg["thresholds_m"]["paper1_break_even"]]
    put("pBreakEvenRms", f"{round(100 * be[0] * math.sqrt(2)):d}--{round(100 * be[1] * math.sqrt(2)):d}", script="scripts/p2_make_paper.py",
        config="configs/p2.yaml thresholds_m.paper1_break_even (paper-1 per-axis sigma)",
        aggregation="RMS distance error sigma * sqrt(2) of a 2-D isotropic Gaussian with that per-axis sigma [cm], rounded", raw=be)
    put("pBreakEvenMedDist", f"{round(100 * be[0] * math.sqrt(2 * math.log(2))):d}--{round(100 * be[1] * math.sqrt(2 * math.log(2))):d}",
        script="scripts/p2_make_paper.py", config="configs/p2.yaml thresholds_m.paper1_break_even (paper-1 per-axis sigma)",
        aggregation="median distance error sigma * sqrt(2 ln 2) of a 2-D isotropic Gaussian (Rayleigh median) [cm], rounded", raw=be)
    for name, K in (("bw400_tdoa_s1_p2_b", "FourHundred"), ("bw100_tdoa_s1_p2_b", "Hundred"), ("bw400_tdoa_s0_p0_b", "IdealHw")):
        if name in c:
            put(f"pEstRmse{K}", f"{c[name]['rmse_m']:.1f}", script=se, config=name, aggregation="RMSE [m] (errors capped at 100 m); " + seedagg)
    for cname, K in ((elabel("bw400_tdoa_s1_p2_b"), "Est"), ("PEB los-only", "PebLos"), ("PEB map-aided", "PebMap"), ("perfect", "Perf"), ("white 1 m", "White"),
                     (elabel("bw400_tdoa_s0_p0_b"), "EstIdeal"), (elabel("bw400_tdoa_s1_p2_b") + " capped at p90", "EstCap")):
        if cname not in cl:
            continue
        for lab, M in (("15 dB", "Fifteen"), ("20 dB", "Twenty"), ("25 dB", "TwentyFive"), ("30 dB", "Thirty"), ("3GPP short-range reference", "Ref")):
            v = cl[cname][lab]
            put(f"pClose{K}{M}", f"{v['mean_diff']:+.3f}" + ("$^\\dagger$" if not v["wilcoxon_p_two_sided"] < 0.05 else ""),
                script="scripts/p2_closing.py -> results/P2/closing_%s.json" % tag, config=f"paper-1 planner with UE position '{cname}'; margin {lab}",
                aggregation="planner - A5 outage [s/UE-min], mean over seeds of per-seed means; dagger = exact Wilcoxon p >= 0.05 (pointwise, exploratory)", raw=v)
    of = P2 / f"diag_mirror_oracle_{tag}.json"
    if of.exists():  # ORACLE diagnostic (uses the ground truth; not an estimator)
        od = json.loads(of.read_text())
        so = "scripts/p2_diag_mirror_oracle.py -> results/P2/diag_mirror_oracle_%s.json (ORACLE diagnostic)" % tag
        ocfg = "estimator A, 400 MHz TDoA main configuration; mirror epochs replaced by the hypothesis closest to the true UE (oracle)"
        o = od["oracle"]
        for key, K in (("A", "A"), ("A_mirror_corrected_oracle", "Oracle")):
            put(f"pMirror{K}Med", f"{100 * o[key]['median_m']:.0f}", script=so, config=ocfg, aggregation="median error [cm], pooled epochs after 2 s", raw=o[key])
            put(f"pMirror{K}Pninety", f"{o[key]['p90_m']:.2f}", script=so, config=ocfg, aggregation="90th percentile error [m], pooled")
            put(f"pMirror{K}Rmse", f"{o[key]['rmse_m']:.2f}", script=so, config=ocfg, aggregation="RMSE [m], pooled")
        put("pMirrorShareEpochs", pct(o["mirror_share_of_epochs"]), script=so, config=ocfg, aggregation="share of epochs classified as mirror solutions")
        put("pMirrorAboveP", pct(o["mirror_epochs_above_A_p90_share"]), script=so, config=ocfg, aggregation="share of mirror epochs whose A error exceeds A's p90")
        put("pMirrorOfTopTen", pct(o["tail_above_p90_that_is_mirror_share"]), script=so, config=ocfg, aggregation="share of the epochs above A's p90 that are mirror solutions")
        oc = {m["label"]: m["vs_a5"] for m in od["conditions"]["estimator A bw400_tdoa_s1_p2_b mirror-corrected (oracle)"]["margins"]}
        for lab, M in (("15 dB", "Fifteen"), ("20 dB", "Twenty"), ("25 dB", "TwentyFive"), ("30 dB", "Thirty"), ("3GPP short-range reference", "Ref")):
            v = oc[lab]
            put(f"pCloseOracle{M}", f"{v['mean_diff']:+.3f}" + ("$^\\dagger$" if not v["wilcoxon_p_two_sided"] < 0.05 else ""), script=so,
                config=ocfg + f"; paper-1 planner; margin {lab}", aggregation="planner - A5 outage [s/UE-min], seed level; dagger = Wilcoxon p >= 0.05", raw=v)
    # ---- external review of the draft (diagnostics; bound-level inputs are SYNTHETIC)
    bf = P2 / f"diag_boundlevel_{tag}.json"
    if bf.exists():
        bd = json.loads(bf.read_text())
        sb = "scripts/p2_diag_boundlevel.py -> results/P2/diag_boundlevel_%s.json (SYNTHETIC bound-level positions)" % tag
        for cname, K in (("bound-level LoS-only, full covariance (synthetic)", "FullLos"), ("bound-level map-aided, full covariance (synthetic)", "FullMap")):
            bl = {m["label"]: m["vs_a5"] for m in bd["conditions"][cname]["margins"]}
            for lab, M in (("15 dB", "Fifteen"), ("20 dB", "Twenty"), ("25 dB", "TwentyFive"), ("30 dB", "Thirty"), ("3GPP short-range reference", "Ref")):
                v = bl[lab]
                put(f"pClose{K}{M}", f"{v['mean_diff']:+.3f}" + ("$^\\dagger$" if not v["wilcoxon_p_two_sided"] < 0.05 else ""), script=sb,
                    config=f"{cname}: error ~ N(0, J_p(t)^-1) of the single-epoch position EFIM, independent per epoch; main configuration; margin {lab}",
                    aggregation="planner - A5 outage [s/UE-min], seed level; dagger = Wilcoxon p >= 0.05", raw=v)
        put("pAniso", f"{bd['tails']['anisotropy_sqrt_eig_ratio_los']['median']:.1f}", script=sb, config="LoS-only, main configuration",
            aggregation="median over epochs of sqrt(largest / smallest eigenvalue) of J_p(t)^-1 (anisotropy of the bound-level error)")
    vf = P2 / f"diag_rmse_vs_peb_{tag}.json"
    if vf.exists():
        vd = json.loads(vf.read_text())["cells"]
        sv = "scripts/p2_diag_rmse_vs_peb.py -> results/P2/diag_rmse_vs_peb_%s.json" % tag
        los = [c for c in vd if c["geometry"].startswith("both LoS") and c["bw"] == "400" and c["hardware"] == "main"]
        blk = [c for c in vd if c["geometry"] == "blocked"]
        rng_ = lambda vals, f: f.format(min(vals)) + "--" + f.format(max(vals))  # noqa: E731
        put("pValNumGeom", str(len({(c['mount'], c['geometry']) for c in vd})), script=sv, config="fixed geometries", aggregation="count")
        put("pValNumDraws", str(json.loads(vf.read_text())["n_draws"]), script=sv, config="independent noise + hardware draws per cell", aggregation="count")
        put("pValSpreadToPeb", rng_([c["spread_rms_m"] / c["peb_los_m"] for c in los], "{:.1f}"), script=sv, config="both-LoS geometries, 400 MHz, main hardware",
            aggregation="min-max over geometries of (rms spread of the single-epoch fix around its mean) / single-epoch LoS-only PEB", raw=los)
        put("pValBiasLos", rng_([c["bias_m"] for c in los], "{:.2f}"), script=sv, config="both-LoS geometries, 400 MHz, main hardware",
            aggregation="min-max over geometries of the bias |mean fix - true UE| [m]")
        put("pValRmseToPeb", rng_([c["rmse_to_peb"] for c in los], "{:.0f}"), script=sv, config="both-LoS geometries, 400 MHz, main hardware",
            aggregation="min-max over geometries of RMSE / single-epoch LoS-only PEB")
        put("pValBiasBlk", rng_([c["bias_m"] for c in blk], "{:.1f}"), script=sv, config="blocked geometries, all bandwidths and hardware",
            aggregation="min-max of the bias [m]")
    # ---- review 3: closing-experiment settings and verification statistics (from configs / results)
    cfj = P2 / f"closing_{tag}.json"
    if cfj.exists():
        hs = sorted(set(float(v) for v in json.loads(cfj.read_text())["H_fixed"].values()))
        put("pPlanH", f"{hs[0]:g}" if len(hs) == 1 else f"{hs[0]:g}--{hs[-1]:g}", script="scripts/p2_closing.py -> results/P2/closing_%s.json" % tag,
            config="planner horizon per margin (perfect-track H tuned on the tuning seeds, results/M5/review_b5.json)", aggregation="min-max over margins [s]",
            raw=json.loads(cfj.read_text())["H_fixed"])
    m3 = load_yaml(ROOT / "configs" / "m3.yaml")["rework"]
    put("pLoopDelay", f"{1e3 * float(m3['e2']['loop_delay_s']):g}", script="configs/m3.yaml rework.e2.loop_delay_s", config="paper-1 planner (closing experiment)",
        aggregation="E2 control-loop delay [ms]")
    put("pTauHO", f"{1e3 * float(m3['e2']['tau_ho_s']):g}", script="configs/m3.yaml rework.e2.tau_ho_s", config="paper-1 planner and A5 (closing experiment)",
        aggregation="handover interruption [ms]")
    ovh = float(m3["sensing"]["symbols_fraction"]) * float(m3["sensing"]["duty_cycle"])
    put("pOverhead", f"{100 * ovh:.1f}\\%", script="sim/comm/linkbudget.sensing_overhead with configs/m3.yaml rework.sensing",
        config="sensing overhead charged to the planner in the closing experiment (plan and rate; A5 pays none)",
        aggregation="(sensing symbols per slot / 14) x CPI duty cycle", raw={"symbols_fraction": m3["sensing"]["symbols_fraction"], "duty_cycle": m3["sensing"]["duty_cycle"]})
    vsf = P2 / "verification_stats.json"
    if vsf.exists():
        vs = json.loads(vsf.read_text())
        sv2 = "scripts/p2_verification_stats.py -> results/P2/verification_stats.json"
        put("pDerivAgree", pct(vs["derivatives"]["share_paths"]), script=sv2, config="+/-1 cm re-trace subset (2 jobs x 20 snapshots)",
            aggregation="share of paths whose six analytic derivatives (delay, azimuth, elevation x x, y) agree with the re-trace to 1e-3 relative", raw=vs["derivatives"])
        put("pDerivAgreeValues", pct(vs["derivatives"]["share_values"]), script=sv2, config="+/-1 cm re-trace subset",
            aggregation="share of individual derivative values agreeing to 1e-3 relative")
        e_ = vs["fim"]["largest_test_deviation"]
        ex = int(math.floor(math.log10(e_)))
        put("pFimToyErr", f"\\ensuremath{{{e_ / 10 ** ex:.1f}\\times 10^{{{ex}}}}}", script=sv2,
            config="unit tests: GPU Gram vs NumPy synthesis; GPU PEB pipeline vs brute-force numerical FIM (toy case with sync and calibration priors)",
            aggregation="largest relative deviation", raw=vs["fim"])
        jk = f"job_gpu_vs_numpy_{tag}" if f"job_gpu_vs_numpy_{tag}" in vs["fim"] else "job_gpu_vs_numpy_heldout2"
        e2 = vs["fim"][jk]
        ex2 = int(math.floor(math.log10(e2)))
        put("pFimJobErr", f"\\ensuremath{{{e2 / 10 ** ex2:.1f}\\times 10^{{{ex2}}}}}", script=sv2, config="job-level GPU PEB vs independent NumPy full-matrix reference (random samples)",
            aggregation="largest relative deviation")
    # ---- review 4: derivative validation (scripts/p2_diag_derivatives.py) and the reference margin
    dvf = P2 / "diag_derivatives.json"
    if dvf.exists():
        dv = json.loads(dvf.read_text())
        sdv = "scripts/p2_diag_derivatives.py -> results/P2/diag_derivatives.json"
        a = dv["a_float64"]
        e_ = a["max_rel_at_best_step"]
        ex = int(math.floor(math.log10(e_)))
        put("pDerivFdMaxRel", f"\\ensuremath{{{e_ / 10 ** ex:.1f}\\times 10^{{{ex}}}}}", script=sdv,
            config=f"(a) float64 image method, {a['n_ue']} random UE positions, single- and double-bounce specular paths on the facades and ground, "
                   f"central differences at the best step h = {a['best_step_m']:g} m", aggregation="max over paths and {delay, az, el} x {x, y} of |fd - analytic| / |analytic|",
            raw={"max_rel_per_step": a["max_rel_per_step"], "n_paths": a["n_paths"]})
        put("pDerivFdBestStep", f"\\ensuremath{{10^{{{int(round(math.log10(a['best_step_m'])))}}}}}", script=sdv, config="(a) float64 check, steps 1e-4..1e-1 m",
            aggregation="step [m] with the smallest max relative error")
        put("pDerivFdNumUe", str(a["n_ue"]), script=sdv, config="(a) float64 check", aggregation="number of random UE positions")
        st = dv["b_sionna_float32"]["steps"]["1cm"]["counts"]["share_of_disagreements_exclusive"]
        cfg_b = "(b) Sionna float32 re-trace, +/-1 cm (subset of \\pDerivAgree), disagreeing paths (1e-3 relative), exclusive causes"
        put("pDerivDisFloat", pct(st["float32_rounding"]), script=sdv, config=cfg_b, aggregation="share explained by float32 rounding (within 3x the single-precision floor)",
            raw=dv["b_sionna_float32"]["steps"]["1cm"]["counts"])
        put("pDerivDisNearZero", pct(st["near_zero_derivative"]), script=sdv, config=cfg_b,
            aggregation="share whose failing values are near-zero derivatives (|an| < 1e-2 |grad|, error <= 1e-3 |grad|)")
        put("pDerivDisPathChange", pct(st["path_changes_vanish"] + st["path_changes_kink"]), script=sdv, config=cfg_b,
            aggregation="share where the path appears / vanishes or leaves the fixed-plane image-method model across the step")
        put("pDerivDisOther", pct(st["unexplained"]), script=sdv, config=cfg_b, aggregation="share unexplained")
    m3m = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    put("pRefMargin", f"{m3m['budget']['margin_ref_db']:.1f}", script="scripts/run_m3.py budget (configs/m3.yaml rework.budget) -> results/M3/metrics.json budget.margin_ref_db",
        config="3GPP short-range reference link (TR 38.802 Table A.2.1-1 budget); the 'Ref.' margin of the closing experiment (scripts/p2_closing.py uses the same run_m3 budget on the tuning seeds)",
        aggregation="median best-cell unblocked SNR over tuning jobs 101-105 minus SNR_req [dB] (paper-1 \\numRefMargin)")
    vt = P2 / f"validation_table_{tag}.json"
    if vt.exists():
        vtd = json.loads(vt.read_text())
        svt = "scripts/p2_validation_table.py -> paper2/tables/validation.tex, results/P2/validation_table_%s.json" % tag
        for bw, B in (("400", "Four"), ("100", "Hundred")):
            cen = [r["bandwidths"][bw]["centred_to_peb"] for r in vtd["rows"]]
            bia = [r["bandwidths"][bw]["bias_to_rmse"] for r in vtd["rows"]]
            put(f"pValCenToPeb{B}", f"{min(cen):.2g}--{max(cen):.2g}", script=svt, config=f"6 fixed geometries, {bw} MHz, main hardware, 500 single-epoch draws",
                aggregation="min-max over geometries of centred RMS sqrt(E||p_hat - E p_hat||^2) / single-epoch LoS-only PEB", raw=cen)
            put(f"pValBiasToRmse{B}", f"{min(bia):.2f}--{max(bia):.2f}", script=svt, config=f"6 fixed geometries, {bw} MHz, main hardware",
                aggregation="min-max over geometries of bias magnitude ||E p_hat - p|| / RMSE", raw=bia)
        bm = [(r["bandwidths"]["400"]["bias_to_rmse"] ** 2, r) for r in vtd["rows"]]
        big = [v for v, _ in bm if v > 0.5]
        small = [(v, r) for v, r in bm if v <= 0.5]
        put("pValBiasMseFour", f"{100 * min(big):.0f}--{100 * max(big):.0f}\\%", script=svt, config="400 MHz, main hardware; geometries where bias^2/MSE > 50 %",
            aggregation="min-max of squared bias / MSE (MSE = RMSE^2)", raw=[v for v, _ in bm])
        if small:
            put("pValBiasMseFourOther", ", ".join(f"{100 * v:.0f}\\%" for v, _ in small), script=svt,
                config="400 MHz, main hardware; the remaining geometry: " + "; ".join(f"{r['mount']} {r['state']} x={r['ue_x_m']:.1f} m" for _, r in small),
                aggregation="squared bias / MSE")
        put("pValIdentity", f"{vtd['identity_max_rel_dev']:.0e}", script=svt, config="all table cells",
            aggregation="max relative deviation of RMSE^2 from bias^2 + centred^2 (sample moments)")
    odf = P2 / f"onset_definitions_{tag}.json"
    if odf.exists():
        odd = json.loads(odf.read_text())
        put("pOnsetNumEvents", str(odd["n_events"]), script="scripts/p2_onset_definitions.py", config="serving-cell 10 dB LoS events, held-out runs",
            aggregation="number of events", raw=odd["epochs_per_window_total"])
    of2 = P2 / f"diag_onset_{tag}.json"
    if of2.exists():
        od2 = json.loads(of2.read_text())["variants"]["A"]
        so2 = "scripts/p2_diag_onset.py -> results/P2/diag_onset_%s.json" % tag
        for w, W in (("[-1,-0.5) s", "PreOne"), ("[-0.5,0) s", "PreHalf"), ("during", "During"), ("elsewhere", "Else")):
            r = od2[w]
            put(f"pOnset{W}Med", f"{100 * r['median_m']['mean']:.0f}", script=so2, config=f"estimator A, main configuration; window {w} of the serving-cell 10 dB events",
                aggregation="median error [cm]: per-seed median, mean over seeds", raw=r)
            put(f"pOnset{W}Pninety", f"{r['p90_m']['mean']:.2f}", script=so2, config=f"estimator A; window {w}", aggregation="p90 error [m]; per-seed p90, mean over seeds")
            put(f"pOnset{W}ShTen", pct(r["share_gt_0.1"]["mean"]), script=so2, config=f"estimator A; window {w}", aggregation="share of epochs with error > 0.1 m; mean over seeds")
            put(f"pOnset{W}ShQuarter", pct(r["share_gt_0.25"]["mean"]), script=so2, config=f"estimator A; window {w}", aggregation="share of epochs with error > 0.25 m; mean over seeds")
    for vv, V in (("A", "A"), ("AB", "B")):
        df_ = P2 / f"est_dev_{vv}.json"
        if df_.exists():
            r = json.loads(df_.read_text())["configs"]["bw400_tdoa_s1_p2_b"]
            sd = "scripts/p2_est_report.py --set dev --variant %s -> results/P2/est_dev_%s.json (development seeds)" % (vv, vv)
            put(f"pDev{V}Med", f"{100 * r['median_m']:.1f}", script=sd, config=f"variant {vv}, main configuration, DEVELOPMENT seeds 1001-1010",
                aggregation="median EKF error [cm]; per-seed, mean over seeds", raw={k: r[k] for k in r if k != "cfg"})
            put(f"pDev{V}Pninety", f"{r['p90_m']:.2f}", script=sd, config=f"variant {vv}, development seeds", aggregation="p90 [m]; per-seed, mean over seeds")
            put(f"pDev{V}Rmse", f"{r['rmse_m']:.1f}", script=sd, config=f"variant {vv}, development seeds", aggregation="RMSE [m]; per-seed, mean over seeds")
    if "P4_part1" in est:
        put("pPartOneRatio", f"{est['P4_part1']['median']:.1f}", script=se, config="P4 part 1, main configuration", aggregation="median over seeds of median error/PEB")
    return out


def used_macros() -> list[str]:
    f = PAPER2 / "main.tex"
    if not f.exists():
        return []
    names = []
    for line in f.read_text().splitlines():
        code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        names += MACRO.findall(code)
    return sorted(set(names))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="heldout")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--variant", default="v1", choices=["v1", "A", "AB"], help="estimator variant quoted by the paper")
    args = ap.parse_args()
    global VARIANT
    VARIANT = args.variant
    cat = catalog(args.set)
    (P2 / "numbers_catalog.json").write_text(json.dumps(cat, indent=1, default=str) + "\n")
    PAPER2.mkdir(exist_ok=True)
    lines = [f"% Generated by scripts/p2_make_paper.py from results/P2 ({args.set} seeds). Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v['value']}}}" for k, v in sorted(cat.items())]
    (PAPER2 / "numbers2.tex").write_text("\n".join(lines) + "\n")
    missing = [m for m in used_macros() if m not in cat]
    print(f"catalog: {len(cat)} macros -> paper2/numbers2.tex; used in main.tex: {len(used_macros())}; undefined: {', '.join(missing) or 'none'}")
    if args.compile:
        cmd = ["docker", "run", "--rm", "-v", f"{PAPER2}:/w", "-w", "/w", "texlive/texlive:latest", "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        log = (PAPER2 / "main.log").read_text(errors="ignore") if (PAPER2 / "main.log").exists() else ""
        undefined = sorted(set(re.findall(r"Undefined control sequence.*?\n.*?\\(\w+)", log)))
        warn = re.findall(r"LaTeX Warning: (Reference|Citation) `([^']+)' .*undefined", log)
        pages = re.findall(r"Output written on main.pdf \((\d+) page", log)
        print(f"latexmk exit {r.returncode}; pages {pages[-1] if pages else '?'}; undefined control sequences: {undefined or 'none'}; "
              f"undefined refs/cites: {sorted(set(w[1] for w in warn)) or 'none'}")
        if r.returncode != 0:
            print(r.stdout[-3000:])


if __name__ == "__main__":
    main()
