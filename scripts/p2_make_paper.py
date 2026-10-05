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

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "results" / "P2"
PAPER2 = ROOT / "paper2"
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
    seeds = "held-out test seeds 2001-2010 (configs/seeds_heldout.yaml)" if tag == "heldout" else "development seeds 1001-1010"

    def put(name, value, *, script, config, aggregation, raw=None):
        out[name] = {"value": value, "script": script, "config": config, "seeds": seeds + ", 40 runs (2 mounts x 2 densities), 2 UEs, 600 epochs",
                     "aggregation": aggregation, "raw": raw}

    peb = json.loads((P2 / f"peb_{tag}.json").read_text())
    est = json.loads((P2 / f"est_{tag}.json").read_text())
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
    for cname, K in (("estimator bw400_tdoa_s1_p2_b", "Est"), ("PEB los-only", "PebLos"), ("PEB map-aided", "PebMap"), ("perfect", "Perf"), ("white 1 m", "White")):
        if cname not in cl:
            continue
        for lab, M in (("15 dB", "Fifteen"), ("20 dB", "Twenty"), ("3GPP short-range reference", "Ref")):
            v = cl[cname][lab]
            put(f"pClose{K}{M}", f"{v['mean_diff']:+.3f}" + ("$^\\dagger$" if not v["wilcoxon_p_two_sided"] < 0.05 else ""),
                script="scripts/p2_closing.py -> results/P2/closing_%s.json" % tag, config=f"paper-1 planner (v1.4.1, perfect blocker tracks, H fixed) with UE position '{cname}'; margin {lab}",
                aggregation="planner - A5 outage [s/UE-min], mean over seeds of per-seed means; dagger = exact Wilcoxon p >= 0.05 (pointwise, exploratory)", raw=v)
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
    args = ap.parse_args()
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
