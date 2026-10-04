"""Regenerate every paper figure, the table and paper/numbers.tex, then compile the paper.

Steps:
1. Run the figure/table scripts (each one command, from saved results):
   fig_blockage_table.py, fig_leadtime.py, fig_onset.py,
   fig_outage_margin.py, fig_headroom.py.
2. Build the number catalog from results (results/M5/numbers_catalog.json):
   every value the paper may quote, under a descriptive key, as a plain
   string with its precision fixed here.
3. Read the macro names from the header comment of paper/main.tex (lines
   like ``% \\Name: definition``), map each to a catalog key (MACRO_MAP)
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
MACRO_MAP: dict[str, str] = {}

MACRO_LINE = re.compile(r"^%+\s*\\([A-Za-z]+)\s*(?:[:=\-–—]|\s)\s*(.*)$")


def run_scripts(leadtime_recompute: bool) -> None:
    for name in FIGURE_SCRIPTS:
        cmd = [sys.executable, str(ROOT / "scripts" / name)]
        if name == "fig_leadtime.py" and not leadtime_recompute and (RESULTS / "M5" / "leadtime.json").exists():
            cmd.append("--plot-only")
        print("$", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def _pct(x: float, digits: int = 0) -> str:
    return f"{100.0 * x:.{digits}f}"


def _num(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def catalog() -> dict[str, dict[str, str]]:
    """Every quotable number: key -> {value, description}."""
    out: dict[str, dict[str, str]] = {}

    def put(key: str, value: str, desc: str) -> None:
        out[key] = {"value": value, "description": desc}

    m3 = json.loads((RESULTS / "M3" / "metrics.json").read_text())
    gen = json.loads((RESULTS / "M3" / "genie.json").read_text())
    gen2 = json.loads((RESULTS / "M3" / "genie2.json").read_text())
    hyb = json.loads((RESULTS / "M3" / "hybrid.json").read_text())
    char = json.loads((RESULTS / "M5" / "characterization.json").read_text())
    lead = json.loads((RESULTS / "M5" / "leadtime.json").read_text())
    cfg = __import__("yaml").safe_load((ROOT / "configs" / "m3.yaml").read_text())["rework"]

    b = m3["budget"]
    put("budget.tx_power_dbm", _num(float(cfg["budget"]["tx_power_dbm"]), 0), "O-RU transmit power [dBm], TR 38.802 Tab. A.2.1-1")
    put("budget.ue_nf_db", _num(float(cfg["budget"]["ue_noise_figure_db"]), 0), "UE noise figure [dB], TR 38.802 Tab. A.2.1-1")
    put("budget.bandwidth_mhz", _num(m3["bandwidth_hz"] / 1e6, 2), "carrier bandwidth [MHz]")
    put("budget.service_rate_mbps", _num(float(cfg["service_rate_bps"]) / 1e6, 0), "service rate [Mbit/s]")
    put("budget.snr_req_db", _num(b["snr_req_db"], 1), "SNR needed for the service rate [dB]")
    put("budget.margin_ref_db", _num(b["margin_ref_db"], 1), "3GPP short-range reference margin [dB]")
    put("sensing.overhead_pct", _num(100.0 * m3["overhead_xapp"], 2), "sensing overhead [% of resources]")

    ov = char["overall"]
    put("char.events_per_ue_min", _num(ov["events_per_ue_min"], 1), "serving-link 10 dB events per UE-minute (eval seeds, 0.1 s)")
    for c, key in (("bus/truck", "bus"), ("pedestrian", "ped"), ("car", "car")):
        share = ov["class_share"][c]
        put(f"char.class_share_{key}_pct", _pct(share or 0.0), f"share of 10 dB events caused by {c} [%]")
    for c, key in (("bus/truck", "bus"), ("pedestrian", "ped")):
        oc = ov["other_cell_clear"][c]
        put(f"char.other_clear_{key}_pct", _pct(oc["unblocked"] / oc["n"]), f"{c} events with the other cell clear (<3 dB) [%]")
        sp = char["same_cell_surviving_path"]["all"][c]["p10_p50_p90_db"]
        put(f"char.surviving_path_{key}_p50_db", _num(float(sp["0.5"]), 0), f"{c}: median best surviving same-cell path vs unblocked LoS [dB]")
        on = char["onset_10ms"][c]["onset_10_90_s"]
        put(f"char.onset_{key}_p50_s", _num(on["p50"], 2), f"{c}: median 10-90 % onset [s] (10 ms)")
        put(f"char.onset_{key}_p90_s", _num(on["p90"], 2), f"{c}: 90th-percentile 10-90 % onset [s]")
        put(f"char.actionable_{key}_pct", _pct(char["onset_10ms"][c]["actionable_share"]), f"{c}: actionable 10 dB events (10 ms) [%]")
    sp_all = char["same_cell_surviving_path"]["all"]["all"]
    put("char.surviving_path_p50_db", _num(float(sp_all["p10_p50_p90_db"]["0.5"]), 0), "median best surviving same-cell path vs unblocked LoS [dB]")
    put("char.surviving_within_10db_pct", _pct(sp_all["share_above_minus10db"] or 0.0), "events whose best surviving path is within 10 dB of LoS [%]")
    both = char["both_blocked_time"]
    tot = sum(v["total"] for v in both.values())
    put("char.both_blocked_time_pct", _num(100.0 * sum(v["both"] for v in both.values()) / tot, 1), "time with both cells >= 10 dB down [%]")

    for tracker in ("map", "unconstrained"):
        for c, key in (("bus/truck", "bus"), ("pedestrian", "ped")):
            s = lead["series"][f"{tracker}|{c}"]
            for i, L in enumerate(lead["leads_s"]):
                put(f"lead.{tracker}_{key}_{str(L).replace('.', 'p')}s_pct", _pct(s["share"][i]), f"{c}, {tracker} tracker: events tracked {L} s before onset [%]")

    labels = b["labels"]
    tau = f"{gen['tau_ho_default_s']:.3f}"
    for mi, lab in enumerate(labels):
        tag = {"3GPP short-range reference": "ref", "v1 radio (high margin)": "v1"}.get(lab, lab.replace(" dB", "db"))
        ev = m3["evaluation"][mi]
        put(f"out.{tag}.a3", _num(ev["a3"]["outage_req_s_per_min"]["mean"], 3), f"A3 outage at the service rate, margin {lab} [s/UE-min]")
        put(f"out.{tag}.oracle", _num(ev["oracle"]["outage_req_s_per_min"]["mean"], 3), f"oracle outage, margin {lab} [s/UE-min]")
        put(f"out.{tag}.xapp", _num(ev["xapp"]["outage_req_s_per_min"]["mean"], 3), f"xApp + A3 outage, margin {lab} [s/UE-min]")
        put(f"out.{tag}.hybrid", _num(hyb["margins"][mi]["evaluation"]["hybrid_joint"]["outage_req_s_per_min"]["mean"], 3), f"hybrid outage, margin {lab} [s/UE-min]")
        g = gen["margins"][mi]["tau_ho"][tau]
        put(f"out.{tag}.genie", _num(g["genie_no_overhead"]["outage_req_s_per_min"]["mean"], 3), f"genie + A3 (no overhead) outage, margin {lab} [s/UE-min]")
        gc = gen["margins"][mi]["gap_closure"]
        if gc["a3_gap_tau_default"] > 0:
            put(f"tau0.{tag}.gap_closed_pct", _pct(1.0 - gc["a3_gap_tau0"] / gc["a3_gap_tau_default"]), f"A3-oracle gap closed by tau_HO = 0, margin {lab} [%]")
        fs = gen2["margins"][mi]["foresight"]["all"]
        put(f"fs.{tag}.share_pct", _num(100.0 * (fs["share_of_a3_outage_per_job"]["mean"] or 0.0), 1), f"foresight bound, per-job mean share of A3 outage, margin {lab} [%]")
        d = g["a3_decomp"]
        a3o = g["a3"]["outage_req_s_per_min"]["mean"]
        put(f"dec.{tag}.interruption_share_pct", _pct(d["b_interruption"]["mean"] / a3o if a3o else 0.0), f"handover interruption share of A3 outage, margin {lab} [%]")
    xp = m3["evaluation"][labels.index("3GPP short-range reference")]["xapp"]
    put("xapp.ref.precision_pct", _pct(xp["precision"]["mean"]), "xApp handover precision at the 3GPP reference [%]")
    return out


def macros_in_main() -> list[tuple[str, str]]:
    main = PAPER / "main.tex"
    if not main.exists():
        return []
    names = []
    for line in main.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("%"):
            if line.strip().startswith("\\documentclass"):
                break
            continue
        m = MACRO_LINE.match(line.strip())
        if m:
            names.append((m.group(1), m.group(2).strip()))
    return names


def write_numbers(cat: dict[str, dict[str, str]]) -> dict[str, Any]:
    wanted = macros_in_main()
    lines = ["% Generated by scripts/make_paper.py from results/. Do not edit."]
    defined, unmapped, missing = [], [], []
    for name, desc in wanted:
        key = MACRO_MAP.get(name)
        if key is None:
            unmapped.append((name, desc))
            continue
        if key not in cat:
            missing.append((name, key))
            continue
        lines.append(f"\\newcommand{{\\{name}}}{{{cat[key]['value']}}}  % {cat[key]['description']}")
        defined.append(name)
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"main_tex_found": (PAPER / "main.tex").exists(), "macros_in_main": len(wanted), "defined": defined, "unmapped": unmapped, "mapped_key_missing": missing}


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
            print(f"  NOT DEFINED (catalog key '{key}' missing): \\{name}")
    compile_paper()


if __name__ == "__main__":
    main()
