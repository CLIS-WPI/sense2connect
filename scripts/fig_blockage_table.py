"""Blockage characterization table per class and mount (M1.5 part 2 re-run on evaluation seeds).

Runs the M1.5 two-O-RU analysis (``scripts/diversity_study.py``:
serving-link 10 dB LoS events, other-O-RU availability, both-blocked time,
fixed vs oracle cell selection) from the cached comm traces
``results/cache/<mount>/<density>/seed_<s>/comm_trace.json`` (PathSolver +
model B, 0.1 s; written by ``scripts/cache_comm.py``) -- no re-tracing.
Raw UE events are rebuilt from the cached per-link trace with the same
``_ue_events`` call the trace loop uses; the M1.5 code then applies the
0.5 s hysteresis itself.

Check: the same code on the tuning seeds must reproduce
``results/M1.5/diversity.json`` (other-O-RU counts per mount and class,
both-blocked counts, fixed/oracle rates); the script stops otherwise.

Also writes results/M5/characterization.json: overall event rate and class
shares, same-cell surviving path (best post-model-B non-LoS path vs the
unblocked LoS during serving-link 10 dB events, M1.5 part-1 definition on
the two-O-RU config), other-cell availability, and the 10 ms onset
statistics from results/M3/metrics.json.

Writes paper/tables/tab_blockage.tex and results/M5/blockage_table.json.
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

import diversity_study as D  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from sim.scenes.loop import _ue_events  # noqa: E402

CLASSES = ("bus/truck", "pedestrian")


def load_rows(seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        for mount in D.MOUNTS:
            for density in D.DENSITIES:
                path = ROOT / "results" / "cache" / mount / density / f"seed_{seed}" / "comm_trace.json"
                tr = json.loads(path.read_text(encoding="utf-8"))
                times = [k * float(tr["dt_s"]) for k in range(int(tr["n_snapshots"]))]
                raw = _ue_events(tr["ue_trace"], times, [3.0, 10.0, 20.0])
                rows.append({
                    "seed": seed, "mount": mount, "density": density, "n_ue": int(tr["n_ue"]), "n_oru": int(tr["n_oru"]),
                    "n_snapshots": int(tr["n_snapshots"]), "dt_s": float(tr["dt_s"]), "raw_events": raw, "trace": tr["ue_trace"],
                })
    return rows


def analyse(rows: list[dict]) -> dict:
    events = D.part2_events(rows)
    time = D.part2_time(rows)
    rates = D.part2_rates(rows)
    per_class = {}
    for mount in D.MOUNTS:
        chosen = [r for r in rows if r["mount"] == mount]
        minutes = sum(r["n_ue"] * D.DURATION_S / 60.0 for r in chosen)
        evs = [ev for r in chosen for ev in D._serving_events(r)]
        per_class[mount] = {}
        for c in CLASSES + ("car",):
            sel = [ev for ev in evs if D._kind(ev) == c]
            dur = [(int(ev["end_snapshot"]) - int(ev["start_snapshot"]) + 1) * D.DT_S for ev in sel]
            per_class[mount][c] = {
                "n_events": len(sel),
                "events_per_ue_min": len(sel) / minutes if minutes else 0.0,
                "duration_p10_p50_p90_s": D._quantile(dur),
            }
    surviving = {}
    for mount in D.MOUNTS + ("all",):
        chosen = [r for r in rows if mount == "all" or r["mount"] == mount]
        surviving[mount] = {}
        for c in ("all",) + CLASSES:
            vals, no_alt, removed = [], 0, 0
            for r in chosen:
                links = D._by_link(r["trace"])
                for ev in D._serving_events(r):
                    if c != "all" and D._kind(ev) != c:
                        continue
                    v = D.best_alt_db(ev, links)
                    if v is None:
                        no_alt += 1
                    elif not np.isfinite(v):
                        removed += 1
                    else:
                        vals.append(v)
            surviving[mount][c] = {"n": len(vals) + no_alt + removed, "no_alternative": no_alt, "alternative_removed": removed,
                                   "p10_p50_p90_db": D._quantile(vals), "share_above_minus10db": (sum(1 for v in vals if v >= -10.0) / (len(vals) + no_alt + removed)) if vals or no_alt or removed else None}
    all_events = {c: sum(per_class[m][c]["n_events"] for m in D.MOUNTS) for c in CLASSES + ("car",)}
    total = sum(all_events.values())
    minutes = sum(r["n_ue"] * D.DURATION_S / 60.0 for r in rows)
    overall = {
        "events_per_ue_min": total / minutes if minutes else 0.0,
        "class_share": {c: (n / total if total else None) for c, n in all_events.items()},
        "n_events": total,
        "other_cell_clear": {c: {"unblocked": sum(events[m]["classes"][c]["unblocked"] for m in D.MOUNTS),
                                 "n": sum(events[m]["classes"][c]["n_events"] for m in D.MOUNTS)} for c in D.CLASSES},
    }
    return {"other_oru": events, "time_both_blocked": time, "rates": rates, "per_class": per_class, "surviving_path": surviving, "overall": overall}


def check_against_m15(res: dict) -> list[str]:
    ref = json.loads((ROOT / "results" / "M1.5" / "diversity.json").read_text())
    bad = []
    for mount in D.MOUNTS:
        for key in ("n_events", "unblocked"):
            if ref["other_oru"][mount][key] != res["other_oru"][mount][key]:
                bad.append(f"other_oru {mount} {key}: {ref['other_oru'][mount][key]} vs {res['other_oru'][mount][key]}")
        for c in D.CLASSES:
            for key in ("n_events", "unblocked"):
                a, b = ref["other_oru"][mount]["classes"][c][key], res["other_oru"][mount]["classes"][c][key]
                if a != b:
                    bad.append(f"other_oru {mount} {c} {key}: {a} vs {b}")
        if ref["time_both_blocked"][mount] != res["time_both_blocked"][mount]:
            bad.append(f"time_both_blocked {mount}")
    for a, b in zip(ref["rates"], res["rates"]):
        for key in ("fixed_rate", "oracle_rate"):
            if abs(float(a[key]) - float(b[key])) > 1e-9:
                bad.append(f"rates {a['mount']} {a['density']} {key}: {a[key]} vs {b[key]}")
    return bad


def p50(q: dict) -> str:
    return "--" if not q else f"{q['0.5'] if '0.5' in q else q[0.5]:.2f}"


def latex(res: dict) -> str:
    lines = [
        "% Generated by scripts/fig_blockage_table.py from results/cache comm traces (evaluation seeds). Do not edit.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Mount & Class & Events/UE-min & Dur.\\ p50 [s] & Other cell clear [\\%] & Other cell p50 [dB] \\\\",
        "\\midrule",
    ]
    for mount in D.MOUNTS:
        oo = res["other_oru"][mount]["classes"]
        for c in CLASSES:
            pc = res["per_class"][mount][c]
            share = 100.0 * oo[c]["unblocked"] / oo[c]["n_events"] if oo[c]["n_events"] else float("nan")
            ratio = D._quantile(oo[c]["ratios_db"]).get(0.5)
            lines.append(
                f"{mount} & {c} & {pc['events_per_ue_min']:.2f} & {p50(pc['duration_p10_p50_p90_s'])} & "
                f"{share:.0f} ({oo[c]['unblocked']}/{oo[c]['n_events']}) & {'--' if ratio is None else f'{ratio:.1f}'} \\\\"
            )
        t = res["time_both_blocked"][mount]
        rates = [r for r in res["rates"] if r["mount"] == mount]
        orc = "; ".join(f"{r['density']}: {r['fixed_rate']:.1f}$\\to${r['oracle_rate']:.1f}" for r in rates)
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\footnotesize {mount}: both cells $\\geq$10 dB {100.0 * t['both'] / t['total']:.1f}\\% of time; "
                     f"10 dB events/UE-min fixed$\\to$oracle {orc}}} \\\\")
        lines.append("\\midrule" if mount != D.MOUNTS[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds
    seeds = load_seeds()
    tune = analyse(load_rows([int(s) for s in seeds["tuning"]]))
    bad = check_against_m15(tune)
    if bad:
        raise SystemExit("cache re-run does not reproduce M1.5 on tuning seeds:\n" + "\n".join(bad))
    print("tuning-seed check: reproduces results/M1.5/diversity.json exactly", flush=True)
    res = analyse(load_rows([int(s) for s in seeds["evaluation"]]))
    out = ROOT / "results" / "M5"
    out.mkdir(parents=True, exist_ok=True)
    m3 = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    char = {
        "config": "configs/m2_scenario.yaml (two O-RUs, 8x8, 1024 SC), evaluation seeds, serving cell = strongest unblocked, 10 dB LoS events, min gap 0.5 s",
        "dt_s": D.DT_S,
        "overall": res["overall"],
        "per_class_per_mount": res["per_class"],
        "same_cell_surviving_path": res["surviving_path"],
        "both_blocked_time": res["time_both_blocked"],
        "fixed_vs_oracle_rates": res["rates"],
        "onset_10ms": {"source": "results/M3/metrics.json event_stats (10 ms model B, fixed cell = strongest unblocked)", **m3["event_stats"]["evaluation"]},
    }
    (out / "characterization.json").write_text(json.dumps(char, indent=1, default=str) + "\n")
    (out / "blockage_table.json").write_text(json.dumps({"evaluation": res, "tuning_check": "identical to results/M1.5/diversity.json"}, indent=1, default=str) + "\n")
    tab = ROOT / "paper" / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "tab_blockage.tex").write_text(latex(res), encoding="utf-8")
    print(latex(res))


if __name__ == "__main__":
    main()
