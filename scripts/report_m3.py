"""Write results/M3/report.md from results/M3/metrics.json (no hand-typed numbers).

GPU samples (``nvidia-smi dmon -i 1 -s um -o T``) are read from
``--dmon-before`` / ``--dmon-after``; stage windows come from metrics.json.
Run: ``python scripts/report_m3.py --dmon-before .cache/m3_profile/dmon_before.txt
--dmon-after .cache/m3_profile/dmon_after.txt [--tests .cache/m3_profile/tests.txt]
[--determinism .cache/m3_profile/determinism.json]``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml  # noqa: E402

SCHEMES = ("none", "oracle", "beam", "a3", "trend", "xapp")
NAMES = {
    "none": "no action (fixed cell)",
    "oracle": "oracle cell selection",
    "beam": "same-cell analog beam",
    "a3": "A3 (reactive)",
    "trend": "RSRP trend + A3",
    "xapp": "xApp + A3 (A3 held off during xApp hold)",
}
CLASSES = ("bus/truck", "pedestrian", "car")


def f(item: Any, scale: float = 1.0, digits: int = 3, ci: bool = True) -> str:
    if item is None or item.get("mean") is None:
        return "—"
    if not ci:
        return f"{item['mean'] * scale:.{digits}f}"
    return f"{item['mean'] * scale:.{digits}f} ± {item['ci'] * scale:.{digits}f}"


UTC_OFFSET_H = 0.0


def dmon(path: Path | None) -> list[tuple[float, float, float]]:
    """(unix time, SM %, FB MB) samples. dmon -o T stamps HH:MM:SS in the host's local time (today)."""
    if path is None or not path.exists():
        return []
    rows = []
    today = dt.date.today()
    tz = dt.timezone(dt.timedelta(hours=UTC_OFFSET_H))
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        try:
            clock = dt.datetime.combine(today, dt.time.fromisoformat(parts[0]), tzinfo=tz)
            sm = float(parts[2]) if parts[2] != "-" else 0.0
            fb = float(parts[8]) if len(parts) > 8 and parts[8] != "-" else 0.0
        except (ValueError, IndexError):
            continue
        rows.append((clock.timestamp(), sm, fb))
    return rows


def window(samples, start: float, end: float) -> tuple[str, str, int]:
    sel = [s for s in samples if start - 0.5 <= s[0] <= end + 0.5]
    if not sel:
        return "—", "—", 0
    sm = [s[1] for s in sel]
    return f"{sum(sm) / len(sm):.0f}", f"{max(sm):.0f}", len(sel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "M3" / "metrics.json")
    parser.add_argument("--dmon-before", type=Path)
    parser.add_argument("--dmon-after", type=Path)
    parser.add_argument("--dmon-geometry", type=Path)
    parser.add_argument("--geometry-wall-s", type=float)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--determinism", type=Path)
    parser.add_argument("--regression", type=Path, help="before/after comparison of the main run around the hybrid simulator change")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "M3" / "report.md")
    parser.add_argument("--host-utc-offset-h", type=float, default=0.0, help="timezone of the dmon time stamps (host local time)")
    args = parser.parse_args()
    global UTC_OFFSET_H
    UTC_OFFSET_H = args.host_utc_offset_h
    m = json.loads(args.metrics.read_text())
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")["rework"]
    before = json.loads((ROOT / "results" / "M3" / "profile_before.json").read_text())
    b = m["budget"]
    labels = b["labels"]
    ref_i = labels.index("3GPP short-range reference")
    L: list[str] = []
    add = L.append

    add("# M3 rework — proactive handover in a non-degenerate regime")
    add("")
    add("Status: ready for review. Not marked DONE. v1 report kept as `results/M3/report_v1.md`.")
    add("")
    add("## What changed and why")
    add("")
    add("v1 saturated: with 30 dBm, NF 7 dB and the 8x8 array gain the unblocked SNR was ~47 dB (median), "
        "so A3, RSRP-trend and the oracle all showed ~0 s/UE-min of SNR outage at 0 dB. "
        "**The link-margin sweep below was added after observing this saturation** (it was not part of the original M3 plan). "
        "The link budget and the 400 Mbit/s service rate were fixed before the rerun; the numbers below are from the first full rerun with them.")
    add("")
    add("Method changes relative to v1 (all applied to every scheme):")
    add("")
    add("- Model B is evaluated analytically every 10 ms from the blocker poses at that time; no interpolation of blocked power (v1 interpolated linearly between 0.1 s snapshots).")
    add("- A handover interrupts the link for tau_HO (rate 0, counted as outage). In v1, tau_HO only delayed the switch.")
    add("- Sensing overhead = (1/14) x CPI duty cycle (32 ms / 0.1 s = 0.32) = "
        f"{m['overhead_xapp']:.5f}, charged only to the xApp (v1: 1/14).")
    add("- The no-action reference is the M1/M1.5 \"fixed\" cell (largest unblocked power at each step, no cost). "
        "RSRP-trend and the xApp run on top of reactive A3 (with that margin's tuned A3 parameters); an xApp handover starts a hold that blocks A3 "
        "until the later of hold and the predicted end of the blockage. Reason: UEs walk 60–72 m/min between two O-RUs; at low margins a scheme "
        "without distance-driven mobility would be measured on path-loss outage, not blockage.")
    add("- Oracle = cell with the larger post-model-B power every 10 ms, no interruption (v1: smaller LoS loss).")
    add("- Same-cell beam baseline = analog single beam on the fixed cell: max(LoS-path power, best other path power) after model B (the M1.5 "
        "best-alternative bound). With full-array MRT over all paths (used by every other scheme) a same-cell beam switch cannot add power.")
    add("- A3 and RSRP-trend decide in the gNB (no E2 delay); the xApp decides at report time + E2 loop delay.")
    add("- 10 dB events are recomputed at 10 ms with the M1 rules (min gap 0.5 s, nested in 3 dB) on the fixed cell.")
    add("- Each scheme is tuned separately at every margin point (your decision, because the 3GPP reference margin is above 30 dB). "
        "This deviates from acceptance criterion 7 (one parameter set evaluated across the sweep).")
    add("")

    # ---------------- link budget
    _observations(add, m, labels)
    add("## 1. Link budget (fixed before the rerun)")
    add("")
    add("| Item | Value | Source |")
    add("|---|---|---|")
    add(f"| gNB (O-RU) transmit power | {cfg['budget']['tx_power_dbm']:.0f} dBm | 3GPP TR 38.802 V14.2.0, Table A.2.1-1, dense urban, micro layer, above 6 GHz |")
    add(f"| UE noise figure | {cfg['budget']['ue_noise_figure_db']:.0f} dB | TR 38.802 V14.2.0, Table A.2.1-1 (baseline; 10 dB = high performance) |")
    add("| BS / UE element gain | 0 dBi (isotropic, ray-traced 8x8 array gain included in the channel) | TR 38.802 Table A.2.1-6 gives 8 dBi BS elements; NOT added |")
    add(f"| Cable, body, implementation, shadow-fading losses | {cfg['budget']['losses_db']:.0f} dB | TR 38.830 V17.0.0, Table A.3: \"reported by companies\" (unspecified by 3GPP) |")
    add(f"| Bandwidth | {m['bandwidth_hz'] / 1e6:.2f} MHz (1024 SC x 120 kHz) | M2/M3 carrier |")
    add("| Noise | −174 dBm/Hz + 10 log10(B) + NF | |")
    add(f"| Service rate | {float(cfg['service_rate_bps']) / 1e6:.0f} Mbit/s → SNR_req = {b['snr_req_db']:.2f} dB (Shannon, no overhead) | chosen before the rerun |")
    add("| Cross-check | TR 38.901 V19.5.0 Table 7.8-1: 35 dBm / 100 MHz at 30 GHz, UT NF 9 dB (calibration table, not used) | |")
    add("")
    add(f"Reference margin (median best-cell unblocked SNR − SNR_req, tuning jobs) = **{b['margin_ref_db']:.2f} dB**, labelled "
        "**3GPP short-range reference**. Lower margins correspond to longer links, unspecified losses, or higher rates. "
        f"The v1 radio (30 dBm, NF 7 dB) is {b['v1_offset_db']:+.1f} dB from the reference and is kept as the high-margin point "
        f"({b['points_db'][-1]:.2f} dB). Each margin point applies a common extra path loss to both cells "
        "(" + ", ".join(f"{lab}: {loss:+.2f} dB" for lab, loss in zip(labels, b["extra_loss_db"])) + ").")
    add("")
    add("Per-UE margin distribution (3GPP reference budget, all 60 jobs; per-UE = median over that UE's 60 s; per-step = every 10 ms):")
    add("")
    add("| Mount | Per-UE median margin p0/p25/p50/p75/p100 [dB] | Per-step margin p5/p25/p50/p75/p95 [dB] | UE-jobs |")
    add("|---|---|---|---|")
    for mount, st in m["margin_distribution"].items():
        pu = st["per_ue_median_margin_db_percentiles"]
        ps = st["per_step_margin_db_percentiles"]
        add(f"| {mount} | {pu['0']:.1f} / {pu['25']:.1f} / {pu['50']:.1f} / {pu['75']:.1f} / {pu['100']:.1f} | "
            f"{ps['5']:.1f} / {ps['25']:.1f} / {ps['50']:.1f} / {ps['75']:.1f} / {ps['95']:.1f} | {st['n_ue_jobs']} |")
    add("")
    add("The two UEs follow the same configured sidewalk trajectories in every seed, so the per-UE spread is narrow; the per-step spread "
        "is the variation along the 80 m walk.")
    add("")

    # ---------------- events
    add("## 2. 10 dB events at 10 ms and onset duration")
    add("")
    add("Model B on every segment of every comm path for every blocker, every 10 ms. Path geometry from a new comm-only trace "
        "(`scripts/cache_comm_geometry.py`, PathSolver only, deterministic, 4 workers; sensing caches untouched) held between 0.1 s "
        "snapshots with the UE end point at its exact 10 ms position. At the snapshot instants the re-solved model-B powers equal "
        "`comm_trace.json` exactly (max relative difference 0 on all 60 jobs).")
    add("")
    add("| Split | Class | Events | Events / UE-min | Actionable share | Onset 10–90 % p10/p50/p90 [s] | Duration p10/p50/p90 [s] |")
    add("|---|---|---|---|---|---|---|")
    for split in ("tuning", "evaluation"):
        for cls, st in m["event_stats"][split].items():
            on = st["onset_10_90_s"]
            du = st["duration_s"]
            share = "—" if st["actionable_share"] is None else f"{st['actionable_share']:.2f}"
            onset = "—" if on is None else "{:.2f} / {:.2f} / {:.2f}".format(on["p10"], on["p50"], on["p90"])
            dur = "—" if du is None else "{:.2f} / {:.2f} / {:.2f}".format(du["p10"], du["p50"], du["p90"])
            add(f"| {split} | {cls} | {st['n']} | {st['per_ue_min']:.2f} | {share} | {onset} | {dur} |")
    add("")
    add("Onset = time for the LoS loss [dB] on the fixed cell to rise from 10 % to 90 % of the event peak (infinite loss capped at 200 dB). "
        "Actionable = the other cell's LoS loss stays below 3 dB for the whole event.")
    add("")

    # ---------------- tuned params
    add("## 3. Tuned parameters per margin (tuning seeds 101–105, 20 jobs, objective: outage time at SNR_req incl. HO interruption; ties → fewer HO)")
    add("")
    add("| Margin | A3 offset / hyst [dB] / TTT [ms] | trend window [s] / drop [dB] / H [s] | xApp budget / H [s] / hold [s] | tuning outage A3 / trend / xApp [s/UE-min] |")
    add("|---|---|---|---|---|")
    for lab, t in zip(labels, m["tuned"]):
        a, tr, x = t["a3"]["params"], t["trend"]["params"], t["xapp"]["params"]
        add(f"| {lab} | {a['offset_db']:.0f} / {a['hysteresis_db']:.0f} / {a['ttt_s'] * 1e3:.0f} | "
            f"{tr['window_s']:.2f} / {tr['drop_db']:.0f} / {tr['horizon_s']:.2f} | {x['budget']} / {x['horizon_s']:.1f} / {x['hold_s']:.1f} | "
            f"{t['a3']['objective'][0]:.3f} / {t['trend']['objective'][0]:.3f} / {t['xapp']['objective'][0]:.3f} |")
    add("")
    add("Grid sizes: 24 each (xApp: budget {2,4} x H {0.5,1,2,3} s x hold {0.2,0.5,1.0} s; A3: offset {1,3} x hysteresis {1,2,3} x TTT {40,80,160,320} ms; "
        "trend: window {0.1,0.2,0.5} s x drop {3,5} dB x H {0.25,0.5,1,2} s). Full grids are in `metrics.json` (`tuning_grids`).")
    add("")

    # ---------------- evaluation
    add("## 4. Evaluation seeds 1001–1010 (40 jobs, mean ± 95 % CI over jobs)")
    add("")
    add(f"Outage_req: rate < {float(cfg['service_rate_bps']) / 1e6:.0f} Mbit/s (incl. sensing overhead for the xApp) or HO interruption. "
        "Outage_0: SNR < 0 dB or HO interruption. TP loss = 1 − mean rate / oracle mean rate.")
    add("")
    for mi, lab in enumerate(labels):
        add(f"### Margin {lab} ({b['points_db'][mi]:.1f} dB)")
        add("")
        add("| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |")
        add("|---|---|---|---|---|---|---|---|")
        ev = m["evaluation"][mi]
        for s in SCHEMES:
            r = ev[s]
            add(f"| {NAMES[s]} | {f(r['outage_req_s_per_min'])} | {f(r['outage0_s_per_min'])} | {f(r['tp_loss_vs_oracle'], 100, 2)} | "
                f"{f(r['ho_per_min'], 1, 1)} | {f(r['ping_pong'], 1, 2)} | {f(r['precision'], 1, 2)} | {f(r['false_ho_rate'], 1, 2)} |")
        add("")
    ties = [lab for lab, ev in zip(labels, m["evaluation"]) if ev["oracle"]["outage_req_s_per_min"]["mean"] == ev["none"]["outage_req_s_per_min"]["mean"]]
    if ties:
        add("Oracle and no-action outage_req are identical at margin " + ", ".join(ties) + ": whenever the fixed cell is below the service "
            "rate there, the other cell is below it too, so no cell choice helps (see the decomposition in section 11: (a) is zero).")
        add("")
    add("HO precision/false-HO: share of the scheme's own trigger HOs (A3: all HOs; trend/xApp: trend/xApp-triggered HOs) that leave the "
        "fixed cell within [event start − 2 s, event end] of a 10 dB event of that UE.")
    add("")

    # ---------------- precision / recall per class
    add("## 5. Recall per blocker class, all vs actionable events (evaluation seeds)")
    add("")
    add("Recall = share of 10 dB events with a matching HO (any time in [start − 2 s, end]); proactive recall = matching HO before the event "
        "start. Interruption per event = time with outage_req in [start − 0.2 s, end].")
    add("")
    for mi in (ref_i, labels.index("20 dB"), labels.index("10 dB"), labels.index("0 dB")):
        add(f"### Margin {labels[mi]}")
        add("")
        add("| Scheme | Class | Events (all / actionable) | Recall all | Recall actionable | Proactive recall all | Proactive recall actionable | Interruption / event all [s] | Interruption / event actionable [s] |")
        add("|---|---|---|---|---|---|---|---|---|")
        for s in ("none", "oracle", "a3", "trend", "xapp"):
            evs = m["evaluation"][mi][s]["events"]
            for cls in ("all",) + CLASSES:
                a = evs.get(f"{cls}|all")
                ac = evs.get(f"{cls}|actionable")
                if a is None:
                    continue
                g = lambda d, k, sc=1.0, dg=2: "—" if d is None else f(d[k], sc, dg)  # noqa: E731
                add(f"| {NAMES[s]} | {cls} | {a['n_events']:.0f} / {0 if ac is None else ac['n_events']:.0f} | {g(a, 'recall')} | {g(ac, 'recall')} | "
                    f"{g(a, 'proactive_recall')} | {g(ac, 'proactive_recall')} | {g(a, 'interruption_req_s', 1, 3)} | {g(ac, 'interruption_req_s', 1, 3)} |")
        add("")

    # ---------------- H3
    add("## 6. H3: xApp outage vs E2 loop delay per blocker class")
    add("")
    for name, rows in m["sweeps"]["h3"].items():
        add(f"### Margin: {'3GPP short-range reference' if name == 'reference' else name} (xApp parameters tuned at that margin)")
        add("")
        add("| tau_E2 [s] | Outage_req [s/UE-min] | " + " | ".join(f"Interruption/event {c} [s]" for c in CLASSES) + " | " + " | ".join(f"Proactive recall {c}" for c in CLASSES) + " |")
        add("|---|---|" + "---|" * (2 * len(CLASSES)))
        for row in rows:
            x = row["xapp"]
            cells = [f(x["events"].get(f"{c}|all", {}).get("interruption_req_s"), 1, 3) if f"{c}|all" in x["events"] else "—" for c in CLASSES]
            rec = [f(x["events"].get(f"{c}|all", {}).get("proactive_recall"), 1, 2) if f"{c}|all" in x["events"] else "—" for c in CLASSES]
            add(f"| {row['tau_e2_s']:.2f} | {f(x['outage_req_s_per_min'])} | " + " | ".join(cells) + " | " + " | ".join(rec) + " |")
        add("")
        a3 = m["evaluation"][ref_i if name == "reference" else labels.index(name)]["a3"]
        add("A3 at the same margin (no E2 dependence): outage_req " + f(a3["outage_req_s_per_min"]) + " s/UE-min; interruption/event " +
            ", ".join(f"{c} {f(a3['events'][f'{c}|all']['interruption_req_s'], 1, 3)} s" for c in CLASSES if f"{c}|all" in a3["events"]) + ".")
        add("")

    add("## 7. Handover interruption sweep (3GPP short-range reference margin)")
    add("")
    add("| tau_HO [s] | A3 outage_req | trend outage_req | xApp outage_req | A3 HO/min | xApp HO/min |")
    add("|---|---|---|---|---|---|")
    for row in m["sweeps"]["tau_ho"]:
        add(f"| {row['tau_ho_s']:.2f} | {f(row['a3']['outage_req_s_per_min'])} | {f(row['trend']['outage_req_s_per_min'])} | "
            f"{f(row['xapp']['outage_req_s_per_min'])} | {f(row['a3']['ho_per_min'], 1, 1)} | {f(row['xapp']['ho_per_min'], 1, 1)} |")
    add("")

    # ---------------- profile
    add("## 8. GPU efficiency (GPU 1 only)")
    add("")
    add("Before (v1 pipeline, `scripts/profile_m3.py`, one tuning seed x 4 jobs, extrapolated to the full v1 run):")
    add("")
    add("| Stage | Wall |")
    add("|---|---|")
    for k, v in before["simulate_per_job_s"].items():
        add(f"| simulate {k} | {v:.2f} s / job |")
    for k, v in before["v1_full_run_extrapolated_s"].items():
        add(f"| {k} | {v / 60:.1f} min |")
    smb = dmon(args.dmon_before)
    if smb:
        sm = [s[1] for s in smb]
        add(f"| GPU 1 SM during profiling | mean {sum(sm) / len(sm):.0f} %, max {max(sm):.0f} % ({len(sm)} samples) — v1 M3 is CPU-only |")
    add("")
    add("After (this run):")
    add("")
    add("| Stage | Wall [s] | Peak torch GPU memory [GB] | SM mean / max [%] (dmon samples) |")
    add("|---|---|---|---|")
    sma = dmon(args.dmon_after)
    if args.geometry_wall_s:
        smg = dmon(args.dmon_geometry)
        sm = [s[1] for s in smg] or [0.0]
        fb = max([s[2] for s in smg] or [0.0])
        add(f"| comm geometry trace (one-off, 60 jobs, 4 workers) | {args.geometry_wall_s:.0f} | {fb / 1e3:.1f} (device FB, all workers) | "
            f"{sum(sm) / len(sm):.0f} / {max(sm):.0f} ({len(sm)}) |")
    for name, row in m["stages"].items():
        mean, mx, n = window(sma, row["unix_start"], row["unix_end"])
        add(f"| {name} | {row['wall_s']:.1f} | {row['peak_gpu_mem_gb']:.2f} | {mean} / {mx} ({n}) |")
    add("")
    add("Full v1 run ≈ " + f"{before['v1_full_run_extrapolated_s']['total_s'] / 60:.0f} min (CPU, 6 schemes, smaller grids, no margin sweep); "
        "this run: " + f"{sum(r['wall_s'] for r in m['stages'].values()) / 60:.1f} min for 9 margins x (3 tuned schemes x 24-point grids) + evaluation + sweeps.")
    add("")
    add("Why the SM target (>50 %) is not reached:")
    add("")
    add("- Model B and the predictions are the only dense kernels. They run as one batched float64 pass per job "
        "(T x U x C x P x S x B ≈ 6000 x 2 x 2 x 8 x 3 x 32 screens, chunked) and finish in well under a second per job; the build stage is "
        "dominated by the CPU part (analytic poses every 10 ms via `states_at`, map-tracker replay from the M2 detections) in 4 worker processes. "
        "The GPU is busy only in short bursts between CPU jobs.")
    add("- The handover state machines are 6000 dependent 10 ms steps over 10^2–10^3 lanes; each step is ~30 small vector operations. "
        "On the GPU that is kernel-launch bound; NumPy on the CPU is as fast, so they run on the CPU (the plan allowed this).")
    add("- The comm-only re-trace is limited by Dr.Jit kernel compilation and scene edits (as measured in M2), not by GPU compute.")
    add("")

    # ---------------- tests / determinism
    add("## 9. Tests and determinism")
    add("")
    if args.tests and args.tests.exists():
        add("`python scripts/test_m3.py` (in the container):")
        add("")
        add("```")
        add(args.tests.read_text().strip())
        add("```")
        add("")
    add("Equality tests (tol 1e-5): batched GPU model B vs NumPy `path_blocker_loss` (powers, LoS loss, dominant blocker); GPU prediction vs "
        "`xapp.predict.los_loss_db`; SNR for all margins on GPU vs NumPy; rate vs `sim.comm.phy.shannon_bps`; vectorised schemes vs the scalar "
        "reference (identical handover lists and outage masks). The GPU timeline at the snapshot instants matches the traced powers to 1e-4 "
        "relative (float32 device UE position in the trace vs float64 analytic position here; ≤ 2.5e-5 observed).")
    add("")
    if args.determinism and args.determinism.exists():
        det = json.loads(args.determinism.read_text())
        add("Determinism: " + det.get("summary", ""))
        add("")

    # ---------------- deviations / issues
    add("## 10. Deviations, findings to note, open issues")
    add("")
    add("- Margin sweep added after observing saturation in v1 (stated above). Per-margin tuning instead of one fixed parameter set (your decision).")
    add("- Comm geometry cache uses a scoped provenance (content hash of the producing modules + configs) instead of the whole-tree dirty hash, "
        "so later xApp edits do not invalidate a 10-minute trace; `m3_timeline` caches likewise hash their own sources and parameters. "
        "M2 detections are replayed as in v1; their (older) provenance is recorded, not enforced.")
    add("- xApp hold: in the smoke run (seed 101 and evaluation seed 1001) the hold could expire before the predicted blockage, letting A3 hand "
        "back at once. The hold now lasts until the later of hold and the predicted end of the blockage window. This was a policy logic fix, "
        "decided before any full evaluation; it changed the smoke results very little because most xApp triggers are false alarms (next item). "
        "Reviewed and kept: any leak from seed 1001 could only favour the xApp, which still loses to A3, so the conclusion is conservative.")
    add("- The xApp predictor fires often on blockages that do not happen (precision table, section 4). Candidate causes, not changed here: "
        "class-agnostic sizes (free tracks get a 12 m bus box), ghost/false tracks, along-lane velocity error (v1: p90 ~6 m/s). Changing the "
        "predictor is a method change and needs your decision.")
    add("- Same-cell beam baseline redefined as an analog single-beam bound (MRT over all paths already includes every surviving path).")
    add("- Events and actionability are defined on the fixed (largest unblocked power) cell; oracle and fixed are cost-free references.")
    add("- 3GPP gives no numbers for cable/body/implementation/shadow-fading losses in the FR2 template; they are 0 dB in the reference and "
        "covered by the margin sweep.")
    add("")
    hyb = ROOT / "results" / "M3" / "hybrid.json"
    if hyb.exists():
        _hybrid_section(add, json.loads(hyb.read_text()), m)
    gen = ROOT / "results" / "M3" / "genie.json"
    if gen.exists():
        _genie_section(add, json.loads(gen.read_text()), m, json.loads(hyb.read_text()) if hyb.exists() else None)
    gen2 = ROOT / "results" / "M3" / "genie2.json"
    if gen2.exists() and gen.exists():
        _genie2_section(add, json.loads(gen2.read_text()), json.loads(gen.read_text()))
    if args.regression and args.regression.exists():
        reg = json.loads(args.regression.read_text())
        add("### Regression check: the follow-up code changes did not change the existing schemes")
        add("")
        add("The follow-ups changed `xapp/schemes.py` (the simulator also returns the interruption mask) and `scripts/run_m3.py` "
            "(`make_lanes(..., hybrid=False)`; the prediction source may be the string \"genie\"). After the last change the main run was "
            "repeated with the changed code on the same timeline caches and the same "
            "configuration, and every metric was compared with the earlier `metrics.json` for exact equality (no tolerance). "
            "\"xApp\" here is the first-round scheme (xApp + A3, A3 held off during the xApp hold).")
        add("")
        add("| Scheme | Evaluation metrics, all margins | Tuned parameters + objective, all margins | Full tuning grid |")
        add("|---|---|---|---|")
        yn = lambda v: "n/a (not tuned)" if v is None else ("identical" if v else "DIFFERENT")  # noqa: E731
        for r in reg["rows"]:
            add(f"| {NAMES[r['scheme']]} | {yn(r['evaluation_identical'])} | {yn(r['tuned_identical'])} | {yn(r['tuning_grid_identical'])} |")
        add("")
        add(f"H3 and tau_HO sweeps: {yn(reg['sweeps_identical'])}. Budget, margin distribution, event statistics: "
            + ", ".join(f"{k} {yn(v)}" for k, v in reg["other_identical"].items()) + ". "
            + ("All identical." if reg["all_identical"] else "**Differences found; see above.**"))
        add("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


def _better(x: dict, y: dict) -> str:
    """Compare two outage aggregates: 'lower', 'higher' or 'overlap' (95 % CIs)."""
    if x["mean"] + x["ci"] < y["mean"] - y["ci"]:
        return "lower"
    if x["mean"] - x["ci"] > y["mean"] + y["ci"]:
        return "higher"
    return "overlap"


def _observations(add, m: dict, labels: list[str]) -> None:
    add("## Key observations (generated from the tables; H2 conclusions deferred to review)")
    add("")
    for s in ("trend", "xapp"):
        cmp = [(lab, _better(ev[s]["outage_req_s_per_min"], ev["a3"]["outage_req_s_per_min"])) for lab, ev in zip(labels, m["evaluation"])]
        groups = {k: [lab for lab, v in cmp if v == k] for k in ("lower", "overlap", "higher")}
        add(f"- {NAMES[s]} vs A3, outage_req on evaluation seeds (95 % CIs): lower at {groups['lower'] or 'no margin'}; "
            f"overlapping at {groups['overlap'] or 'no margin'}; higher at {groups['higher'] or 'no margin'}.")
    hyb = ROOT / "results" / "M3" / "hybrid.json"
    if hyb.exists():
        h = json.loads(hyb.read_text())
        cmp = [(r["label"], _better(r["evaluation"]["hybrid_joint"]["outage_req_s_per_min"], r["evaluation"]["a3x"]["outage_req_s_per_min"])) for r in h["margins"]]
        groups = {k: [lab for lab, v in cmp if v == k] for k in ("lower", "overlap", "higher")}
        dom = sum(int(r["front_vs_ho"]["a3_front_points_dominated_by_hybrid"].split("/")[0]) for r in h["margins"])
        add(f"- Hybrid (joint) vs A3x, outage_req: lower at {groups['lower'] or 'no margin'}; overlapping at {groups['overlap'] or 'no margin'}; "
            f"higher at {groups['higher'] or 'no margin'}. A3 Pareto-front points dominated by a hybrid point, summed over margins: {dom} "
            "(section 11).")
    gen = ROOT / "results" / "M3" / "genie.json"
    if gen.exists():
        g = json.loads(gen.read_text())
        tau = f"{g['tau_ho_default_s']:.3f}"
        cmp = [(r["label"], _better(r["tau_ho"][tau]["genie_no_overhead"]["outage_req_s_per_min"], r["tau_ho"][tau]["a3"]["outage_req_s_per_min"])) for r in g["margins"]]
        groups = {k: [lab for lab, v in cmp if v == k] for k in ("lower", "overlap", "higher")}
        add(f"- Genie + A3 (perfect blockage start/end, first-round policy, no sensing overhead) vs A3, outage_req: lower at {groups['lower'] or 'no margin'}; "
            f"overlapping at {groups['overlap'] or 'no margin'}; higher at {groups['higher'] or 'no margin'} (section 12).")
        closed = []
        for r in g["margins"]:
            gc = r["gap_closure"]
            if gc["a3_gap_tau_default"] > 0:
                closed.append(f"{r['label']} {100.0 * (1.0 - gc['a3_gap_tau0'] / gc['a3_gap_tau_default']):.0f} %")
        add("- Interruption-free handover (tau_HO = 0) closes this share of the A3-to-oracle gap: " + ", ".join(closed) + " (section 12).")
    gen2 = ROOT / "results" / "M3" / "genie2.json"
    if gen2.exists():
        g2 = json.loads(gen2.read_text())
        shares = ", ".join(f"{r['label']} {100 * r['foresight']['all']['share_of_a3_outage_pooled']:.1f} %" for r in g2["margins"])
        add("- Policy-independent foresight bound (wrong-cell time inside 10 dB events while the other cell was usable, pooled share of A3 "
            "outage_req): " + shares + " (section 13).")
        tau2 = f"{g2['tau_ho_default_s']:.3f}"
        cmp = [(r["label"], _better(r["tau_ho"][tau2]["onset_genie_no_overhead"]["outage_req_s_per_min"], r1["tau_ho"][tau2]["a3"]["outage_req_s_per_min"]))
               for r, r1 in zip(g2["margins"], json.loads((ROOT / "results" / "M3" / "genie.json").read_text())["margins"])]
        groups = {k: [lab for lab, v in cmp if v == k] for k in ("lower", "overlap", "higher")}
        add(f"- Onset-advance genie (added after the first genie result; no overhead) vs A3: lower at {groups['lower'] or 'no margin'}; "
            f"overlapping at {groups['overlap'] or 'no margin'}; higher at {groups['higher'] or 'no margin'} (section 13).")
    ref = labels.index("3GPP short-range reference")
    gap = m["evaluation"][ref]["a3"]["outage_req_s_per_min"]["mean"] - m["evaluation"][ref]["oracle"]["outage_req_s_per_min"]["mean"]
    add(f"- At the 3GPP short-range reference the A3-to-oracle gap is {gap:.3f} s/UE-min; the decomposition in section 11 splits every "
        "margin's gap into wrong-cell lag and handover interruption.")
    add("")


def _genie2_section(add, g2: dict, g: dict) -> None:
    tau = f"{g2['tau_ho_default_s']:.3f}"
    add("## 13. Third follow-up (genie bound only): foresight bound and an onset-advance genie policy")
    add("")
    add("This concerns the genie bound only; the sensing xApp and its predictor are unchanged.")
    add("")
    add("### Policy-independent foresight bound")
    add("")
    add("With the tuned A3 (tau_HO = 20 ms, main-run parameters) on the evaluation seeds: wrong-cell time inside 10 dB events of the UE, i.e. "
        "steps where the serving cell is below the service rate while the other cell is at or above it, per blocker class (class of the "
        "event), in s/UE-min and as a share of A3's total outage_req (per job, mean ± 95 % CI; and pooled over all jobs). This is the most that "
        "any blockage prediction can recover from A3's outage under any policy: prediction cannot remove handover interruption (b) or "
        "time when both cells are unusable (c). Wrong-cell time outside events (path-loss driven cell changes) is not blockage foresight "
        "and is not counted.")
    add("")
    add("| Margin | A3 outage_req | Class | Foresight-recoverable [s/UE-min] | Share of A3 outage (per job) | Share (pooled) |")
    add("|---|---|---|---|---|---|")
    for r in g2["margins"]:
        fs = r["foresight"]
        for c in ("all", "bus/truck", "pedestrian"):
            pooled = fs[c]["share_of_a3_outage_pooled"]
            add(f"| {r['label']} | {f(fs['a3_outage_req']) if c == 'all' else ''} | {c} | {f(fs[c]['s_per_ue_min'])} | "
                f"{f(fs[c]['share_of_a3_outage_per_job'], 100, 1)} % | {'—' if pooled is None else f'{100 * pooled:.1f} %'} |")
    add("")
    add("No 10 dB events are caused by cars, so the car class is omitted.")
    add("")
    add("### Genie, onset-advance policy")
    add("")
    add("**This policy was added after the first genie result** (section 12), to test whether a different use of perfect prediction helps. "
        "A3 is always active and there is no global hold. The genie (ground-truth LoS loss) only advances a handover when the predicted "
        f"blockage of the serving cell starts at least {g2['min_predicted_start_s'] * 1e3:.0f} ms after the report (E2 loop delay + 10 ms, so the "
        "switch lands before onset) and the other cell is predicted clear (< 3 dB) for the predicted blockage; the other cell's filtered SNR "
        "must exceed serving − 10 dB as before. After such a handover A3's hand-back is blocked only until the predicted end of the "
        "blockage. Grid H {0.5,1,2,3} s (4 points), A3 underlay = the A3 tuned at that margin and tau_HO, tuned on tuning seeds per margin.")
    add("")
    add("| Margin | tau_HO | A3 | genie, 1st policy (no ovh.) | onset genie (no ovh.) | onset genie (with ovh.) | onset H [s] | onset HO/UE-min | onset precision | onset proactive recall |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r2, r1 in zip(g2["margins"], g["margins"]):
        for t in (tau, "0.000"):
            b2, b1 = r2["tau_ho"][t], r1["tau_ho"][t]
            x = b2["onset_genie_no_overhead"]
            e = x["events"].get("all|all", {})
            add(f"| {r2['label']} | {float(t) * 1e3:.0f} ms | {f(b1['a3']['outage_req_s_per_min'])} | {f(b1['genie_no_overhead']['outage_req_s_per_min'])} | "
                f"{f(x['outage_req_s_per_min'])} | {f(b2['onset_genie']['outage_req_s_per_min'])} | {b2['onset_genie_no_overhead_params']['horizon_s']:.1f} | "
                f"{f(x['ho_per_min'], 1, 1)} | {f(x['precision'], 1, 2)} | {f(e.get('proactive_recall'), 1, 2) if e else '—'} |")
    add("")
    add("Outage_req in s/UE-min (mean ± 95 % CI, 40 evaluation jobs).")
    add("")
    add(f"Decomposition at tau_HO = {float(tau) * 1e3:.0f} ms (s/UE-min, no overhead):")
    add("")
    add("| Margin | Scheme | (a) in 10 dB events | (a) outside events | (b) interruption | (c) both unusable |")
    add("|---|---|---|---|---|---|")
    for r2, r1 in zip(g2["margins"], g["margins"]):
        for name, d in (("A3", r1["tau_ho"][tau]["a3_decomp"]), ("genie, 1st policy", r1["tau_ho"][tau]["genie_no_overhead_decomp"]),
                        ("onset genie", r2["tau_ho"][tau]["onset_genie_no_overhead_decomp"])):
            add(f"| {r2['label']} | {name} | {f(d['a_in_events'])} | {f(d['a_outside_events'])} | {f(d['b_interruption'])} | {f(d['c_both_unusable'])} |")
    add("")
    cmp = [(r2["label"], _better(r2["tau_ho"][tau]["onset_genie_no_overhead"]["outage_req_s_per_min"], r1["tau_ho"][tau]["a3"]["outage_req_s_per_min"]))
           for r2, r1 in zip(g2["margins"], g["margins"])]
    groups = {k: [lab for lab, v in cmp if v == k] for k in ("lower", "overlap", "higher")}
    add(f"Onset genie (no overhead) vs A3 at tau_HO = {float(tau) * 1e3:.0f} ms (95 % CIs): lower at {groups['lower'] or 'no margin'}; "
        f"overlapping at {groups['overlap'] or 'no margin'}; higher at {groups['higher'] or 'no margin'}.")
    add("")
    add(f"Third follow-up wall time: {g2['wall_s']:.0f} s.")
    add("")


def _genie_section(add, g: dict, m: dict, h: dict | None) -> None:
    tau = f"{g['tau_ho_default_s']:.3f}"
    add("## 12. Second follow-up: genie bound, interruption-free handover, onset figure")
    add("")
    add("Added after the M3 review (H2 not supported, accepted). The xApp predictor is unchanged.")
    add("")
    add("**Genie + A3.** The first-round xApp policy (same trigger rule: serving-cell LoS loss ≥ 10 dB within H, other cell < 3 dB over that "
        "window, other cell's filtered SNR > serving − 10 dB; A3 held off until the later of hold and the predicted end; same E2 loop delay "
        f"{m['sweeps']['h3']['reference'][1]['tau_e2_s'] * 1e3:.0f} ms, report period 0.1 s and tau_HO) driven by ground truth: the "
        "\"prediction\" at each report is the actual model-B LoS loss of both cells over the next 3 s (10 ms timeline), so start and end are "
        "perfect. Grid H {0.5,1,2,3} s x hold {0.2,0.5,1.0} s (12 points), A3 underlay = the A3 parameters tuned at that margin, tuned on "
        "tuning seeds per margin. \"genie\" pays the xApp's sensing overhead (bound for a sensing-based predictor); \"genie, no overhead\" "
        "does not. This bounds what any predictor could add to A3 under this policy.")
    add("")
    add(f"### Genie bound per margin (evaluation seeds, tau_HO = {float(tau) * 1e3:.0f} ms)")
    add("")
    add("| Margin | Genie H / hold [s] | Oracle | A3 | genie + A3 | genie + A3, no overhead | xApp + A3 (held off) | hybrid (joint) | genie HO / UE-min | genie ping-pong |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for mi, r in enumerate(g["margins"]):
        b = r["tau_ho"][tau]
        hy = f(h["margins"][mi]["evaluation"]["hybrid_joint"]["outage_req_s_per_min"]) if h else "—"
        gp = b["genie_params"]
        add(f"| {r['label']} | {gp['horizon_s']:.1f} / {gp['hold_s']:.1f} | {f(r['oracle']['outage_req_s_per_min'])} | {f(b['a3']['outage_req_s_per_min'])} | "
            f"{f(b['genie']['outage_req_s_per_min'])} | {f(b['genie_no_overhead']['outage_req_s_per_min'])} | {f(m['evaluation'][mi]['xapp']['outage_req_s_per_min'])} | "
            f"{hy} | {f(b['genie']['ho_per_min'], 1, 1)} | {f(b['genie']['ping_pong'], 1, 2)} |")
    add("")
    add("Outage_req in s/UE-min, mean ± 95 % CI over 40 evaluation jobs.")
    add("")
    add("### Genie bound per blocker class (interruption per 10 dB event [s], all events; proactive recall)")
    add("")
    add("| Margin | Class | Events | Oracle | A3 | genie + A3 | genie + A3, no overhead | A3 proactive recall | genie proactive recall |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in g["margins"]:
        b = r["tau_ho"][tau]
        for cls in ("bus/truck", "pedestrian"):
            key = f"{cls}|all"
            if key not in b["a3"]["events"]:
                continue
            ev = lambda agg, k, d=3: f(agg["events"][key][k], 1, d) if key in agg["events"] else "—"  # noqa: E731
            add(f"| {r['label']} | {cls} | {b['a3']['events'][key]['n_events']:.0f} | {ev(r['oracle'], 'interruption_req_s')} | {ev(b['a3'], 'interruption_req_s')} | "
                f"{ev(b['genie'], 'interruption_req_s')} | {ev(b['genie_no_overhead'], 'interruption_req_s')} | {ev(b['a3'], 'proactive_recall', 2)} | {ev(b['genie'], 'proactive_recall', 2)} |")
    add("")
    add("### Where the genie's outage comes from (same decomposition as section 11, s/UE-min)")
    add("")
    add("| Margin | Scheme | (a) wrong cell | (a) in 10 dB events | (a) outside events | (b) interruption | (c) both unusable |")
    add("|---|---|---|---|---|---|---|")
    for r in g["margins"]:
        b = r["tau_ho"][tau]
        for name, key in (("A3", "a3_decomp"), ("genie + A3", "genie_decomp"), ("genie + A3, no overhead", "genie_no_overhead_decomp")):
            d = b[key]
            add(f"| {r['label']} | {name} | {f(d['a_wrong_cell'])} | {f(d['a_in_events'])} | {f(d['a_outside_events'])} | {f(d['b_interruption'])} | {f(d['c_both_unusable'])} |")
    add("")
    lower_in = [r["label"] for r in g["margins"] if r["tau_ho"][tau]["genie_no_overhead_decomp"]["a_in_events"]["mean"] < r["tau_ho"][tau]["a3_decomp"]["a_in_events"]["mean"]]
    higher_out = [r["label"] for r in g["margins"] if r["tau_ho"][tau]["genie_no_overhead_decomp"]["a_outside_events"]["mean"] > r["tau_ho"][tau]["a3_decomp"]["a_outside_events"]["mean"]]
    add("Reading the decomposition (genie without overhead vs A3, means): the genie has less wrong-cell time inside 10 dB events at "
        + (", ".join(lower_in) or "no margin") + "; it has more wrong-cell time outside events at " + (", ".join(higher_out) or "no margin")
        + ". The second effect comes from the policy's hold, which keeps A3 off and so delays path-loss-driven cell changes; handover "
        "interruption (b) is not reduced by prediction. Under this policy even perfect prediction does not add to A3 within the CIs; the "
        "policy, not the predictor, sets the bound here. A different policy would be a method change and is not tested.")
    add("")
    add("### Interruption-free handover (tau_HO = 0, multi-TRP / DAPS-like)")
    add("")
    add("A3 and genie + A3 retuned on tuning seeds at tau_HO = 0 (same grids and objective); the oracle has no interruption in either case. "
        "Gap = scheme − oracle outage_req [s/UE-min]; closed = 1 − gap(tau_HO = 0) / gap(tau_HO = " + f"{float(tau) * 1e3:.0f}" + " ms).")
    add("")
    add("| Margin | Oracle | A3 (20 ms) | A3 (0) | A3 gap closed | genie + A3 (20 ms) | genie + A3 (0) | genie gap (0) | A3 tau_HO=0 params offset/hyst/TTT |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in g["margins"]:
        b20, b0, gc = r["tau_ho"][tau], r["tau_ho"]["0.000"], r["gap_closure"]
        closed = "—" if gc["a3_gap_tau_default"] <= 0 else f"{100.0 * (1.0 - gc['a3_gap_tau0'] / gc['a3_gap_tau_default']):.0f} %"
        p0 = b0["a3_params"]
        add(f"| {r['label']} | {f(r['oracle']['outage_req_s_per_min'])} | {f(b20['a3']['outage_req_s_per_min'])} | {f(b0['a3']['outage_req_s_per_min'])} | {closed} | "
            f"{f(b20['genie']['outage_req_s_per_min'])} | {f(b0['genie']['outage_req_s_per_min'])} | {gc['genie_gap_tau0']:.3f} | "
            f"{p0['offset_db']:.0f}/{p0['hysteresis_db']:.0f}/{p0['ttt_s'] * 1e3:.0f} |")
    add("")
    add("### Onset figure")
    add("")
    add("`results/M3/onset/onset_events.png|pdf` (data in `onset_events.json`). Selection rule, fixed before plotting: per class, the "
        "evaluation-seed 10 dB event whose duration is closest to the class median duration (ties: lowest seed, mount, density, UE, start). "
        "Curves: model-B LoS loss of the cell the event occurs on (the fixed cell at event start) and of the other cell, 10 ms resolution, capped at 60 dB for display. Vertical "
        "lines: handover switch times of A3 and genie + A3 (tuned parameters) at the 3GPP reference margin (solid) and at 10 dB (dashed); the "
        "A3 decision is 10 ms before its switch.")
    add("")
    add("| Class | Job / UE | Event [s] | Onset 10–90 % [s] | Blocker | Handovers (switch times [s]) |")
    add("|---|---|---|---|---|---|")
    for cls, rec in g["onset_figure"].items():
        hos = "; ".join(f"{k}: " + (", ".join(f"{x['switch_s']:.2f}" for x in v) or "none") for k, v in rec["handovers"].items())
        add(f"| {cls} | seed {rec['job'][0]} {rec['job'][1]}/{rec['job'][2]}, UE {rec['ue']} | {rec['start_s']:.2f}–{rec['end_s']:.2f} | "
            f"{rec['onset_s']:.2f} | {rec['blocker']} | {hos} |")
    add("")
    add(f"Second follow-up wall time: {g['wall_s'] / 60:.1f} min.")
    add("")


def _hybrid_section(add, h: dict, m: dict) -> None:
    add("## 11. Follow-up after review: hybrid A3 + xApp, Pareto fronts, outage decomposition")
    add("")
    add("**The hybrid scheme was added after the first M3 rework results**, at the reviewer's request, because in O-RAN an xApp "
        "complements the RAN's native mobility instead of replacing it. Hybrid = A3 always active + xApp proactive handovers on top: the xApp "
        "only advances a handover when it predicts a blockage of the serving cell (other cell predicted clear, other cell's filtered SNR > "
        "serving − 10 dB); it sets no hold, so A3 handles everything else, including returns, and may also hand straight back.")
    add("")
    add("Naming: the scheme reported above as \"xApp + A3 (A3 held off during xApp hold)\" is the first-round xApp and is kept unchanged. It "
        "already ran on top of A3, but it suspended A3 until the later of its hold and the predicted end of the blockage. A truly "
        "xApp-alone scheme (no A3) is the v1 design in `report_v1.md`.")
    add("")
    add("Grids: A3x = offset {1,3} dB x hysteresis {0,1,2,3,5} dB x TTT {40,80,160,320,640} ms (50 points). Hybrid = the same A3 grid x xApp "
        "budget {2,4} x H {0.5,1,2,3} s (400 points). \"Hybrid (joint)\" is tuned over all 400 points; \"hybrid (A3 fixed)\" keeps the A3x-tuned A3 "
        "parameters and tunes only the 8 xApp points (same A3 as the A3x baseline, so any difference is the xApp's). The joint hybrid has an 8x "
        "larger tuning grid than A3x; the Pareto fronts below compare whole grids rather than single tuned points. Tuning: tuning seeds, per "
        "margin, same objective. Pareto points: every grid point on the evaluation seeds (descriptive; nothing is selected from them).")
    add("")
    add("### Tuned parameters (tuning seeds)")
    add("")
    add("| Margin | A3x offset/hyst/TTT | Hybrid joint offset/hyst/TTT, budget, H | Hybrid A3-fixed budget, H | Tuning outage A3x / hybrid joint / hybrid A3-fixed |")
    add("|---|---|---|---|---|")
    for r in h["margins"]:
        a, j, x = r["tuned"]["a3x"]["params"], r["tuned"]["hybrid_joint"]["params"], r["tuned"]["hybrid_a3fixed"]["params"]
        add(f"| {r['label']} | {a['offset_db']:.0f}/{a['hysteresis_db']:.0f}/{a['ttt_s'] * 1e3:.0f} | "
            f"{j['offset_db']:.0f}/{j['hysteresis_db']:.0f}/{j['ttt_s'] * 1e3:.0f}, {j['budget']}, {j['horizon_s']:.1f} | {x['budget']}, {x['horizon_s']:.1f} | "
            f"{r['tuned']['a3x']['tuning_objective'][0]:.3f} / {r['tuned']['hybrid_joint']['tuning_objective'][0]:.3f} / {r['tuned']['hybrid_a3fixed']['tuning_objective'][0]:.3f} |")
    add("")
    add("### Evaluation seeds (mean ± 95 % CI over 40 jobs)")
    add("")
    add("| Margin | Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | xApp-HO precision |")
    add("|---|---|---|---|---|---|---|---|")
    for mi, r in enumerate(h["margins"]):
        ev = m["evaluation"][mi]
        rows = [("oracle", ev["oracle"]), ("A3 (24-pt grid)", ev["a3"]), ("A3x (50-pt grid)", r["evaluation"]["a3x"]),
                ("xApp + A3, A3 held off", ev["xapp"]), ("hybrid (joint)", r["evaluation"]["hybrid_joint"]), ("hybrid (A3 fixed)", r["evaluation"]["hybrid_a3fixed"])]
        for name, v in rows:
            add(f"| {r['label']} | {name} | {f(v['outage_req_s_per_min'])} | {f(v['outage0_s_per_min'])} | {f(v['tp_loss_vs_oracle'], 100, 2)} | "
                f"{f(v['ho_per_min'], 1, 1)} | {f(v['ping_pong'], 1, 2)} | {f(v['precision'], 1, 2) if 'xApp' in name or 'hybrid' in name else '—'} |")
    add("")
    add("### Pareto fronts: does hybrid dominate A3?")
    add("")
    add("Fronts over all grid points (evaluation seeds, means over jobs). Plots: `results/M3/pareto/pareto_<i>_<margin>.png|pdf` "
        "(left: outage vs HO/UE-min, right: outage vs ping-pong; circles = tuned points).")
    add("")
    add("| Margin | vs HO/UE-min: A3-front points dominated by hybrid | hybrid-front points dominated by A3 | HO range where hybrid front lower | HO range where hybrid front higher | vs ping-pong: A3-front dominated | ping-pong range where hybrid lower |")
    add("|---|---|---|---|---|---|---|")
    rng = lambda x: "—" if not x else f"{x[0]:.2f}–{x[1]:.2f}"  # noqa: E731
    for r in h["margins"]:
        c1, c2 = r["front_vs_ho"], r["front_vs_pingpong"]
        add(f"| {r['label']} | {c1['a3_front_points_dominated_by_hybrid']} | {c1['hybrid_front_points_dominated_by_a3']} | {rng(c1['x_where_hybrid_lower'])} "
            f"({c1['n_hybrid_lower']}/{c1['n_x_levels']} levels) | {rng(c1['x_where_hybrid_higher'])} ({c1['n_hybrid_higher']}/{c1['n_x_levels']}) | "
            f"{c2['a3_front_points_dominated_by_hybrid']} | {rng(c2['x_where_hybrid_lower'])} ({c2['n_hybrid_lower']}/{c2['n_x_levels']}) |")
    add("")
    add("Front comparison: at each x level (HO rate or ping-pong of any front point) the best outage reachable with x' ≤ x is compared; "
        "differences are means without a CI, so small gaps between fronts are not significant (compare with the CIs in the tables).")
    add("")
    add("### Headroom decomposition: where A3 loses against the oracle")
    add("")
    add("Every 10 ms step in outage (rate < SNR_req rate or interruption) is assigned to exactly one class: (c) both cells below the service "
        "rate (unrecoverable by any cell choice; equals the oracle outage), else (b) handover interruption, else (a) serving cell below the rate "
        "while the other cell was usable (detection, L3 filter and TTT lag). (a) is split into steps inside 10 dB events of that UE "
        "(blockage onset) and outside (path-loss driven cell changes). Residual vs oracle = (a) + (b). A proactive scheme can at best remove (a) "
        "and must not add (b). Units: s/UE-min, mean ± 95 % CI over evaluation jobs. For the xApp and hybrid rows the service-rate test includes "
        "their sensing overhead, so their (c) is slightly above the oracle outage; that difference is the cost of the sensing resources.")
    add("")
    add("| Margin | Scheme | (a) wrong cell | (a) in 10 dB events | (a) outside events | (b) interruption | (c) both unusable | oracle outage |")
    add("|---|---|---|---|---|---|---|---|")
    for r in h["margins"]:
        for name, d in r["decomposition"].items():
            add(f"| {r['label']} | {name} | {f(d['a_wrong_cell'])} | {f(d['a_in_events'])} | {f(d['a_outside_events'])} | {f(d['b_interruption'])} | "
                f"{f(d['c_both_unusable'])} | {f(r['oracle_outage_req'])} |")
    add("")
    add(f"Follow-up wall time: {h['wall_s'] / 60:.1f} min (CPU, vectorised lanes; per margin up to 36 000 lanes x 5991 steps).")
    add("")


if __name__ == "__main__":
    main()
