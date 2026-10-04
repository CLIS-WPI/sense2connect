"""Lead time per class: map-constrained vs unconstrained tracker (paper/figs/fig_leadtime.pdf).

DEFAULT (after review A2): the FINAL detector configuration used for the
paper's Pd/FA/lead macros -- budget 4 FA/CPI, blind clutter, image-method
ghost handling -- for both trackers, from results/M5/tracking.json
(scripts/run_m5_tracking.py). ``--config noghost`` (below) recomputes the
earlier no-ghost comparison and checks it against results/M2/followup.md.

Recomputed from the M2 detection caches (``detections_1024.json``), which are
validated with the stage-scoped provenance in ``sim/sensing/provenance.py``
(sensing sources + sensing config; legacy caches against the M2 commit).
Protocol of ``scripts/run_m2_followup.py``: detector = the budget-4 blind,
no-ghost pick of results/M2/metrics.json; map tracker tuned on the tuning
seeds over the same 108-point grid for bus/truck track Pd (first maximum in
grid order); both trackers replayed on the evaluation seeds with the same
detections. Lead = share of 10 dB LoS events (oru-0, 0.1 s) with a
confirmed track inside the class gate L seconds before onset. Pooled over
mounts, 95 % Wilson intervals. The recomputed counts are compared with the
table in results/M2/followup.md.

Writes results/M5/leadtime.json and paper/figs/fig_leadtime.pdf.
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LEADS = (0.1, 0.3, 0.5, 1.0)
GRID = list(itertools.product((4.0, 6.0, 10.0), (2, 4, 8), (0.25, 1.0, 4.0), (1.5, 2.5), (2.0, 4.0)))
ROW = re.compile(r"^\| (unconstrained|map) \| (bus/truck|pedestrian|car) \| (lamppost|facade) \| (\d+) \| (\d+)/\d+ \| (\d+)/\d+ \| (\d+)/\d+ \| (\d+)/\d+ \|$")
_CTX: dict[str, Any] = {}


def load_case(mount: str, density: str, seed: int) -> dict[str, Any]:
    from sim.sensing.provenance import require_detection_stage

    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    payload = json.loads((directory / "detections_1024.json").read_text(encoding="utf-8"))
    payload["stage_check"] = require_detection_stage(payload)
    events = json.loads((directory / "events.json").read_text(encoding="utf-8"))
    payload["events"] = events["events"] if isinstance(events, dict) else events
    payload.update({"mount": mount, "density": density})
    return payload


def _setup() -> dict[str, Any]:
    import run_m2_followup as F
    from sim.scenes.config import load_yaml

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    metrics = json.loads((ROOT / "results" / "M2" / "metrics.json").read_text(encoding="utf-8"))
    point = F._budget_point(metrics["picks"])
    if point is None:
        raise SystemExit("budget-4 blind/no-ghost detector pick not found in results/M2/metrics.json")
    params = dict(point["params"])
    params["ghost"] = False
    params["clutter"] = "blind"
    return {"F": F, "raw": raw, "params": params, "point": point}


def _init(jobs: list[tuple[str, str, int]]) -> None:
    import torch

    torch.set_num_threads(1)
    _CTX.update(_setup())
    _CTX["cases"] = [load_case(*job) for job in jobs]


def _score_combo(combo: tuple) -> tuple[tuple, float]:
    from sim.sensing.metrics import empty_score, scaled_gates, summarize

    F, raw, params = _CTX["F"], _CTX["raw"], _CTX["params"]
    association_m, coast, process_q, gate, leave_m = combo
    tuned = {"association_m": association_m, "coast": coast, "process_q": process_q, "lane_gate_m": gate, "sidewalk_gate_m": gate, "leave_m": leave_m}
    total = empty_score()
    for case in _CTX["cases"]:
        tracker = F._map_tracker(params, raw["sensing_radar"], case["radar_position_m"], list(raw["lanes"]), list(raw["sidewalks"]), tuned)
        part, _ = F._replay_tracker(case, tracker, params, raw["sensing_radar"], raw["blocker_kinds"], scaled_gates(1.0))
        F._add(total, part)
    pd = summarize(total)["classes"]["bus/truck"]["track_pd"] or -1.0
    return combo, float(pd)


def _eval_case(item: tuple) -> dict[str, Any]:
    from sim.sensing.metrics import scaled_gates

    job, tuned = item
    F, raw, params = _CTX["F"], _CTX["raw"], _CTX["params"]
    case = load_case(*job)
    gates = scaled_gates(1.0)
    spec = raw["sensing_radar"]
    out = {"mount": case["mount"], "stage_check": case["stage_check"]}
    for name, tracker in (("unconstrained", F._unconstrained(params, spec, case["radar_position_m"])),
                          ("map", F._map_tracker(params, spec, case["radar_position_m"], list(raw["lanes"]), list(raw["sidewalks"]), tuned))):
        _, tracks = F._replay_tracker(case, tracker, params, spec, raw["blocker_kinds"], gates)
        out[name] = F._lead_counts(case, tracks, gates)
    return out


def recompute(workers: int) -> dict[str, Any]:
    from sim.scenes.config import load_yaml

    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds

    seeds = load_seeds()
    tune_jobs = [(m, d, int(s)) for s in seeds["tuning"] for m in ("lamppost", "facade") for d in ("low", "high")]
    eval_jobs = [(m, d, int(s)) for s in seeds["evaluation"] for m in ("lamppost", "facade") for d in ("low", "high")]
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=(tune_jobs,)) as pool:
        scores = dict(pool.map(_score_combo, GRID, chunksize=1))
    best = max(GRID, key=lambda c: (scores[c], -GRID.index(c)))  # first maximum in grid order, as _tune_map
    tuned = {"association_m": best[0], "coast": best[1], "process_q": best[2], "lane_gate_m": best[3], "sidewalk_gate_m": best[3], "leave_m": best[4]}
    t1 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=([],)) as pool:
        parts = pool.map(_eval_case, [(job, tuned) for job in eval_jobs], chunksize=1)
    table: dict[str, Any] = {}
    for part in parts:
        for tracker in ("unconstrained", "map"):
            for cls, row in part[tracker].items():
                e = table.setdefault(tracker, {}).setdefault(cls, {"n": 0, "hits": [0] * len(LEADS), "by_mount": {}})
                e["n"] += row["n"]
                hits = [row["hits"][str(lead)] for lead in LEADS]
                e["hits"] = [a + b for a, b in zip(e["hits"], hits)]
                bm = e["by_mount"].setdefault(part["mount"], {"n": 0, "hits": [0] * len(LEADS)})
                bm["n"] += row["n"]
                bm["hits"] = [a + b for a, b in zip(bm["hits"], hits)]
    checks = sorted({p["stage_check"] for p in parts})
    return {"tuned_map": tuned, "tuning_bus_truck_pd": scores[best], "table": table, "stage_checks": checks,
            "tune_wall_s": t1 - t0, "eval_wall_s": time.perf_counter() - t1}


def compare_followup(table: dict[str, Any]) -> list[str]:
    text = (ROOT / "results" / "M2" / "followup.md").read_text(encoding="utf-8")
    section = text.split("## Lead time, same detector", 1)[1].split("\n## ", 1)[0]
    diffs = []
    for line in section.splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        tracker, cls, mount, n = m.group(1), m.group(2), m.group(3), int(m.group(4))
        hits = [int(m.group(5 + i)) for i in range(4)]
        got = table.get(tracker, {}).get(cls, {}).get("by_mount", {}).get(mount)
        if got is None or got["n"] != n or got["hits"] != hits:
            diffs.append(f"{tracker} {cls} {mount}: followup {n} {hits} vs recomputed {got}")
    return diffs


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


def plot(table: dict[str, Any]) -> dict[str, Any]:
    from figstyle import COLUMN_IN, save, setup

    plt = setup()
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.1))
    style = {("map", "bus/truck"): ("#1f77b4", "-", "o"), ("unconstrained", "bus/truck"): ("#1f77b4", "--", "o"),
             ("map", "pedestrian"): ("#d62728", "-", "s"), ("unconstrained", "pedestrian"): ("#d62728", "--", "s")}
    series = {}
    for (tracker, cls), (color, ls, marker) in style.items():
        e = table[tracker][cls]
        pts = [wilson(k, e["n"]) for k in e["hits"]]
        y = [p for p, _, _ in pts]
        ax.errorbar(LEADS, y, yerr=[[p - lo for p, lo, _ in pts], [hi - p for p, _, hi in pts]], color=color, ls=ls, marker=marker,
                    markerfacecolor=color if tracker == "map" else "white", capsize=1.5, elinewidth=0.6,
                    label=f"{cls}, {'map-constrained' if tracker == 'map' else 'unconstrained'} (n={e['n']})")
        series[f"{tracker}|{cls}"] = {"n": e["n"], "hits": e["hits"], "share": y, "wilson95": [[lo, hi] for _, lo, hi in pts], "by_mount": e["by_mount"]}
    ax.set_xlabel("Lead before 10 dB onset [s]")
    ax.set_ylabel("Events with confirmed track")
    ax.set_xticks(LEADS)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower left", frameon=False, fontsize=6.3)
    print(f"wrote {save(fig, 'fig_leadtime.pdf')}")
    return series


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from results/M5/leadtime.json (no-ghost config)")
    parser.add_argument("--config", choices=("final", "noghost"), default="final")
    args = parser.parse_args()
    if args.config == "final":
        trk = json.loads((ROOT / "results" / "M5" / "tracking.json").read_text())
        table: dict[str, Any] = {}
        for tracker in ("unconstrained", "map"):
            leads = trk["variants"][f"image|{tracker}"]["leads"]
            for mount, per_cls in leads.items():
                for cls, row in per_cls.items():
                    e = table.setdefault(tracker, {}).setdefault(cls, {"n": 0, "hits": [0] * len(LEADS), "by_mount": {}})
                    hits = [row["hits"][str(L)] for L in LEADS]
                    e["n"] += row["n"]
                    e["hits"] = [a + b for a, b in zip(e["hits"], hits)]
                    e["by_mount"][mount] = {"n": row["n"], "hits": hits}
        series = plot(table)
        dest = ROOT / "results" / "M5" / "leadtime_final.json"
        dest.write_text(json.dumps({"config": "final: budget 4 FA/CPI, blind clutter, image-method ghost handling; trackers: unconstrained EKF and map-constrained (MAP_TUNED)",
                                    "source": "results/M5/tracking.json", "leads_s": LEADS, "series": series}, indent=1) + "\n")
        print(f"wrote {dest}")
        return
    dest = ROOT / "results" / "M5" / "leadtime.json"
    if args.plot_only:
        data = json.loads(dest.read_text())
        plot(data["table"])
        return
    res = recompute(max(1, min(4, args.workers)))
    diffs = compare_followup(res["table"])
    res["followup_match"] = "identical to results/M2/followup.md" if not diffs else diffs
    res["series"] = plot(res["table"])
    res["leads_s"] = LEADS
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("tuned_map", "tuning_bus_truck_pd", "stage_checks", "followup_match", "tune_wall_s", "eval_wall_s")}, indent=1, default=str))


if __name__ == "__main__":
    main()
