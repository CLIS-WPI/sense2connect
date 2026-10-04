"""M3 follow-up: hybrid (A3 always on + xApp advance), Pareto fronts, outage decomposition.

Added after the first M3 rework results, at the reviewer's request: in O-RAN
an xApp complements the RAN's native mobility, it does not replace it.

- A3 extended grid ("A3x"): offset {1,3} x hysteresis {0,1,2,3,5} dB x TTT
  {40,80,160,320,640} ms (50 points).
- hybrid: the same A3 grid x xApp budget {2,4} x H {0.5,1,2,3} s (400
  points). xApp handovers set no hold, so A3 stays active the whole time
  and handles every return (it may also hand straight back).
  "hybrid (joint)" is tuned over all 400 points; "hybrid (A3 fixed)" keeps
  the A3x-tuned A3 parameters and tunes only the 8 xApp points.
- Tuning: tuning seeds, per margin, objective = outage at SNR_req incl.
  interruption, ties -> fewer HO/UE-min (as in run_m3.py).
- Pareto: every grid point evaluated on the evaluation seeds (descriptive;
  no parameter is chosen from these points).
- Outage decomposition per step in outage: (c) both cells below the service
  rate (unrecoverable; equals the oracle outage), else (b) handover
  interruption, else (a) serving cell below the rate while the other cell
  was usable (detection / L3 filter / TTT lag). (a) is split into steps
  inside 10 dB events of that UE and outside.

Reads the timeline caches and results/M3/metrics.json (tuned A3/xApp
parameters of the main run). Writes results/M3/hybrid.json and
results/M3/pareto/*.png|pdf.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_m3 as R  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.metrics import lane_summary  # noqa: E402
from xapp.schemes import REASON, simulate  # noqa: E402

A3X = {"offset_db": [1.0, 3.0], "hysteresis_db": [0.0, 1.0, 2.0, 3.0, 5.0], "ttt_s": [0.04, 0.08, 0.16, 0.32, 0.64]}
XH = {"budget": [2, 4], "horizon_s": [0.5, 1.0, 2.0, 3.0]}
MAX_LANES = 6000


def decompose(lanes, sim, events_by_lane, bw, rate_req, dt) -> list[dict[str, float]]:
    """Per-lane outage split [s/UE-min] into (a) wrong cell, (b) interruption, (c) both cells unusable."""
    snr = lanes.snr_db
    se = np.minimum(np.log2(1.0 + 10.0 ** (snr / 10.0)), MAX_NR_SE)
    usable = bw * (1.0 - lanes.overhead)[:, None, None] * se >= rate_req  # [L, T, 2]
    both_bad = ~usable.any(-1)
    out = sim["outage_req"]
    c = out & both_bad
    b = out & ~both_bad & sim["interrupted"]
    a = out & ~both_bad & ~sim["interrupted"]
    minutes = snr.shape[1] * dt / 60.0
    rows = []
    for lane in range(snr.shape[0]):
        in_ev = np.zeros(snr.shape[1], dtype=bool)
        for ev in events_by_lane[lane]:
            in_ev[ev["start_k"] : ev["end_k"] + 1] = True
        rows.append(
            {
                "a_wrong_cell": float(a[lane].sum() * dt / minutes),
                "a_in_events": float((a[lane] & in_ev).sum() * dt / minutes),
                "a_outside_events": float((a[lane] & ~in_ev).sum() * dt / minutes),
                "b_interruption": float(b[lane].sum() * dt / minutes),
                "c_both_unusable": float(c[lane].sum() * dt / minutes),
            }
        )
    return rows


def run_combos(jobs, combos, scheme, snr_m, built, rw, bw, rate_req, oracle_rates, *, hybrid=False, a3=None) -> list[dict[str, Any]]:
    """Lane summaries (+ decomposition) for every combo, in chunks of at most MAX_LANES lanes."""
    lanes_per_combo = sum(built[j]["data"]["blocked_power"].shape[1] for j in jobs)
    per_chunk = max(1, MAX_LANES // lanes_per_combo)
    rows: list[dict[str, Any]] = []
    proactive = (REASON["xapp"],) if scheme == "xapp" else ()
    for lo in range(0, len(combos), per_chunk):
        chunk = combos[lo : lo + per_chunk]
        lanes, index = R.make_lanes(jobs, chunk, snr_m, built, scheme, a3 or {}, rw, None, hybrid=hybrid)
        sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
        evs = [[ev for ev in built[j]["events"] if ev["ue"] == u] for _, j, u in index]
        dec = decompose(lanes, sim, evs, bw, rate_req, R.DT_COMM)
        for lane, (ci, job, u) in enumerate(index):
            s = lane_summary(sim, lane, evs[lane], oracle_rates[job][:, u], R.DT_COMM, proactive_reasons=proactive)
            s.update({"combo": lo + ci, "job": job, "ue": u, "decomp": dec[lane]})
            rows.append(s)
        del sim, lanes
    return rows


def by_combo(rows: list[dict], n: int) -> list[list[dict]]:
    out: list[list[dict]] = [[] for _ in range(n)]
    for r in rows:
        out[r["combo"]].append(r)
    return out


def best(rows: list[dict], n: int, allowed: list[int] | None = None) -> int:
    obj = R.objective(rows, n)
    cand = allowed if allowed is not None else list(range(n))
    return min(cand, key=lambda i: (obj[i][0], obj[i][1], i))


def point(rows: list[dict]) -> dict[str, Any]:
    agg = R.aggregate(rows)
    return {k: agg[k] for k in ("outage_req_s_per_min", "ho_per_min", "ping_pong", "outage0_s_per_min", "tp_loss_vs_oracle")}


def decomp_summary(rows: list[dict]) -> dict[str, Any]:
    by_job: dict[tuple, list[dict]] = {}
    for r in rows:
        by_job.setdefault(tuple(r["job"]), []).append(r["decomp"])
    keys = rows[0]["decomp"].keys()
    return {k: R.ci95([float(np.mean([d[k] for d in v])) for v in by_job.values()]) for k in keys}


def front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points (minimise both)."""
    idx = []
    for i, (x, y) in enumerate(points):
        if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for j, (x2, y2) in enumerate(points) if j != i):
            idx.append(i)
    return sorted(idx, key=lambda i: points[i][0])


def compare_fronts(a3_pts, hy_pts) -> dict[str, Any]:
    """Share of A3-front points dominated by a hybrid point, and the x-ranges where the hybrid front is lower."""
    fa, fh = front(a3_pts), front(hy_pts)
    dominated = [
        i for i in fa
        if any(hy_pts[j][0] <= a3_pts[i][0] and hy_pts[j][1] <= a3_pts[i][1] and (hy_pts[j][0] < a3_pts[i][0] or hy_pts[j][1] < a3_pts[i][1]) for j in fh)
    ]
    hyb_dominated = [
        j for j in fh
        if any(a3_pts[i][0] <= hy_pts[j][0] and a3_pts[i][1] <= hy_pts[j][1] and (a3_pts[i][0] < hy_pts[j][0] or a3_pts[i][1] < hy_pts[j][1]) for i in fa)
    ]
    xs = sorted({p[0] for p in [a3_pts[i] for i in fa] + [hy_pts[j] for j in fh]})

    def env(pts, f, x):
        ys = [pts[i][1] for i in f if pts[i][0] <= x]
        return min(ys) if ys else float("inf")

    better = [x for x in xs if env(hy_pts, fh, x) < env(a3_pts, fa, x)]
    worse = [x for x in xs if env(hy_pts, fh, x) > env(a3_pts, fa, x)]
    return {
        "a3_front": [a3_pts[i] for i in fa],
        "hybrid_front": [hy_pts[j] for j in fh],
        "a3_front_points_dominated_by_hybrid": f"{len(dominated)}/{len(fa)}",
        "hybrid_front_points_dominated_by_a3": f"{len(hyb_dominated)}/{len(fh)}",
        "x_where_hybrid_lower": [min(better), max(better)] if better else None,
        "x_where_hybrid_higher": [min(worse), max(worse)] if worse else None,
        "n_x_levels": len(xs),
        "n_hybrid_lower": len(better),
        "n_hybrid_higher": len(worse),
    }


def plot(margin_label: str, a3_points, hy_points, a3_tuned, hy_tuned, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, key, xlabel in ((axes[0], "ho_per_min", "Handovers per UE-minute"), (axes[1], "ping_pong", "Ping-pong rate")):
        for pts, color, label, marker in ((hy_points, "tab:orange", "hybrid (A3 + xApp)", "."), (a3_points, "tab:blue", "A3", "o")):
            xy = [(p[key]["mean"], p["outage_req_s_per_min"]["mean"]) for p in pts]
            ax.scatter([x for x, _ in xy], [y for _, y in xy], s=6 if marker == "." else 10, c=color, alpha=0.35, marker=marker, linewidths=0)
            f = front(xy)
            ax.plot([xy[i][0] for i in f], [xy[i][1] for i in f], color=color, lw=1.4, label=f"{label} front")
        for tuned, color in ((a3_tuned, "tab:blue"), (hy_tuned, "tab:orange")):
            ax.scatter([tuned[key]["mean"]], [tuned["outage_req_s_per_min"]["mean"]], s=40, facecolors="none", edgecolors=color, linewidths=1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Outage at SNR_req [s/UE-min]")
        ax.grid(True, lw=0.3, alpha=0.6)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Margin {margin_label} (evaluation seeds; circles = tuned points)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    main_metrics = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    if abs(info["margin_ref_db"] - main_metrics["budget"]["margin_ref_db"]) > 1e-9:
        raise SystemExit("reference margin differs from the main run")
    snr_all = R.all_snr(built, tune_jobs + eval_jobs, rw, bw, info["extra_loss_db"])
    a3_combos = R._grid(A3X)
    x_combos = R._grid(XH)
    hy_combos = [{**a, **x, "hold_s": 0.0} for a in a3_combos for x in x_combos]
    out = {"labels": info["labels"], "points_db": info["points_db"], "grids": {"a3x": A3X, "xapp_hybrid": XH}, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        t0 = time.perf_counter()
        snr_t = {j: snr_all[j][mi] for j in tune_jobs}
        snr_e = {j: snr_all[j][mi] for j in eval_jobs}
        orates_t = R.stateless_rows(tune_jobs, snr_t, built, "oracle", bw, rate_req)[1]
        oracle_e_rows, orates_e = R.stateless_rows(eval_jobs, snr_e, built, "oracle", bw, rate_req)
        # tuning
        a3_t = run_combos(tune_jobs, a3_combos, "a3", snr_t, built, rw, bw, rate_req, orates_t)
        i_a3 = best(a3_t, len(a3_combos))
        hy_t = run_combos(tune_jobs, hy_combos, "xapp", snr_t, built, rw, bw, rate_req, orates_t, hybrid=True)
        i_hy = best(hy_t, len(hy_combos))
        fixed = [k for k, c in enumerate(hy_combos) if all(c[key] == a3_combos[i_a3][key] for key in A3X)]
        i_hf = best(hy_t, len(hy_combos), fixed)
        # evaluation of every grid point
        a3_e = by_combo(run_combos(eval_jobs, a3_combos, "a3", snr_e, built, rw, bw, rate_req, orates_e), len(a3_combos))
        hy_e = by_combo(run_combos(eval_jobs, hy_combos, "xapp", snr_e, built, rw, bw, rate_req, orates_e, hybrid=True), len(hy_combos))
        a3_pts = [point(r) for r in a3_e]
        hy_pts = [point(r) for r in hy_e]
        # main-run schemes for the decomposition (their tuned parameters, same lanes code)
        tuned_main = main_metrics["tuned"][mi]
        a3_main = run_combos(eval_jobs, [tuned_main["a3"]["params"]], "a3", snr_e, built, rw, bw, rate_req, orates_e)
        xp = tuned_main["xapp"]["params"]
        xapp_main = run_combos(eval_jobs, [xp], "xapp", snr_e, built, rw, bw, rate_req, orates_e, a3={k: xp[k] for k in ("offset_db", "hysteresis_db", "ttt_s")})
        evaluation = {
            "a3x": R.aggregate(a3_e[i_a3]),
            "hybrid_joint": R.aggregate(hy_e[i_hy]),
            "hybrid_a3fixed": R.aggregate(hy_e[i_hf]),
        }
        decomposition = {
            "a3 (main, 24-pt grid)": decomp_summary(a3_main),
            "a3x (50-pt grid)": decomp_summary(a3_e[i_a3]),
            "xapp (main, A3 held off)": decomp_summary(xapp_main),
            "hybrid (joint)": decomp_summary(hy_e[i_hy]),
            "hybrid (A3 fixed)": decomp_summary(hy_e[i_hf]),
        }
        oracle_out = R.aggregate(oracle_e_rows)["outage_req_s_per_min"]
        cmp_ho = compare_fronts([(p["ho_per_min"]["mean"], p["outage_req_s_per_min"]["mean"]) for p in a3_pts], [(p["ho_per_min"]["mean"], p["outage_req_s_per_min"]["mean"]) for p in hy_pts])
        cmp_pp = compare_fronts([(p["ping_pong"]["mean"], p["outage_req_s_per_min"]["mean"]) for p in a3_pts], [(p["ping_pong"]["mean"], p["outage_req_s_per_min"]["mean"]) for p in hy_pts])
        safe = lab.replace(" ", "_").replace("(", "").replace(")", "")
        plot(lab, a3_pts, hy_pts, a3_pts[i_a3], hy_pts[i_hy], ROOT / "results" / "M3" / "pareto" / f"pareto_{mi}_{safe}")
        tune_obj = lambda rows, n, i: R.objective(rows, n)[i]  # noqa: E731
        out["margins"].append(
            {
                "label": lab,
                "margin_db": info["points_db"][mi],
                "tuned": {
                    "a3x": {"params": a3_combos[i_a3], "tuning_objective": tune_obj(a3_t, len(a3_combos), i_a3)},
                    "hybrid_joint": {"params": hy_combos[i_hy], "tuning_objective": tune_obj(hy_t, len(hy_combos), i_hy)},
                    "hybrid_a3fixed": {"params": hy_combos[i_hf], "tuning_objective": tune_obj(hy_t, len(hy_combos), i_hf)},
                },
                "evaluation": evaluation,
                "oracle_outage_req": oracle_out,
                "decomposition": decomposition,
                "pareto_points": {"a3x": [{**a3_combos[i], **a3_pts[i]} for i in range(len(a3_combos))], "hybrid": [{**hy_combos[i], **hy_pts[i]} for i in range(len(hy_combos))]},
                "front_vs_ho": cmp_ho,
                "front_vs_pingpong": cmp_pp,
                "wall_s": time.perf_counter() - t0,
            }
        )
        print(
            f"{lab}: A3x {a3_combos[i_a3]} out {evaluation['a3x']['outage_req_s_per_min']['mean']:.3f} | hybrid {hy_combos[i_hy]} out "
            f"{evaluation['hybrid_joint']['outage_req_s_per_min']['mean']:.3f} | hybrid(A3 fixed) out {evaluation['hybrid_a3fixed']['outage_req_s_per_min']['mean']:.3f} | "
            f"A3-front dominated {cmp_ho['a3_front_points_dominated_by_hybrid']} | {time.perf_counter() - t0:.0f} s",
            flush=True,
        )
    out["wall_s"] = time.perf_counter() - clock
    (ROOT / "results" / "M3" / "hybrid.json").write_text(json.dumps(out, indent=1, default=R._json) + "\n", encoding="utf-8")
    print(f"wrote results/M3/hybrid.json in {out['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
