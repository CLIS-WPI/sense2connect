"""M3 follow-up 2: genie bound, interruption-free handover, onset figure.

1. Genie + A3: the first-round xApp policy (trigger table, A3 held off until
   the later of hold and the predicted end, same E2 delay and tau_HO) driven
   by ground truth: the "prediction" at report r is the actual model-B LoS
   loss of every cell at r + tau (10 ms timeline), tau in [0, 3] s. Grid:
   H {0.5,1,2,3} s x hold {0.2,0.5,1.0} s (12 points), A3 underlay = the
   A3 parameters tuned at that margin. Reported with the xApp's sensing
   overhead (bound for sensing-based predictors) and without it.
2. Interruption-free handover: A3, genie + A3 and oracle with tau_HO = 0,
   retuned on tuning seeds (same protocol); gap closure vs tau_HO = 20 ms.
3. Onset figure: per class (bus/truck, pedestrian) the evaluation-seed
   event whose duration is closest to the class median (ties: lowest
   seed, mount, density, UE, start); model-B LoS loss of both cells, with
   the A3 and genie handovers at the 3GPP reference and 10 dB margins.

Writes results/M3/genie.json and results/M3/onset/onset_events.{png,pdf,json}.
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
from run_m3_hybrid import decomp_summary, decompose  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.metrics import lane_summary  # noqa: E402
from xapp.schemes import REASON, simulate  # noqa: E402

GENIE = {"budget": ["genie"], "horizon_s": [0.5, 1.0, 2.0, 3.0], "hold_s": [0.2, 0.5, 1.0]}


def add_genie(built: dict) -> None:
    """Ground-truth 'prediction' [R, U, C, tau] = actual LoS loss at report + tau."""
    for item in built.values():
        d = item["data"]
        n_t = d["los_loss_db"].shape[0]
        steps = np.round(d["taus"] / R.DT_COMM).astype(np.int64)
        n_r = d["pred_2"].shape[0]
        k = np.minimum(np.arange(n_r)[:, None] * int(round(R.DT_SENSE / R.DT_COMM)) + steps[None, :], n_t - 1)  # [R, tau]
        d["pred_genie"] = np.transpose(d["los_loss_db"][k], (0, 2, 3, 1))  # [R, U, C, tau]


def run(jobs, combos, scheme, snr_m, built, rw, bw, rate_req, orates, *, a3, tau_ho, overhead=True) -> list[dict[str, Any]]:
    lanes, index = R.make_lanes(jobs, combos, snr_m, built, scheme, a3, rw, None, tau_ho_s=tau_ho)
    if not overhead:
        lanes.overhead = np.zeros_like(lanes.overhead)
    sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
    evs = [[ev for ev in built[j]["events"] if ev["ue"] == u] for _, j, u in index]
    dec = decompose(lanes, sim, evs, bw, rate_req, R.DT_COMM)
    proactive = (REASON["xapp"],) if scheme == "xapp" else ()
    rows = []
    for lane, (ci, job, u) in enumerate(index):
        s = lane_summary(sim, lane, evs[lane], orates[job][:, u], R.DT_COMM, proactive_reasons=proactive)
        s.update({"combo": ci, "job": job, "ue": u, "decomp": dec[lane], "ho_steps": sim["handovers"]["step"][sim["handovers"]["lane"] == lane].tolist(),
                  "ho_reason": sim["handovers"]["reason"][sim["handovers"]["lane"] == lane].tolist()})
        rows.append(s)
    return rows


def tune(jobs, combos, scheme, snr_m, built, rw, bw, rate_req, orates, **kw) -> tuple[int, tuple[float, float]]:
    rows = run(jobs, combos, scheme, snr_m, built, rw, bw, rate_req, orates, **kw)
    obj = R.objective(rows, len(combos))
    i = min(range(len(combos)), key=lambda c: (obj[c][0], obj[c][1], c))
    return i, obj[i]


def pick_events(built: dict, eval_jobs: list) -> dict[str, dict[str, Any]]:
    chosen = {}
    for cls in ("bus/truck", "pedestrian"):
        pool = [(job, ev) for job in eval_jobs for ev in built[job]["events"] if ev["class"] == cls]
        dur = np.array([ev["end_s"] - ev["start_s"] + R.DT_COMM for _, ev in pool])
        med = float(np.median(dur))
        job, ev = min(pool, key=lambda item: (abs(item[1]["end_s"] - item[1]["start_s"] + R.DT_COMM - med), item[0], item[1]["ue"], item[1]["start_k"]))
        chosen[cls] = {"job": list(job), "event": ev, "class_median_duration_s": med, "duration_s": ev["end_s"] - ev["start_s"] + R.DT_COMM}
    return chosen


def onset_figure(chosen, built, snr_all, margins_idx, tuned_a3, tuned_genie, rw, bw, rate_req, out: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    record = {}
    for ax, (cls, item) in zip(axes, chosen.items()):
        job = tuple(item["job"])
        ev = item["event"]
        u = ev["ue"]
        d = built[job]["data"]
        lo = max(0, ev["start_k"] - 300)
        hi = min(d["los_loss_db"].shape[0] - 1, ev["end_k"] + 200)
        t = np.arange(lo, hi + 1) * R.DT_COMM
        cell = int(ev["cell"])
        own = np.clip(np.nan_to_num(d["los_loss_db"][lo : hi + 1, u, cell], posinf=60.0), 0, 60)
        oth = np.clip(np.nan_to_num(d["los_loss_db"][lo : hi + 1, u, 1 - cell], posinf=60.0), 0, 60)
        ax.plot(t, own, color="black", lw=1.3, label=f"event cell (O-RU {cell}) LoS loss")
        ax.plot(t, oth, color="gray", lw=1.0, ls="--", label=f"other cell (O-RU {1 - cell}) LoS loss")
        ax.axhline(10.0, color="gray", lw=0.5, ls=":")
        ax.axvspan(ev["start_s"], ev["end_s"], color="tab:red", alpha=0.08, label="10 dB event")
        rec = {"job": item["job"], "ue": u, "start_s": ev["start_s"], "end_s": ev["end_s"], "onset_s": ev["onset_s"], "blocker": ev["blocker_id"], "handovers": {}}
        styles = {0: "-", 1: "--"}
        for n, (mi, lab) in enumerate(margins_idx):
            snr_m = {job: snr_all[job][mi]}
            orates = R.stateless_rows([job], snr_m, built, "oracle", bw, rate_req)[1]
            for scheme, params, color in (("a3", tuned_a3[mi], "tab:blue"), ("xapp", tuned_genie[mi], "tab:orange")):
                rows = run([job], [params], scheme, snr_m, built, rw, bw, rate_req, orates, a3={k: params[k] for k in ("offset_db", "hysteresis_db", "ttt_s")}, tau_ho=rw["e2"]["tau_ho_s"])
                row = next(r for r in rows if r["ue"] == u)
                steps = [s for s in row["ho_steps"] if lo <= s <= hi]
                name = "A3" if scheme == "a3" else "genie+A3"
                rec["handovers"][f"{name} @ {lab}"] = [{"switch_s": s * R.DT_COMM, "decision_s": (s - 1) * R.DT_COMM} for s in steps]
                for s in steps:
                    ax.axvline(s * R.DT_COMM, color=color, lw=1.0, ls=styles[n], alpha=0.9)
                ax.plot([], [], color=color, ls=styles[n], label=f"{name} HO ({lab})")
        ax.set_title(f"{cls}: seed {job[0]}, {job[1]}/{job[2]}, UE {u}", fontsize=8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Model-B LoS loss [dB] (capped 60)")
        ax.grid(True, lw=0.3, alpha=0.5)
        record[cls] = rec
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "onset_events.png", dpi=150)
    fig.savefig(out / "onset_events.pdf")
    plt.close(fig)
    (out / "onset_events.json").write_text(json.dumps({"rule": "per class, the evaluation-seed 10 dB event with duration closest to the class median (ties: seed, mount, density, UE, start); A3 decision = switch - 10 ms", "events": record, "chosen": chosen}, indent=1, default=R._json))
    return record


def main() -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    clock = time.perf_counter()
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    main_metrics = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    add_genie(built)
    info = R.budget(built, tune_jobs, rw, bw)
    if abs(info["margin_ref_db"] - main_metrics["budget"]["margin_ref_db"]) > 1e-9:
        raise SystemExit("reference margin differs from the main run")
    snr_all = R.all_snr(built, tune_jobs + eval_jobs, rw, bw, info["extra_loss_db"])
    genie_combos = R._grid(GENIE)
    a3_combos = R._grid(rw["grids"]["a3"])
    tau_default = float(rw["e2"]["tau_ho_s"])
    out = {"labels": info["labels"], "points_db": info["points_db"], "genie_grid": GENIE, "tau_ho_default_s": tau_default, "margins": []}
    tuned_a3_main, tuned_genie_main = [], []
    for mi, lab in enumerate(info["labels"]):
        t0 = time.perf_counter()
        snr_t = {j: snr_all[j][mi] for j in tune_jobs}
        snr_e = {j: snr_all[j][mi] for j in eval_jobs}
        orates_t = R.stateless_rows(tune_jobs, snr_t, built, "oracle", bw, rate_req)[1]
        oracle_rows, orates_e = R.stateless_rows(eval_jobs, snr_e, built, "oracle", bw, rate_req)
        res: dict[str, Any] = {"label": lab, "margin_db": info["points_db"][mi], "oracle": R.aggregate(oracle_rows), "tau_ho": {}}
        for tau in (tau_default, 0.0):
            if tau == tau_default:
                a3p = main_metrics["tuned"][mi]["a3"]["params"]
                a3_obj = main_metrics["tuned"][mi]["a3"]["objective"]
            else:
                i, a3_obj = tune(tune_jobs, a3_combos, "a3", snr_t, built, rw, bw, rate_req, orates_t, a3={}, tau_ho=tau)
                a3p = a3_combos[i]
            block: dict[str, Any] = {"a3_params": a3p, "a3_tuning_objective": a3_obj}
            a3_rows = run(eval_jobs, [a3p], "a3", snr_e, built, rw, bw, rate_req, orates_e, a3={}, tau_ho=tau)
            block["a3"] = R.aggregate(a3_rows)
            block["a3_decomp"] = decomp_summary(a3_rows)
            for variant, ovh in (("genie", True), ("genie_no_overhead", False)):
                gi, gobj = tune(tune_jobs, genie_combos, "xapp", snr_t, built, rw, bw, rate_req, orates_t, a3=a3p, tau_ho=tau, overhead=ovh)
                gp = {**a3p, **genie_combos[gi]}
                rows = run(eval_jobs, [gp], "xapp", snr_e, built, rw, bw, rate_req, orates_e, a3=a3p, tau_ho=tau, overhead=ovh)
                block[variant] = R.aggregate(rows)
                block[f"{variant}_decomp"] = decomp_summary(rows)
                block[f"{variant}_params"] = gp
                block[f"{variant}_tuning_objective"] = gobj
                if variant == "genie" and tau == tau_default:
                    tuned_genie_main.append(gp)
            res["tau_ho"][f"{tau:.3f}"] = block
            if tau == tau_default:
                tuned_a3_main.append(a3p)
        o = res["oracle"]["outage_req_s_per_min"]["mean"]
        g20 = res["tau_ho"][f"{tau_default:.3f}"]
        g0 = res["tau_ho"]["0.000"]
        gap = lambda b, k: b[k]["outage_req_s_per_min"]["mean"] - o  # noqa: E731
        res["gap_closure"] = {
            "a3_gap_tau_default": gap(g20, "a3"),
            "a3_gap_tau0": gap(g0, "a3"),
            "genie_gap_tau_default": gap(g20, "genie"),
            "genie_gap_tau0": gap(g0, "genie"),
            "genie_no_overhead_gap_tau_default": gap(g20, "genie_no_overhead"),
            "genie_no_overhead_gap_tau0": gap(g0, "genie_no_overhead"),
        }
        res["wall_s"] = time.perf_counter() - t0
        out["margins"].append(res)
        print(
            f"{lab}: oracle {o:.3f} | A3 {g20['a3']['outage_req_s_per_min']['mean']:.3f} genie {g20['genie']['outage_req_s_per_min']['mean']:.3f} "
            f"genie-no-ovh {g20['genie_no_overhead']['outage_req_s_per_min']['mean']:.3f} | tau0: A3 {g0['a3']['outage_req_s_per_min']['mean']:.3f} "
            f"genie {g0['genie']['outage_req_s_per_min']['mean']:.3f} | {res['wall_s']:.0f} s",
            flush=True,
        )
    chosen = pick_events(built, eval_jobs)
    margins_idx = [(info["labels"].index("3GPP short-range reference"), "3GPP ref."), (info["labels"].index("10 dB"), "10 dB")]
    out["onset_figure"] = onset_figure(chosen, built, snr_all, margins_idx, tuned_a3_main, tuned_genie_main, rw, bw, rate_req, ROOT / "results" / "M3" / "onset")
    out["wall_s"] = time.perf_counter() - clock
    (ROOT / "results" / "M3" / "genie.json").write_text(json.dumps(out, indent=1, default=R._json) + "\n", encoding="utf-8")
    print(f"wrote results/M3/genie.json in {out['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
