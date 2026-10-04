"""Cost-aware oracle (upper bound for any foresight policy) and genie diagnosis.

Per lane (job, UE), a Viterbi over the serving cell in 10 ms steps with
perfect knowledge of both cells' blocked SNR minimises the outage time at
the service rate. A switch at step k makes steps k .. k + tau_HO - 1
outage (interruption, the simulator's rule); no switch during an
interruption; the initial cell is free. Variants: tau_HO = 20 ms and 0;
switches allowed at every 10 ms step ("any step") or only at 0.1 s E2
epochs (k mod 10 == 0). With tau_HO = 0 and any step the result must equal
the instantaneous oracle (checked). No sensing overhead (bound).

Also: per margin, A3 (main tuned; A3 retuned at tau_HO = 0 from
genie.json), genie (first-round policy) and onset-advance genie
(genie2.json), the instantaneous oracle, the share of the A3-oracle gap the
cost-aware oracle closes, and the closed gap split by blocker class
(steps where A3 is in outage and the cost-aware oracle is not, minus the
reverse, by the dominant LoS blocker of A3's cell at that step).

Genie diagnosis (every margin, reported at 10 dB): blockage-caused
wrong-cell steps of A3 (A3's cell unusable, other usable, A3's cell usable
unblocked) split by the genie trigger conditions: LoS loss on A3's cell
< 10 dB, and other cell's LoS loss >= 3 dB.

Writes results/M5/dporacle.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_m3 as R  # noqa: E402
from run_m5_foresight import CLASS_OF, blocker_kinds, usable  # noqa: E402
from sim.comm.linkbudget import snr_ref_db  # noqa: E402
from sim.comm.phy import MAX_NR_SE  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from xapp.schemes import simulate  # noqa: E402

CLASSES = ("bus/truck", "pedestrian", "car", "no LoS blocker")


def viterbi(bad: np.ndarray, tau: int, epoch: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Min-outage cell path. ``bad`` [L, T, 2] bool (cell below the service rate).

    State (c, p): serving cell c; p > 0 = this step is interrupted and p-1
    interrupted steps follow; p = 0 = normal step (outage iff ``bad``). A
    switch at step k enters (other cell, tau) -- exactly tau interrupted
    steps -- or, for tau = 0, (other cell, 0). Switches may start from any
    state, only at steps with (k - offset) % epoch == 0 (epoch 1 = any step;
    offset = grid alignment, e.g. the planner's E2-delay-offset grid); the
    initial cell is free. Returns (outage mask [L, T], cell path [L, T]).
    """
    n_l, n_t, _ = bad.shape
    n_p = tau + 1
    big = np.int64(10**9)
    cost = np.full((n_l, 2, n_p), big, dtype=np.int64)
    cost[:, :, 0] = bad[:, 0, :]
    back_c = np.zeros((n_t, n_l, 2, n_p), dtype=np.int8)
    back_p = np.zeros((n_t, n_l, 2, n_p), dtype=np.int16)
    for k in range(1, n_t):
        bk = bad[:, k, :].astype(np.int64)
        new = np.full_like(cost, big)
        bc = np.zeros((n_l, 2, n_p), dtype=np.int8)
        bp = np.zeros((n_l, 2, n_p), dtype=np.int16)
        for c in range(2):
            for p in range(n_p):  # natural step: p -> max(p-1, 0)
                q = max(p - 1, 0)
                val = cost[:, c, p] + (1 if q > 0 else bk[:, c])
                better = val < new[:, c, q]
                new[:, c, q] = np.where(better, val, new[:, c, q])
                bc[:, c, q] = np.where(better, c, bc[:, c, q])
                bp[:, c, q] = np.where(better, p, bp[:, c, q])
        if (k - offset) % epoch == 0:
            for c in range(2):
                o = 1 - c
                best_p = cost[:, o, :].argmin(1)
                best = cost[np.arange(n_l), o, best_p]
                q = tau
                val = best + (1 if tau > 0 else bk[:, c])
                better = val < new[:, c, q]
                new[:, c, q] = np.where(better, val, new[:, c, q])
                bc[:, c, q] = np.where(better, o, bc[:, c, q])
                bp[:, c, q] = np.where(better, best_p, bp[:, c, q])
        cost = new
        back_c[k], back_p[k] = bc, bp
    flat = cost.reshape(n_l, -1)
    idx = flat.argmin(1)
    c_cur, p_cur = idx // n_p, idx % n_p
    cells = np.zeros((n_l, n_t), dtype=np.int64)
    phase = np.zeros((n_l, n_t), dtype=np.int64)
    lanes = np.arange(n_l)
    for k in range(n_t - 1, -1, -1):
        cells[:, k], phase[:, k] = c_cur, p_cur
        if k == 0:
            break
        c_prev = back_c[k, lanes, c_cur, p_cur].astype(np.int64)
        p_prev = back_p[k, lanes, c_cur, p_cur].astype(np.int64)
        c_cur, p_cur = c_prev, p_prev
    out = (phase > 0) | bad[lanes[:, None], np.arange(n_t)[None, :], cells]
    if not np.array_equal(out.sum(1), flat.min(1)):
        raise RuntimeError("Viterbi backtrack does not reproduce the optimal cost")
    return out, cells


def viterbi_bruteforce(bad: np.ndarray, tau: int, epoch: int, offset: int = 0) -> int:
    """Exhaustive minimum over all switch sequences for one tiny lane (test helper)."""
    n_t = bad.shape[0]
    best = None
    for start in (0, 1):
        stack = [(1, start, 0, int(bad[0, start]))]
        while stack:
            k, c, p, acc = stack.pop()
            if k == n_t:
                best = acc if best is None else min(best, acc)
                continue
            q = max(p - 1, 0)
            stack.append((k + 1, c, q, acc + (1 if q > 0 else int(bad[k, c]))))
            if (k - offset) % epoch == 0:
                o = 1 - c
                stack.append((k + 1, o, tau, acc + (1 if tau > 0 else int(bad[k, o]))))
    return int(best)


def per_job(index, masks: dict[str, np.ndarray], minutes: float) -> dict[str, Any]:
    acc: dict[tuple, dict[str, float]] = {}
    for lane, (_, job, _u) in enumerate(index):
        a = acc.setdefault(job, {"n_ue": 0})
        a["n_ue"] += 1
        for key, m in masks.items():
            a[key] = a.get(key, 0.0) + float(m[lane].sum())
    return acc


def main() -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU (GPU 1)")
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    rw = cfg["rework"]
    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds
    seeds = load_seeds()
    tune_jobs = [(int(s), m, d) for s in seeds["tuning"] for m in R.MOUNTS for d in R.DENSITIES]
    eval_jobs = [(int(s), m, d) for s in seeds["evaluation"] for m in R.MOUNTS for d in R.DENSITIES]
    m3 = json.loads((ROOT / "results" / "M3" / "metrics.json").read_text())
    gen = json.loads((ROOT / "results" / "M3" / "genie.json").read_text())
    gen2 = json.loads((ROOT / "results" / "M3" / "genie2.json").read_text())
    bw = int(raw["n_subcarriers"]) * 15000.0 * (2 ** int(raw["numerology"]))
    rate_req = float(rw["service_rate_bps"])
    built = R.build(tune_jobs + eval_jobs, raw, cfg, 4)
    info = R.budget(built, tune_jobs, rw, bw)
    snr_all = R.all_snr(built, eval_jobs, rw, bw, info["extra_loss_db"])
    kinds = {job: blocker_kinds(raw, job) for job in eval_jobs}
    b = rw["budget"]
    tau_def = float(rw["e2"]["tau_ho_s"])
    out: dict[str, Any] = {"definition": __doc__, "margins": []}
    for mi, lab in enumerate(info["labels"]):
        snr_m = {j: snr_all[j][mi] for j in eval_jobs}
        rows: dict[str, Any] = {"label": lab, "margin_db": info["points_db"][mi]}
        g1 = gen["margins"][mi]["tau_ho"]
        g2 = gen2["margins"][mi]["tau_ho"]
        for tau in (tau_def, 0.0):
            t = f"{tau:.3f}"
            a3p = m3["tuned"][mi]["a3"]["params"] if tau == tau_def else g1[t]["a3_params"]
            lanes, index = R.make_lanes(eval_jobs, [a3p], snr_m, built, "a3", {}, rw, None, tau_ho_s=tau)
            sim = simulate(lanes, bandwidth_hz=bw, rate_req_bps=rate_req, max_se=MAX_NR_SE)
            n_t = lanes.snr_db.shape[1]
            minutes = n_t * R.DT_COMM / 60.0
            bad = ~usable(lanes.snr_db, bw, rate_req)
            inst = bad.all(-1)
            tau_steps = int(round(tau / R.DT_COMM))
            masks = {"a3": sim["outage_req"], "oracle_inst": inst}
            for name, epoch in (("costaware_any", 1), ("costaware_epoch", int(round(R.DT_SENSE / R.DT_COMM)))):
                o, _cells = viterbi(bad, tau_steps, epoch)
                masks[name] = o
            if tau == 0.0 and not np.array_equal(masks["costaware_any"], inst):
                raise SystemExit(f"{lab}: tau_HO = 0, any-step cost-aware oracle differs from the instantaneous oracle")
            # class of the dominant LoS blocker of A3's cell
            cls = np.empty(bad.shape[:2], dtype=object)
            srv = sim["serving"].astype(np.int64)
            for lane, (_, job, u) in enumerate(index):
                dom = built[job]["data"]["los_blocker"][np.arange(n_t), u, srv[lane]]
                cls[lane] = ["no LoS blocker" if x < 0 else CLASS_OF.get(kinds[job][x], "other") for x in dom]
            for name in ("costaware_any", "costaware_epoch"):
                gain = masks["a3"] & ~masks[name]
                loss = masks[name] & ~masks["a3"]
                for c in CLASSES:
                    masks[f"{name}_net_closed|{c}"] = (gain & (cls == c)).astype(np.int8) - (loss & (cls == c)).astype(np.int8)
            acc = per_job(index, masks, minutes)
            block: dict[str, Any] = {"a3_params": a3p}
            for key in ("a3", "oracle_inst", "costaware_any", "costaware_epoch"):
                block[key] = R.ci95([v[key] * R.DT_COMM / (minutes * v["n_ue"]) for v in acc.values()])
            tot = {k: sum(v[k] for v in acc.values()) for k in masks}
            gap = tot["a3"] - tot["oracle_inst"]
            for name in ("costaware_any", "costaware_epoch"):
                block[f"{name}_gap_closed_pooled"] = (tot["a3"] - tot[name]) / gap if gap > 0 else None
                block[f"{name}_gap_closed_by_class_pooled"] = {c: (tot[f"{name}_net_closed|{c}"] / gap if gap > 0 else None) for c in CLASSES}
            block["genie"] = g1[t]["genie"]["outage_req_s_per_min"]
            block["genie_no_overhead"] = g1[t]["genie_no_overhead"]["outage_req_s_per_min"]
            block["onset_genie"] = g2[t]["onset_genie"]["outage_req_s_per_min"]
            block["onset_genie_no_overhead"] = g2[t]["onset_genie_no_overhead"]["outage_req_s_per_min"]
            if tau == tau_def:
                # genie diagnosis on A3's blockage-caused wrong-cell steps
                diag = {"n": 0, "los_lt10": 0, "other_ge3_usable": 0, "either": 0, "both_conditions_met": 0}
                for lane, (_, job, u) in enumerate(index):
                    d = built[job]["data"]
                    k = np.arange(n_t)
                    s = srv[lane]
                    snr_unb = snr_ref_db(d["unblocked_power"][:, u, :], b["tx_power_dbm"], bw, b["ue_noise_figure_db"]) - b["losses_db"] - info["extra_loss_db"][mi]
                    blk = sim["outage_req"][lane] & ~bad[lane].all(-1) & ~sim["interrupted"][lane] & usable(snr_unb, bw, rate_req)[k, s]
                    los_srv = d["los_loss_db"][k, u, s]
                    los_oth = d["los_loss_db"][k, u, 1 - s]
                    c1 = blk & (los_srv < 10.0)
                    c2 = blk & (los_oth >= 3.0)
                    diag["n"] += int(blk.sum())
                    diag["los_lt10"] += int(c1.sum())
                    diag["other_ge3_usable"] += int(c2.sum())
                    diag["either"] += int((c1 | c2).sum())
                    diag["both_conditions_met"] += int((blk & ~c1 & ~c2).sum())
                block["genie_trigger_diagnosis"] = diag
            rows[t] = block
        out["margins"].append(rows)
        r20 = rows[f"{tau_def:.3f}"]
        d = r20["genie_trigger_diagnosis"]
        print(f"{lab}: A3 {r20['a3']['mean']:.3f} | genie {r20['genie_no_overhead']['mean']:.3f} | onset {r20['onset_genie_no_overhead']['mean']:.3f} | "
              f"cost-aware any {r20['costaware_any']['mean']:.3f} epoch {r20['costaware_epoch']['mean']:.3f} | inst {r20['oracle_inst']['mean']:.3f} | "
              f"closed any {r20['costaware_any_gap_closed_pooled'] if r20['costaware_any_gap_closed_pooled'] is None else round(r20['costaware_any_gap_closed_pooled'], 3)} | "
              f"diag n {d['n']} los<10 {d['los_lt10']} other>=3 {d['other_ge3_usable']} either {d['either']}", flush=True)
    dest = ROOT / "results" / "M5" / "dporacle.json"
    dest.write_text(json.dumps(out, indent=1, default=R._json) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
