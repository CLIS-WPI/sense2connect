"""TVT J3 diagnosis (human request after T6; DEVELOPMENT seeds only; measures, changes no method).

a) Value of foresight under the best-beam and the path-power-sum service model on the SAME development
   seeds (T0 sandboxes results/TVT/T0/<model>/M5/planner.json, paper-1 code): A5, instantaneous oracle,
   cost-aware oracle, genie-planner, value of foresight (best reactive - cost-aware oracle) and its share
   of the best-reactive gap, split by blocker class.
b) Planner with PERFECT blocker tracks and UE positions from (i) truth, (ii) the T4 tracker (pred_real,
   sigma_map 0), (iii) truth + white Gaussian errors with the tracker's per-axis RMS over the development
   jobs (results/TVT/T5/handover_j3.json, scripts/tvt_t5_handover.py --dump-steps); (iii') the same with
   the RMS over the epochs outside the 1 s after a UE wrap (the scenario wraps the UEs at the street
   ends; the tracker needs ~0.4 s to re-acquire, which dominates the along-street RMS). Tracker error
   [m, horizontal] at the planner's switch decisions (sensing epoch r = (step - 1 - d) // rs of each
   executed handover), in the 1 s before the onset of every 10 dB blockage event, and over all epochs.
c) Gap of each planner to A5 split per outage step (10 ms) into mutually exclusive error types, the
   same rules for every scheme (usable = rate WITHOUT sensing overhead >= service rate):
     unavoidable          no cell usable
     switch_necessary     handover interruption of a switch whose source cell is unusable at some
                          step within 1 s after the switch
     unnecessary_switch   handover interruption of any other switch, or serving an unusable cell that
                          the scheme switched into while the source cell was usable
     mistimed_switch      serving an unusable cell that went bad while serving and that the scheme left
                          later in the same unusable run (late switch), or that it switched into while
                          both cells were unusable (the source recovered first)
     missed_switch        serving an unusable cell that went bad while serving and that the scheme did
                          not leave before the cell recovered
     overhead             serving cell usable without but not with the sensing overhead
   and per blocker class (10 dB event of the UE covering the step within [onset - 2 s, end], serving
   cell first, then the other cell; "none" otherwise). Gap = planner - A5 per category; seed level
   (per seed mean over 4 runs x 2 UEs), paired exact Wilcoxon over the 10 development seeds.
Signaling round (human decision before the freeze): --model ideal | failure_aware reads
results/TVT/J3_diagnosis/steps_<model>/ and results/TVT/T5/handover_<model>.json; PRIMARY metrics exclude the
1 s after each UE wrap (configs/tvt.yaml evaluation:), unmasked totals are the supplement; extra category
     rlf_reestablishment  outage while the UE re-establishes after a radio link failure (failure-aware)
(after "unavoidable" in the priority order). d) Sensing overhead 0 (human item 5): truth-UE and tracker-UE
planners with perfect tracks vs A5 at overhead x1 (main), x0 with the x1 parameters (T6) and x0 re-tuned.
Without --model the first-round inputs (steps/, handover_j3.json) are used.
Writes results/TVT/J3_diagnosis/diagnosis[_<model>].json and prints markdown tables.
Run: python scripts/tvt_j3_diagnosis.py [--model failure_aware]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "results" / "TVT" / "J3_diagnosis"
STEPS = OUT / "steps"
MARGINS = ("15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference")
PLANNERS = ("planner_true_perfect", "planner_tvt_perfect", "planner_white_perfect", "planner_whitenw_perfect", "planner_tvt")
CATS = ("unavoidable", "rlf_reestablishment", "switch_necessary", "unnecessary_switch", "mistimed_switch", "missed_switch", "overhead")
CLASSES = ("bus/truck", "pedestrian", "car", "no LoS blocker", "none")
NEED_WINDOW_S = 1.0
EVENT_BEFORE_S = 2.0


# ------------------------------------------------------------------ a) service models
def part_a() -> dict:
    out = {}
    for model in ("power_sum", "best_beam"):
        pl = json.loads((ROOT / "results" / "TVT" / "T0" / model / "M5" / "planner.json").read_text())
        rows = {}
        for m in pl["margins"]:
            if m["label"] not in MARGINS:
                continue
            hg = f"{float(m['genie_planner_tuned_H']):.1f}"
            vf = m["value_of_foresight"]
            g = lambda x: x["eval"]["outage_req_s_per_min"] if "eval" in x else x["outage_req_s_per_min"]  # noqa: E731
            rows[m["label"]] = {
                "a5": g(m["a5"]), "oracle_inst": g(m["oracle_inst"]), "costaware_oracle": vf["costaware"],
                "genie_planner": g(m["genie_planner"][hg]), "best_reactive": m["best_reactive"],
                "best_reactive_outage": g(m[m["best_reactive"]]),
                "value_of_foresight": vf["best_reactive_minus_costaware_s_per_ue_min"], "share_of_gap": vf["share_of_best_reactive_gap_pooled"],
                "share_of_gap_by_class": vf["by_class_share_of_gap_pooled"]}
        out[model] = rows
    return out


# ------------------------------------------------------------------ helpers
def usable_mask(snr_db: np.ndarray, bw: float, rate_req: float) -> np.ndarray:
    from sim.comm.phy import MAX_NR_SE

    se = np.minimum(np.log2(1.0 + 10.0 ** (snr_db.astype(np.float64) / 10.0)), MAX_NR_SE)
    return bw * se >= rate_req


def runs_start(bad: np.ndarray) -> np.ndarray:
    """For every step, the first step of the run of True values containing it (-1 where False)."""
    st = np.full(bad.shape, -1, dtype=np.int64)
    cur = -1
    for k in range(bad.size):
        if bad[k]:
            cur = k if cur < 0 else cur
            st[k] = cur
        else:
            cur = -1
    return st


def runs_end(bad: np.ndarray) -> np.ndarray:
    """For every step, the last step of the run of True values containing it (-1 where False)."""
    en = np.full(bad.shape, -1, dtype=np.int64)
    cur = -1
    for k in range(bad.size - 1, -1, -1):
        if bad[k]:
            cur = k if cur < 0 else cur
            en[k] = cur
        else:
            cur = -1
    return en


def classify_lane(serv, outr, intr, snr, ovh, hos, events, meta, reest=None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Category index [T] (-1 = no outage) and class index [T] of every outage step; per-switch records."""
    bw, req, dt = meta["bandwidth_hz"], meta["rate_req_bps"], meta["dt_s"]
    tau = int(meta["tau_ho_steps"])
    n_t = serv.size
    us = usable_mask(snr, bw, req)  # [T, 2]
    from sim.comm.phy import MAX_NR_SE

    se = np.minimum(np.log2(1.0 + 10.0 ** (snr.astype(np.float64) / 10.0)), MAX_NR_SE)
    t = np.arange(n_t)
    srv_us = us[t, serv]
    oth_us = us[t, 1 - serv]
    srv_rate_ovh = bw * (1.0 - ovh) * se[t, serv]
    st = [runs_start(~us[:, c]) for c in range(2)]
    en = [runs_end(~us[:, c]) for c in range(2)]
    need_w = int(round(NEED_WINDOW_S / dt))
    steps, frm, to = hos["step"], hos["from"], hos["to"]
    order = np.argsort(steps, kind="stable")
    steps, frm, to = steps[order], frm[order], to[order]
    sw = []
    for s_, f_, t_ in zip(steps, frm, to):
        nec = bool((~us[s_: min(n_t, s_ + need_w + 1), f_]).any())
        sw.append({"step": int(s_), "from": int(f_), "to": int(t_), "necessary": nec, "source_usable": bool(us[min(s_, n_t - 1), f_])})
    cat = np.full(n_t, -1, dtype=np.int64)
    for k in np.flatnonzero(outr):
        if not us[k].any():
            cat[k] = CATS.index("unavoidable")
            continue
        if reest is not None and reest.size and reest[k]:
            cat[k] = CATS.index("rlf_reestablishment")
            continue
        if intr[k]:
            i = np.searchsorted(steps, k, side="right") - 1
            nec = sw[i]["necessary"] if i >= 0 and k < steps[i] + tau else True
            cat[k] = CATS.index("switch_necessary" if nec else "unnecessary_switch")
            continue
        c = serv[k]
        if srv_us[k]:
            cat[k] = CATS.index("overhead") if srv_rate_ovh[k] < req else CATS.index("unavoidable")
            continue
        # serving cell unusable, other usable
        b = st[c][k]
        into = np.flatnonzero((steps <= k) & (to == c))
        s_in = int(steps[into[-1]]) if into.size else -1
        if s_in >= b:  # switched into an already unusable cell
            src_us = sw[into[-1]]["source_usable"]
            cat[k] = CATS.index("unnecessary_switch" if src_us else "mistimed_switch")
            continue
        e = en[c][k]
        left = np.flatnonzero((steps > k) & (steps <= e + 1) & (frm == c))
        cat[k] = CATS.index("mistimed_switch" if left.size else "missed_switch")
    cls = np.full(n_t, CLASSES.index("none"), dtype=np.int64)
    before = int(round(EVENT_BEFORE_S / dt))
    for prefer_serving in (False, True):  # serving-cell events written last (take precedence)
        for ev in events:
            lo, hi = max(0, int(ev["start_k"]) - before), min(n_t - 1, int(ev["end_k"]))
            if hi < lo:
                continue
            ks = np.arange(lo, hi + 1)
            on_srv = serv[ks] == int(ev["cell"])
            sel = ks[on_srv] if prefer_serving else ks[~on_srv]
            cls[sel] = CLASSES.index(ev["class"]) if ev["class"] in CLASSES else CLASSES.index("none")
    cls = np.where(cat >= 0, cls, -1)
    return cat, cls, sw


def load_steps(lab: str, sch: str) -> dict:
    f = np.load(STEPS / f"{lab.split()[0]}_{sch}.npz")
    return {k: f[k] for k in f.files}


def seed_of(lane: str) -> int:
    return int(lane.split("|")[0].split("_")[-1])


# ------------------------------------------------------------------ b) tracker error at the decisions
def part_b(meta, events) -> dict:
    from sim.tvt.stats import paired, seed_summary

    f = np.load(STEPS / "ue.npz")
    ue = {}
    for k in f.files:
        src, job = k.split("|")
        ue.setdefault(job, {})[src] = f[k]
    d, rs, dt = int(meta["e2_delay_steps"]), int(meta["report_steps"]), meta["dt_s"]
    eps_per_s = int(round(1.0 / (rs * dt)))
    res = {"white_rms_m": meta["white_rms_m"], "whitenw_rms_m": meta["whitenw_rms_m"], "windows": {}}
    win_vals: dict[str, dict[int, list[float]]] = {}

    def add(w, seed, v):
        win_vals.setdefault(w, {}).setdefault(seed, []).extend(np.atleast_1d(v).tolist())

    for job, v in ue.items():
        seed = int(job.split("_")[-1])
        e_tvt = np.linalg.norm(v["tvt"] - v["true"], axis=-1)  # [R, U]
        e_wh = np.linalg.norm(v["white"] - v["true"], axis=-1)
        n_r = e_tvt.shape[0]
        for u in range(e_tvt.shape[1]):
            tu = v["true"][:, u]
            wrap = np.zeros(n_r, bool)
            wrap[1:] = np.linalg.norm(np.diff(tu, axis=0), axis=-1) > 10.0
            after = np.zeros(n_r, bool)
            for k_ in range(eps_per_s):
                after[k_:] |= wrap[: n_r - k_]
            keep_e = ~after  # primary: the 1 s after each UE wrap excluded
            add("tvt | all epochs", seed, e_tvt[keep_e, u])
            add("white | all epochs", seed, e_wh[keep_e, u])
            add("whitenw | all epochs", seed, np.linalg.norm(v["whitenw"] - v["true"], axis=-1)[keep_e, u])
            add("tvt | all epochs, unmasked (supplement)", seed, e_tvt[:, u])
            add("tvt | 1 s after a UE wrap", seed, e_tvt[after, u])
            add("tvt | epochs after 2 s", seed, e_tvt[20:, u][keep_e[20:]])
            pre = np.zeros(n_r, bool)
            for ev in events[job]:
                if int(ev["ue"]) != u:
                    continue
                r1 = int(ev["start_k"]) // rs
                pre[max(0, r1 - eps_per_s):max(0, r1)] = True
            add("tvt | 1 s before onset", seed, e_tvt[pre & keep_e, u])
            add("white | 1 s before onset", seed, e_wh[pre & keep_e, u])
    for sch, src in (("planner_tvt_perfect", "tvt"), ("planner_true_perfect", "tvt"), ("planner_white_perfect", "white"), ("planner_whitenw_perfect", "whitenw")):
        lab = "20 dB"
        s = load_steps(lab, sch)
        for i, lane in enumerate(s["lanes"]):
            job, u = lane.split("|")
            u = int(u)
            seed = seed_of(lane)
            st = s["ho_step"][s["ho_lane"] == i]
            r = np.clip((st - 1 - d) // rs, 0, None)
            e = np.linalg.norm(ue[job][src] - ue[job]["true"], axis=-1)[:, u]
            r = r[r < e.size]
            tu = ue[job]["true"][:, u]
            wrap = np.zeros(e.size, bool)
            wrap[1:] = np.linalg.norm(np.diff(tu, axis=0), axis=-1) > 10.0
            after = np.zeros(e.size, bool)
            for k_ in range(eps_per_s):
                after[k_:] |= wrap[: e.size - k_]
            r = r[~after[r]]
            add(f"{src} | switch decisions of {sch} (20 dB)", seed, e[r])
    for w, per in win_vals.items():
        seeds = sorted(per)
        cell = {}
        for stat, fn in (("median", np.median), ("p90", lambda x: np.percentile(x, 90)), ("rms", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                         ("share_gt_0.5m", lambda x: float(np.mean(np.asarray(x) > 0.5)))):
            vals = [float(fn(np.asarray(per[s_]))) if per[s_] else math.nan for s_ in seeds]
            cell[stat] = seed_summary(vals)
            cell[stat]["per_seed"] = vals
        cell["n"] = int(sum(len(per[s_]) for s_ in seeds))
        res["windows"][w] = cell
    base = res["windows"]["tvt | all epochs"]
    res["paired_vs_all_epochs"] = {}
    for w, cell in res["windows"].items():
        if w.startswith("tvt |") and w != "tvt | all epochs":
            for stat in ("median", "p90"):
                a_, b_ = cell[stat]["per_seed"], base[stat]["per_seed"]
                if all(np.isfinite(a_)):
                    res["paired_vs_all_epochs"][f"{w} | {stat}"] = paired(a_, b_)
    return res


# ------------------------------------------------------------------ c) gap breakdown
def wrap_steps(meta) -> dict:
    """Per lane key 'job|u': bool [T_comm] mask of the 1 s after every UE wrap (truth jumps by > 10 m between epochs)."""
    f = np.load(STEPS / "ue.npz")
    rs, dt = int(meta["report_steps"]), meta["dt_s"]
    w = int(round(1.0 / dt))
    out = {}
    for k in f.files:
        src, job = k.split("|")
        if src != "true":
            continue
        tu = f[k]
        for u in range(tu.shape[1]):
            jumps = np.flatnonzero(np.linalg.norm(np.diff(tu[:, u], axis=0), axis=-1) > 10.0) + 1
            out[f"{job}|{u}"] = [(int(r) * rs, int(r) * rs + w) for r in jumps]
    return out


def part_c(meta, events) -> dict:
    from sim.tvt.stats import paired

    wraps = wrap_steps(meta)
    res = {}
    for lab in MARGINS:
        per_scheme = {}
        for sch in ("A5",) + PLANNERS:
            if not (STEPS / f"{lab.split()[0]}_{sch}.npz").exists():
                continue
            s = load_steps(lab, sch)
            n_t = s["serving"].shape[1]
            minutes = n_t * meta["dt_s"] / 60.0
            acc_cat = {}
            acc_cls = {}
            acc_cc = {}
            sw_stats = {}
            total = {}
            total_um = {}
            in_wrap = {}
            for i, lane in enumerate(s["lanes"]):
                job, u = lane.split("|")
                u = int(u)
                seed = seed_of(lane)
                sel = s["ho_lane"] == i
                hos = {"step": s["ho_step"][sel], "from": s["ho_from"][sel], "to": s["ho_to"][sel]}
                evs = [ev for ev in events[job] if int(ev["ue"]) == u]
                re_ = s["reestablishing"][i] if "reestablishing" in s and s["reestablishing"].shape[-1] else None
                cat, cls, sw = classify_lane(s["serving"][i].astype(np.int64), s["outage_req"][i], s["interrupted"][i], s["snr"][i], float(s["overhead"][i]),
                                             hos, evs, meta, reest=re_)
                assert (cat >= 0).sum() == s["outage_req"][i].sum()
                wm = np.zeros(n_t, bool)
                for lo, hi in wraps.get(lane, []):
                    wm[lo:min(hi, n_t)] = True
                keep = ~wm  # primary metrics exclude the 1 s after each UE wrap
                mk = max(keep.sum(), 1) * meta["dt_s"] / 60.0
                total.setdefault(seed, []).append((s["outage_req"][i] & keep).sum() * meta["dt_s"] / mk)
                total_um.setdefault(seed, []).append(s["outage_req"][i].sum() * meta["dt_s"] / minutes)
                in_wrap.setdefault(seed, []).append((s["outage_req"][i] & wm).sum() * meta["dt_s"] / minutes)
                for ci, cname in enumerate(CATS):
                    acc_cat.setdefault(cname, {}).setdefault(seed, []).append(((cat == ci) & keep).sum() * meta["dt_s"] / mk)
                    for kj, kname in enumerate(CLASSES):
                        acc_cc.setdefault(f"{cname}|{kname}", {}).setdefault(seed, []).append(((cat == ci) & (cls == kj) & keep).sum() * meta["dt_s"] / mk)
                for kj, kname in enumerate(CLASSES):
                    acc_cls.setdefault(kname, {}).setdefault(seed, []).append(((cls == kj) & keep).sum() * meta["dt_s"] / mk)
                swk = [x for x in sw if keep[min(x["step"], n_t - 1)]]
                n_nec = sum(x["necessary"] for x in swk)
                sw_stats.setdefault("switches_per_min", {}).setdefault(seed, []).append(len(swk) / mk)
                sw_stats.setdefault("necessary_per_min", {}).setdefault(seed, []).append(n_nec / mk)
                sw_stats.setdefault("unnecessary_per_min", {}).setdefault(seed, []).append((len(swk) - n_nec) / mk)
            red = lambda dct: {k: {s_: float(np.mean(v)) for s_, v in sorted(x.items())} for k, x in dct.items()}  # noqa: E731
            per_scheme[sch] = {"total": {s_: float(np.mean(v)) for s_, v in sorted(total.items())},
                               "total_unmasked": {s_: float(np.mean(v)) for s_, v in sorted(total_um.items())}, "wrap": {s_: float(np.mean(v)) for s_, v in sorted(in_wrap.items())},
                               "cat": red(acc_cat), "cls": red(acc_cls),
                               "cat_cls": red(acc_cc), "switches": red(sw_stats)}
        out = {}
        for sch, v in per_scheme.items():
            seeds = sorted(v["total"])
            mean = lambda dd: float(np.mean([dd[s_] for s_ in seeds]))  # noqa: E731
            cell = {"outage": mean(v["total"]), "outage_unmasked": mean(v["total_unmasked"]), "outage_1s_after_ue_wrap": mean(v["wrap"]), "by_category": {k: mean(x) for k, x in v["cat"].items()},
                    "by_class": {k: mean(x) for k, x in v["cls"].items()}, "by_category_class": {k: mean(x) for k, x in v["cat_cls"].items()},
                    "switches": {k: mean(x) for k, x in v["switches"].items()}}
            if sch != "A5" and "A5" in per_scheme:
                a5 = per_scheme["A5"]
                cell["gap_vs_A5"] = paired([v["total"][s_] for s_ in seeds], [a5["total"][s_] for s_ in seeds])
                cell["gap_vs_A5_unmasked"] = paired([v["total_unmasked"][s_] for s_ in seeds], [a5["total_unmasked"][s_] for s_ in seeds])
                cell["gap_vs_A5_1s_after_ue_wrap"] = paired([v["wrap"][s_] for s_ in seeds], [a5["wrap"][s_] for s_ in seeds])
                if "planner_true_perfect" in per_scheme and sch != "planner_true_perfect":
                    tp = per_scheme["planner_true_perfect"]
                    cell["vs_true_perfect"] = paired([v["total"][s_] for s_ in seeds], [tp["total"][s_] for s_ in seeds])
                    cell["vs_true_perfect_unmasked"] = paired([v["total_unmasked"][s_] for s_ in seeds], [tp["total_unmasked"][s_] for s_ in seeds])
                    cell["vs_true_perfect_1s_after_ue_wrap"] = paired([v["wrap"][s_] for s_ in seeds], [tp["wrap"][s_] for s_ in seeds])
                    cell["vs_true_perfect_by_class"] = {k: paired([x[s_] for s_ in seeds], [tp["cls"][k][s_] for s_ in seeds]) for k, x in v["cls"].items()}
                cell["gap_by_category"] = {k: paired([x[s_] for s_ in seeds], [a5["cat"][k][s_] for s_ in seeds]) for k, x in v["cat"].items()}
                cell["gap_by_class"] = {k: paired([x[s_] for s_ in seeds], [a5["cls"][k][s_] for s_ in seeds]) for k, x in v["cls"].items()}
                cell["gap_by_category_class"] = {k: float(np.mean([x[s_] - a5["cat_cls"][k][s_] for s_ in seeds])) for k, x in v["cat_cls"].items()}
            out[sch] = cell
        res[lab] = out
    return res


def part_d(model: str) -> dict:
    """Human item 5: perfect-track planners vs A5 at sensing overhead x1 (main), x0 with the x1 parameters, x0 re-tuned."""
    from sim.tvt.stats import paired

    T5 = ROOT / "results" / "TVT" / "T5"
    files = {"x1 (main, tuned at x1)": T5 / f"handover_{model}.json", "x0, parameters tuned at x1": T5 / f"handover_ovh_0_{model}.json",
             "x0, re-tuned at x0": T5 / f"handover_ovh0_tuned_{model}.json"}
    out = {}
    for name, f in files.items():
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for lab, cell in d["schemes"].items():
            seeds = sorted(cell["A5"]["per_seed"])
            for sch in ("planner_true_perfect", "planner_tvt_perfect"):
                if sch in cell:
                    out.setdefault(lab, {})[f"{name} | {sch}"] = {
                        "outage": cell[sch]["outage"], "A5": cell["A5"]["outage"], "params": cell[sch]["params"],
                        "vs_A5": paired([cell[sch]["per_seed"][x]["outage"] for x in seeds], [cell["A5"]["per_seed"][x]["outage"] for x in seeds])}
    return out


def part_e(model: str) -> dict | None:
    """Human item 4 (third round): truth-UE planner at sensing overhead 0, blocker states (i) perfect at the decision
    instant and extrapolated (planner_true_perfect) vs (ii) the true future trajectories (planner_true_future), vs A5,
    by blocker class; both re-tuned at overhead 0 on the tuning seeds (T5 run handover_ovh0_tuned_<model>.json with
    --dump-steps results/TVT/J3_diagnosis/steps_ovh0_<model>). Same classifier and masks as part c."""
    global STEPS, PLANNERS
    d = OUT / f"steps_ovh0_{model}"
    if not (d / "meta.json").exists():
        return None
    saved = (STEPS, PLANNERS)
    STEPS, PLANNERS = d, ("planner_true_perfect", "planner_true_future")
    try:
        return part_c(json.loads((d / "meta.json").read_text()), json.loads((d / "events.json").read_text()))
    finally:
        STEPS, PLANNERS = saved


def main() -> None:
    global STEPS
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=["ideal", "failure_aware"])
    a = ap.parse_args()
    if a.model:
        STEPS = OUT / f"steps_{a.model}"
    meta = json.loads((STEPS / "meta.json").read_text())
    events = json.loads((STEPS / "events.json").read_text())
    ho = json.loads((ROOT / "results" / "TVT" / "T5" / (f"handover_{a.model}.json" if a.model else "handover_j3.json")).read_text())
    out = {"definition": __doc__, "a": part_a(), "b_handover": {lab: {sch: {k: v[k] for k in ("outage", "ho_per_min", "ping_pong", "params", "vs_A5") if k in v}
                                                                        for sch, v in cell.items() if " vs " not in sch}
                                                                  for lab, cell in ho["schemes"].items()},
           "b_tracker_error": part_b(meta, events), "c": part_c(meta, events)}
    from sim.tvt.stats import paired

    pairs = {}
    for lab, cell in ho["schemes"].items():
        seeds = sorted(cell["A5"]["per_seed"])
        for x, y in (("planner_tvt_perfect", "planner_true_perfect"), ("planner_white_perfect", "planner_true_perfect"),
                     ("planner_tvt_perfect", "planner_white_perfect"), ("planner_whitenw_perfect", "planner_true_perfect"),
                     ("planner_tvt_perfect", "planner_whitenw_perfect")):
            if x in cell and y in cell:
                pairs[f"{lab} | {x} vs {y}"] = paired([cell[x]["per_seed"][s]["outage"] for s in seeds], [cell[y]["per_seed"][s]["outage"] for s in seeds])
    out["b_pairs"] = pairs
    out["model"] = a.model
    if a.model:
        out["d_overhead0"] = part_d(a.model)
        out["e_future_truth"] = part_e(a.model)
    (OUT / (f"diagnosis_{a.model}.json" if a.model else "diagnosis.json")).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print_tables(out)
    print_part_d(out)
    print_part_e(out)


def print_tables(o: dict) -> None:
    print("### a) Value of foresight, both service models, same development seeds (paper-1 code; mean over 40 runs x 2 UEs, +- 95 % CI half width)\n")
    print("| margin | model | A5 | best reactive | instantaneous oracle | cost-aware oracle | genie-planner | value of foresight (share of gap) | share of gap: bus/truck / pedestrian / car |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lab in MARGINS:
        for model in ("power_sum", "best_beam"):
            r = o["a"][model][lab]
            f = lambda x: f"{x['mean']:.3f} +- {x['ci']:.3f}"  # noqa: E731
            c = r["share_of_gap_by_class"]
            print(f"| {lab} | {model} | {f(r['a5'])} | {f(r['best_reactive_outage'])} ({r['best_reactive']}) | {f(r['oracle_inst'])} | {f(r['costaware_oracle'])} | "
                  f"{f(r['genie_planner'])} | {r['value_of_foresight']:.3f} ({100 * r['share_of_gap']:.0f} %) | "
                  f"{100 * c.get('bus/truck', 0):.0f} / {100 * c.get('pedestrian', 0):.0f} / {100 * c.get('car', 0):.0f} % |")
    print("\n### b) Planner with perfect blocker tracks, UE from truth / tracker / white errors (best beam; tuned on tuning seeds, development seeds)\n")
    rms, rnw = o["b_tracker_error"]["white_rms_m"], o["b_tracker_error"]["whitenw_rms_m"]
    print(f"White errors: per-axis RMS of the tracker error over the development jobs, x {rms[0]:.3f} m, y {rms[1]:.3f} m (white); outside the 1 s after a "
          f"UE wrap x {rnw[0]:.3f} m, y {rnw[1]:.3f} m (white-nw). Outage [s/UE-min] (vs A5: diff, p).\n")
    print("| margin | A5 | truth UE | tracker UE | white UE | white-nw UE | real tracks + tracker UE | tracker - truth (p) | white - truth (p) | white-nw - truth (p) | tracker - white-nw (p) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, cell in o["b_handover"].items():
        g = lambda s: (f"{cell[s]['outage']:.3f}" + (f" ({cell[s]['vs_A5']['mean_diff']:+.3f}, p {cell[s]['vs_A5']['wilcoxon_p_two_sided']:.2g})" if "vs_A5" in cell[s] else "")) if s in cell else "-"  # noqa: E731
        pp = lambda x, y: (lambda v: f"{v['mean_diff']:+.3f} ({v['wilcoxon_p_two_sided']:.2g})")(o["b_pairs"][f"{lab} | {x} vs {y}"])  # noqa: E731
        print(f"| {lab} | {g('A5')} | {g('planner_true_perfect')} | {g('planner_tvt_perfect')} | {g('planner_white_perfect')} | {g('planner_whitenw_perfect')} | {g('planner_tvt')} | "
              f"{pp('planner_tvt_perfect', 'planner_true_perfect')} | {pp('planner_white_perfect', 'planner_true_perfect')} | "
              f"{pp('planner_whitenw_perfect', 'planner_true_perfect')} | {pp('planner_tvt_perfect', 'planner_whitenw_perfect')} |")
    print("\nUE position error [m] (seed level: per seed statistic, mean over the 10 seeds [bootstrap 95 % CI]):\n")
    print("| window | n | median | p90 | RMS | share > 0.5 m |")
    print("|---|---|---|---|---|---|")
    for w, c in o["b_tracker_error"]["windows"].items():
        f = lambda x: f"{x['mean']:.3f} [{x['ci95_boot'][0]:.3f}, {x['ci95_boot'][1]:.3f}]"  # noqa: E731
        print(f"| {w} | {c['n']} | {f(c['median'])} | {f(c['p90'])} | {c['rms']['mean']:.3f} | {c['share_gt_0.5m']['mean']:.3f} |")
    print("\nPaired vs all epochs (tracker):\n")
    for k, v in o["b_tracker_error"]["paired_vs_all_epochs"].items():
        print(f"- {k}: {v['mean_diff']:+.3f} m [{v['ci95_boot'][0]:+.3f}, {v['ci95_boot'][1]:+.3f}], p = {v['wilcoxon_p_two_sided']:.2g}")
    print("\n### c) Gap to A5 by error type [s/UE-min] (planner - A5, seed-level mean; exact Wilcoxon p)\n")
    for lab, cell in o["c"].items():
        print(f"**{lab}**\n")
        print("| scheme | outage | gap vs A5 | " + " | ".join(CATS) + " | switches/min (necessary / unnecessary) |")
        print("|---|---|---|" + "---|" * len(CATS) + "---|")
        a5 = cell["A5"]
        print(f"| A5 (absolute) | {a5['outage']:.3f} | - | " + " | ".join(f"{a5['by_category'][c]:.3f}" for c in CATS)
              + f" | {a5['switches']['switches_per_min']:.2f} ({a5['switches']['necessary_per_min']:.2f} / {a5['switches']['unnecessary_per_min']:.2f}) |")
        for sch in PLANNERS:
            if sch not in cell:
                continue
            v = cell[sch]
            gp = lambda x: f"{x['mean_diff']:+.3f} ({x['wilcoxon_p_two_sided']:.2g})"  # noqa: E731
            print(f"| {sch} | {v['outage']:.3f} | {gp(v['gap_vs_A5'])} | " + " | ".join(gp(v["gap_by_category"][c]) for c in CATS)
                  + f" | {v['switches']['switches_per_min']:.2f} ({v['switches']['necessary_per_min']:.2f} / {v['switches']['unnecessary_per_min']:.2f}) |")
        print()
        print("| scheme | outage in the 1 s after a UE wrap | gap vs A5 there (p) | vs truth-UE planner: total (p) | of which in the 1 s after a wrap (p) |")
        print("|---|---|---|---|---|")
        print(f"| A5 | {a5['outage_1s_after_ue_wrap']:.3f} | - | - | - |")
        for sch in PLANNERS:
            if sch in cell:
                v = cell[sch]
                tp = (f"{v['vs_true_perfect']['mean_diff']:+.3f} ({v['vs_true_perfect']['wilcoxon_p_two_sided']:.2g}) | "
                      f"{v['vs_true_perfect_1s_after_ue_wrap']['mean_diff']:+.3f} ({v['vs_true_perfect_1s_after_ue_wrap']['wilcoxon_p_two_sided']:.2g})") if "vs_true_perfect" in v else "- | -"
                print(f"| {sch} | {v['outage_1s_after_ue_wrap']:.3f} | {v['gap_vs_A5_1s_after_ue_wrap']['mean_diff']:+.3f} ({v['gap_vs_A5_1s_after_ue_wrap']['wilcoxon_p_two_sided']:.2g}) | {tp} |")
        print()
        print("| scheme | gap by blocker class: " + " | ".join(CLASSES) + " |")
        print("|---|" + "---|" * len(CLASSES))
        for sch in PLANNERS:
            if sch in cell:
                v = cell[sch]
                print(f"| {sch} | " + " | ".join(f"{v['gap_by_class'][k]['mean_diff']:+.3f} ({v['gap_by_class'][k]['wilcoxon_p_two_sided']:.2g})" for k in CLASSES) + " |")
        print()


def print_part_d(o: dict) -> None:
    if not o.get("d_overhead0"):
        return
    print("### d) Sensing overhead 0 (human item 5): perfect-track planners vs A5 [s/UE-min, primary (wrap-masked)]\n")
    print("| margin | setting | scheme | outage | A5 | diff vs A5 [CI] (p) | parameters |")
    print("|---|---|---|---|---|---|---|")
    for lab, cell in o["d_overhead0"].items():
        for key, v in cell.items():
            name, sch = key.split(" | ")
            va = v["vs_A5"]
            print(f"| {lab} | {name} | {sch} | {v['outage']:.3f} | {v['A5']:.3f} | {va['mean_diff']:+.3f} [{va['ci95_boot'][0]:+.3f}, {va['ci95_boot'][1]:+.3f}] "
                  f"({va['wilcoxon_p_two_sided']:.2g}) | {v['params']} |")
    print()


def print_part_e(o: dict) -> None:
    e = o.get("e_future_truth")
    if not e:
        return
    pv = lambda v: f"{v['mean_diff']:+.3f} ({v['wilcoxon_p_two_sided']:.2g})"  # noqa: E731
    print("### e) Trajectory predictability vs sensing accuracy (human item 4): truth UE, sensing overhead 0, re-tuned at overhead 0 "
          "[s/UE-min, primary (wrap-masked); seed-level mean difference (exact Wilcoxon p)]\n")
    print("| margin | A5 | (i) perfect at decision, extrapolated | (ii) true future trajectories | (i) vs A5 | (ii) vs A5 | (ii) vs (i) |")
    print("|---|---|---|---|---|---|---|")
    for lab, cell in e.items():
        if "planner_true_future" not in cell or "planner_true_perfect" not in cell:
            continue
        i_, ii = cell["planner_true_perfect"], cell["planner_true_future"]
        print(f"| {lab} | {cell['A5']['outage']:.3f} | {i_['outage']:.3f} | {ii['outage']:.3f} | {pv(i_['gap_vs_A5'])} | {pv(ii['gap_vs_A5'])} | {pv(ii['vs_true_perfect'])} |")
    print()
    print("| margin | scheme | gap vs A5 by blocker class: bus/truck | pedestrian | car | no LoS blocker | none |")
    print("|---|---|---|---|---|---|---|")
    for lab, cell in e.items():
        for sch in ("planner_true_perfect", "planner_true_future"):
            if sch in cell and "gap_by_class" in cell[sch]:
                print(f"| {lab} | {sch} | " + " | ".join(pv(cell[sch]["gap_by_class"][k]) for k in CLASSES) + " |")
        if "planner_true_future" in cell and "vs_true_perfect_by_class" in cell["planner_true_future"]:
            print(f"| {lab} | (ii) - (i) | " + " | ".join(pv(cell["planner_true_future"]["vs_true_perfect_by_class"][k]) for k in CLASSES) + " |")
    print()


if __name__ == "__main__":
    main()
