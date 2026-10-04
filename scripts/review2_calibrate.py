"""ADDED AFTER THE SECOND EXTERNAL REVIEW: calibrate the realistic tracker-error model.

Development seeds only (1001-1010, the former evaluation seeds; 40 jobs).
The final map tracker (detector budget 4, image-method ghost handling,
configs/m3.yaml sensing.map_tracker) is replayed from the stage-validated
detection caches, exactly as the sensing-planner sees it (confirmed tracks;
lane / sidewalk tracks pinned to the line). Every 0.1 s CPI, confirmed
tracks are matched to the ground truth inside the 40 m sensing range with
the class gates of sim/sensing/metrics.py (greedy nearest pairs). Measured:

- state error of matched tracks (track - truth), per class, on x (= along
  every lane / sidewalk), y (cross), vx, vy: mean, standard deviation, and
  the lag-k autocorrelation (k = 1..10 CPIs) pooled over contiguous runs of
  the same (truth, track) pair; AR(1) coefficient phi = lag-1 value and
  correlation time -0.1 s / ln(phi);
- per ground-truth visit of the range (consecutive CPIs inside it) the
  matched / unmatched pattern (resampled by the realistic injection), and the
  initial acquisition delay (CPIs from entering the range to the first
  match), the outage durations after a match, and the matched-run
  durations (all in CPIs; runs cut by the end of the record are kept and
  flagged as censored);
- share of matched CPIs in free (unconstrained) mode per class;
- false tracks (unmatched confirmed tracks): episodes = contiguous CPIs of
  one track id being unmatched; type fragment (inside a true target's box
  + 1 m in the majority of its CPIs) or other (ghost + other); lifetime
  distribution and mean count per CPI per type; fragments store their
  offset to the target (and the target class), others their absolute
  state. Episode libraries are kept per mount (the radar position differs).

Writes results/M5/review2/calibration.json (summary statistics and the
empirical distributions / episode libraries used by
scripts/review2_inject.py).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

BUDGET = 4
MAX_LAG = 10
COMPONENTS = ("x", "y", "vx", "vy")
OUT = ROOT / "results" / "M5" / "review2" / "calibration.json"


def replay_rows(job: tuple, budget: int = BUDGET, covariance: bool = False) -> list[list[dict[str, Any]]]:
    """Confirmed map-tracker rows per CPI (same tracker and parameters as xapp.tracks.replay_map), with ids.

    With ``covariance`` the diagonal of the EKF covariance [x, y, vx, vy]
    (m^2, m^2/s^2) is added; pinned axes carry 1e-4 as set by the tracker.
    """
    from sim.scenes.config import load_yaml
    from sim.sensing.cluster import cluster_detections
    from sim.sensing.ghost import reject_ghosts
    from sim.sensing.provenance import require_detection_stage
    from sim.sensing.track_map import MapTracker
    from xapp.tracks import load_detections

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    seed, mount, density = job
    det = load_detections(ROOT / "results" / "cache" / mount / density / f"seed_{seed}")
    require_detection_stage(det)
    spec = raw["sensing_radar"]
    dd = cfg["sensing"]["budgets"][budget]
    tuned = cfg["sensing"]["map_tracker"]
    tracker = MapTracker(
        dt_s=0.1, association_gate_m=float(tuned["association_m"]), coast_frames=int(tuned["coast"]),
        confirm_hits=int(spec["tracker"]["confirm_hits"]), radar_position_m=np.asarray(det["radar_position_m"], dtype=np.float64),
        process_q=float(tuned["process_q"]), lanes=list(raw["lanes"]), sidewalks=list(raw["sidewalks"]),
        lane_gate_m=float(tuned["lane_gate_m"]), sidewalk_gate_m=float(tuned["sidewalk_gate_m"]), leave_m=float(tuned["leave_m"]),
    )
    key = f"blind:{int(dd['train'])}:{float(dd['pfa'])}"
    walls = [float(v) for v in spec["wall_y_m"]]
    out = []
    for frame in det["frames"]:
        detections = reject_ghosts(list(frame["detections"][key]), walls, float(dd["ghost_association_m"]))
        clusters = cluster_detections(detections, float(dd["eps_m"]), int(spec["cluster"]["min_samples"]))
        rows = []
        for t in tracker.step(clusters):
            if not t.confirmed:
                continue
            row = {"id": int(t.identifier), "x_m": float(t.state[0]), "y_m": float(t.state[1]), "z_m": float(t.state[2]),
                   "vx_mps": float(t.state[3]), "vy_mps": float(t.state[4]), "confirmed": True, "mode": str(getattr(t, "mode", "free")),
                   "line_y_m": getattr(t, "line_y_m", None), "age_s": float(t.age_s)}
            if covariance:
                row["var"] = [float(t.covariance[i, i]) for i in (0, 1, 3, 4)]
            rows.append(row)
        out.append(rows)
    return out


def _predictor_state(row: dict) -> np.ndarray:
    """The state the predictor uses (xapp.predict_torch.pack_tracks pins lane / sidewalk tracks)."""
    x, y, vx, vy = row["x_m"], row["y_m"], row["vx_mps"], row["vy_mps"]
    if row.get("mode") in ("lane", "sidewalk") and row.get("line_y_m") is not None:
        y, vy = float(row["line_y_m"]), 0.0
    return np.array([x, y, vx, vy], dtype=np.float64)


def _job(item: tuple) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from sim.scenes.config import load_yaml
    from sim.sensing.metrics import CLASS_GATES_M, _in_box, _match_classes, _xy, blocker_class, split_false
    from xapp.tracks import load_detections

    job, = item
    seed, mount, density = job
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    sizes = raw["blocker_kinds"]
    walls = [float(v) for v in raw["sensing_radar"]["wall_y_m"]]
    det = load_detections(ROOT / "results" / "cache" / mount / density / f"seed_{seed}")
    frames = replay_rows(job)
    gates = {c: float(v) for c, v in CLASS_GATES_M.items()}
    matched_err: list[dict] = []  # per matched CPI
    presence: dict[str, list[int]] = {}  # gt id -> list of (frame, matched)
    gt_class: dict[str, str] = {}
    false_rows: list[dict] = []
    for f, (frame, rows) in enumerate(zip(det["frames"], frames)):
        gt = frame["ground_truth"]
        gxy = _xy(gt)
        txy = _xy(rows)
        pairs = _match_classes(gt, gxy, txy, gates)
        tmatch = {j: i for i, j in pairs}
        gmatch = {i for i, _ in pairs}
        for i, g in enumerate(gt):
            gid = str(g["id"])
            gt_class[gid] = blocker_class(str(g["kind"]))
            presence.setdefault(gid, []).append((f, int(i in gmatch)))
        for j, row in enumerate(rows):
            st = _predictor_state(row)
            if j in tmatch:
                g = gt[tmatch[j]]
                e = st - np.array([g["x_m"], g["y_m"], g["vx_mps"], g["vy_mps"]])
                matched_err.append({"f": f, "gt": str(g["id"]), "trk": row["id"], "cls": blocker_class(str(g["kind"])), "e": e.tolist(),
                                    "free": row.get("mode", "free") == "free"})
            else:
                frag, ghost, _other = split_false([row], gt, sizes, walls)
                rec = {"f": f, "trk": row["id"], "type": "fragment" if frag else "other", "mode": row.get("mode", "free"), "state": st.tolist()}
                if frag:
                    point = np.array([row["x_m"], row["y_m"]])
                    inside = [g for g in gt if _in_box(point, g, sizes)]
                    g = min(inside, key=lambda g: (g["x_m"] - point[0]) ** 2 + (g["y_m"] - point[1]) ** 2)
                    rec["target_cls"] = blocker_class(str(g["kind"]))
                    rec["offset"] = (st - np.array([g["x_m"], g["y_m"], g["vx_mps"], g["vy_mps"]])).tolist()
                false_rows.append(rec)
    return {"job": job, "n_frames": len(frames), "matched": matched_err, "presence": presence, "gt_class": gt_class, "false": false_rows}


def _runs(seq: list[tuple[int, int]]) -> list[tuple[int, int, bool, bool]]:
    """(value, length, first, censored_at_end) for runs over CONSECUTIVE frames; a frame gap (out of range) restarts."""
    out = []
    i = 0
    while i < len(seq):
        start_f, v = seq[i]
        j = i
        while j + 1 < len(seq) and seq[j + 1][1] == v and seq[j + 1][0] == seq[j][0] + 1:
            j += 1
        first = i == 0 or seq[i][0] != seq[i - 1][0] + 1
        last = j == len(seq) - 1 or seq[j + 1][0] != seq[j][0] + 1
        out.append((v, j - i + 1, first, last))
        i = j + 1
    return out


def summarize(parts: list[dict]) -> dict[str, Any]:
    from sim.sensing.metrics import CLASSES

    n_frames = sum(p["n_frames"] for p in parts)
    err: dict[str, Any] = {}
    for c in CLASSES:
        rows = [(p["job"], m) for p in parts for m in p["matched"] if m["cls"] == c]
        if not rows:
            continue
        e = np.array([m["e"] for _, m in rows])
        mean, std = e.mean(0), e.std(0)
        # autocorrelation over contiguous (job, truth, track) runs
        acc = np.zeros((MAX_LAG + 1, 4))
        cnt = np.zeros(MAX_LAG + 1)
        seqs: dict[tuple, list] = {}
        for job, m in rows:
            seqs.setdefault((tuple(job), m["gt"], m["trk"]), []).append((m["f"], np.array(m["e"])))
        for s in seqs.values():
            s.sort(key=lambda t: t[0])
            # split into contiguous runs
            run = [s[0]]
            for a in s[1:] + [None]:
                if a is not None and a[0] == run[-1][0] + 1:
                    run.append(a)
                    continue
                x = np.array([r[1] for r in run]) - mean
                for k in range(min(MAX_LAG, len(x) - 1) + 1):
                    acc[k] += (x[: len(x) - k] * x[k:]).sum(0)
                    cnt[k] += len(x) - k
                if a is not None:
                    run = [a]
        cov = acc / np.maximum(cnt, 1)[:, None]
        acf = cov / np.where(cov[0] > 0, cov[0], 1.0)
        phi = np.clip(acf[1], 0.0, 0.999)
        err[c] = {
            "n_matched_cpi": int(len(rows)),
            "free_mode_share": float(np.mean([m["free"] for _, m in rows])),
            "mean": dict(zip(COMPONENTS, mean.tolist())), "std": dict(zip(COMPONENTS, std.tolist())),
            "rms": dict(zip(COMPONENTS, np.sqrt((e ** 2).mean(0)).tolist())),
            "acf": {comp: acf[:, i].tolist() for i, comp in enumerate(COMPONENTS)},
            "phi": dict(zip(COMPONENTS, phi.tolist())),
            "corr_time_s": {comp: (float(-0.1 / math.log(v)) if 0 < v < 1 else None) for comp, v in zip(COMPONENTS, phi.tolist())},
            "acf_lag5_vs_phi5": {comp: [float(acf[5, i]), float(phi[i] ** 5)] for i, comp in enumerate(COMPONENTS)},
        }
    outage: dict[str, Any] = {}
    for c in CLASSES:
        acq, outs, ons, acq_c, outs_c, ons_c = [], [], [], [], [], []
        matched = total = 0
        visits: list[str] = []
        for p in parts:
            for gid, seq in p["presence"].items():
                if p["gt_class"][gid] != c:
                    continue
                matched += sum(v for _, v in seq)
                total += len(seq)
                cur = [seq[0]]
                for a in seq[1:] + [None]:
                    if a is not None and a[0] == cur[-1][0] + 1:
                        cur.append(a)
                        continue
                    visits.append("".join(str(v) for _, v in cur))
                    if a is not None:
                        cur = [a]
                for v, n, first, last in _runs(seq):
                    if v == 0 and first:
                        acq.append(n)
                        acq_c.append(int(last))
                    elif v == 0:
                        outs.append(n)
                        outs_c.append(int(last))
                    else:
                        ons.append(n)
                        ons_c.append(int(last))
        outage[c] = {"track_pd": matched / max(total, 1), "visits": visits, "acquisition_cpi": acq, "outage_cpi": outs, "matched_run_cpi": ons,
                     "censored_flags": {"acquisition": acq_c, "outage": outs_c, "matched_run": ons_c},
                     "censored": {"acquisition": sum(acq_c), "outage": sum(outs_c), "matched_run": sum(ons_c)},
                     "median": {"acquisition": float(np.median(acq)) if acq else None, "outage": float(np.median(outs)) if outs else None,
                                "matched_run": float(np.median(ons)) if ons else None},
                     "mean": {"acquisition": float(np.mean(acq)) if acq else None, "outage": float(np.mean(outs)) if outs else None,
                              "matched_run": float(np.mean(ons)) if ons else None}}
    false: dict[str, Any] = {"per_mount": {}}
    tot = {"fragment": 0, "other": 0}
    for mount in sorted({p["job"][1] for p in parts}):
        lib = {"fragment": [], "other": []}
        nf = 0
        for p in parts:
            if p["job"][1] != mount:
                continue
            nf += p["n_frames"]
            by: dict[int, list] = {}
            for r in p["false"]:
                by.setdefault(r["trk"], []).append(r)
            for rs in by.values():
                rs.sort(key=lambda r: r["f"])
                run = [rs[0]]
                for a in rs[1:] + [None]:
                    if a is not None and a["f"] == run[-1]["f"] + 1:
                        run.append(a)
                        continue
                    n_frag = sum(r["type"] == "fragment" for r in run)
                    if 2 * n_frag >= len(run):
                        fr = [r for r in run if r["type"] == "fragment"]
                        cls = max(set(r["target_cls"] for r in fr), key=[r["target_cls"] for r in fr].count)
                        # offsets for every CPI of the episode; non-fragment CPIs reuse the last fragment offset
                        offs, last = [], fr[0]["offset"]
                        for r in run:
                            last = r.get("offset", last)
                            offs.append([round(v, 4) for v in last])
                        lib["fragment"].append({"target_cls": cls, "mode": run[0]["mode"], "offset": offs})
                    else:
                        lib["other"].append({"mode": [r["mode"] for r in run], "state": [[round(v, 4) for v in r["state"]] for r in run]})
                    if a is not None:
                        run = [a]
        stats = {}
        for t in ("fragment", "other"):
            life = [len(e["offset"] if t == "fragment" else e["state"]) for e in lib[t]]
            stats[t] = {"episodes": len(life), "count_per_cpi": sum(life) / nf, "mean_lifetime_cpi": float(np.mean(life)) if life else 0.0,
                        "median_lifetime_cpi": float(np.median(life)) if life else 0.0,
                        "birth_rate_per_cpi": len(life) / nf, "lifetime_cpi": life}
            tot[t] += sum(life)
        false["per_mount"][mount] = {"n_cpi": nf, "stats": stats, "library": lib}
    false["count_per_cpi"] = {t: v / n_frames for t, v in tot.items()}
    false["fragment_share"] = tot["fragment"] / max(1, tot["fragment"] + tot["other"])
    return {"definition": __doc__, "budget": BUDGET, "n_cpi": n_frames, "errors": err, "outages": outage, "false_tracks": false}


def main() -> None:
    from sim.scenes.config import load_yaml

    import os

    if os.environ.get("S2C_EVAL_SET", "dev") != "dev":
        raise SystemExit("the tracker-error calibration is frozen on the development seeds; do not run it on held-out seeds")
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    dev = [int(s) for s in seeds["evaluation"]]  # development seeds 1001-1010 (always, never the held-out set)
    jobs = [(s, m, d) for s in dev for m in ("lamppost", "facade") for d in ("low", "high")]
    with mp.get_context("spawn").Pool(4) as pool:
        parts = pool.map(_job, [(j,) for j in jobs], chunksize=1)
    cal = summarize(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cal) + "\n")
    for c, v in cal["errors"].items():
        o = cal["outages"][c]
        print(f"{c}: n={v['n_matched_cpi']} free={v['free_mode_share']:.2f} std " + " ".join(f"{k}={v['std'][k]:.2f}" for k in COMPONENTS)
              + " phi " + " ".join(f"{k}={v['phi'][k]:.2f}" for k in COMPONENTS)
              + f" | Pd={o['track_pd']:.3f} acq med {o['median']['acquisition']} outage med {o['median']['outage']} mean {o['mean']['outage']} on med {o['median']['matched_run']}", flush=True)
    ft = cal["false_tracks"]
    print("false tracks per CPI", ft["count_per_cpi"], "fragment share", round(ft["fragment_share"], 3))
    for m, v in ft["per_mount"].items():
        print(m, {t: {k: round(s[k], 3) for k in ("count_per_cpi", "mean_lifetime_cpi", "median_lifetime_cpi", "birth_rate_per_cpi")} for t, s in v["stats"].items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
