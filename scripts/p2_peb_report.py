"""Paper 2 (P2-M1 / M4): bound statistics and the pre-specified tests of P1-P3.

Reads results/P2/peb/<set>/*.npz (scripts/p2_peb.py). Unit = SEED: each
quantity is first computed per seed over its 4 runs (2 mounts x 2
densities), 2 UEs and 600 epochs; paired comparisons use the exact
two-sided Wilcoxon signed-rank test over the 10 seeds and 95 % percentile
bootstrap CIs over seeds (10,000 resamples, configs/p2.yaml bootstrap
seed). Pointwise per-condition tests are exploratory.

Epoch state per UE: number of O-RUs whose LoS has model-B loss >= 10 dB
(0 = LoS to both, 1, 2). "Blocked" = at least one.

MAIN configuration for the claims (fixed before any held-out data):
timing tdoa (unknown UE clock bias), sigma_sync 1 ns (constant per run),
sigma_phi 2 deg, blocked-LoS variant "biased" (b), SNR scale 1; P1 at every
bandwidth; P2 with LoS-only information (what a non-map receiver has), also
reported for map-aided and variant (a); P3 compares map-aided with LoS-only.

Decision rules (pre-specified; "supported" / "not supported" / "mixed"):
- P1 "decimeter only with wide bandwidth and LoS to both": share of epochs
  with PEB <= 0.1 m (seed mean). Supported if the share is >= 0.9 at 400 MHz
  with LoS to both AND <= 0.5 both at 100 MHz with LoS to both and at 400 MHz
  with >= 1 blocked O-RU (each difference significant at seed level). Not
  supported if the share stays >= 0.9 in either of those two conditions,
  or is < 0.9 at 400 MHz with LoS to both. Mixed otherwise. Evaluated for
  LoS-only and map-aided information separately.
- P2 "accuracy degrades sharply during blockage": per seed the ratio of
  the median PEB over blocked epochs to the median PEB over LoS-to-both
  epochs. Supported if the seed-level median ratio is >= 2 and the paired
  test (log medians, blocked vs LoS) gives p < 0.05; not supported if the
  ratio is < 2 or p >= 0.05 with the ratio < 1.25; mixed otherwise.
- P3 "map-aided NLoS recovers part of the loss during blockage": per seed
  the median PEB over blocked epochs, map-aided vs LoS-only. Supported if
  map-aided is lower with p < 0.05 and the recovered share of the
  log-gap (log PEB_LoS,blocked - log PEB_map,blocked) / (log
  PEB_LoS,blocked - log PEB_LoS,LoS-epochs) is > 0; not supported if p >= 0.05
  or map-aided is not lower.
Also: shares <= 0.1 m and <= the paper-1 break-even range (0.08 / 0.13 m),
CDF quantiles, PEB along the sidewalk (UE x bins), splits by bandwidth,
mount, density, timing, hardware and variant, and the limit analysis (which
of bandwidth, SNR, sync or calibration error limits the PEB: median PEB
when each is relaxed alone: 400 MHz, SNR x10, sigma_sync 0, sigma_phi 0).

Writes results/P2/peb_<set>.json and results/P2/peb_<set>.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

AX = {"bw": ["100", "200", "400"], "snr": [1.0, 10.0], "info": ["los", "map"], "blocked": ["kept", "biased"], "timing": ["toa", "tdoa", "aoa"],
      "sync": [0.0, 0.3, 1.0, 3.0], "phi": [0.0, 2.0, 5.0]}
MAIN = {"bw": "400", "snr": 1.0, "info": "los", "blocked": "biased", "timing": "tdoa", "sync": 1.0, "phi": 2.0}
X_BINS = np.arange(-40.0, 40.01, 5.0)


def idx(**kw) -> tuple:
    c = {**MAIN, **kw}
    return tuple(AX[k].index(c[k]) for k in ("bw", "snr", "info", "blocked", "timing", "sync", "phi"))


def load(tag: str) -> dict[str, Any]:
    files = sorted((ROOT / "results" / "P2" / "peb" / tag).glob("*_*_*.npz"))
    runs = {}
    for f in files:
        mount, density, seed = f.stem.split("_")
        d = np.load(f)
        nb = (d["los_loss_db"] >= 10.0).sum(-1)  # [T, U]
        runs[(int(seed), mount, density)] = {"peb": d["peb"], "nb": nb, "x": d["ue"][..., 0]}
    return runs


def boot_ci(v: np.ndarray, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def wil(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if np.count_nonzero(d) else float("nan")


def per_seed(runs: dict, fn) -> np.ndarray:
    seeds = sorted({k[0] for k in runs})
    return np.array([fn([runs[k] for k in runs if k[0] == s]) for s in seeds], dtype=float)


def pooled(rs: list, ix: tuple, mask_fn) -> np.ndarray:
    vals = []
    for r in rs:
        v = r["peb"][(slice(None), slice(None)) + ix]
        vals.append(v[mask_fn(r)])
    return np.concatenate(vals) if vals else np.zeros(0)


def share(rs, ix, thr, mask_fn) -> float:
    v = pooled(rs, ix, mask_fn)
    return float(np.mean(v <= thr)) if v.size else float("nan")


def median(rs, ix, mask_fn) -> float:
    v = pooled(rs, ix, mask_fn)
    return float(np.median(v)) if v.size else float("nan")


ALL = lambda r: np.ones_like(r["nb"], dtype=bool)  # noqa: E731
LOS2 = lambda r: r["nb"] == 0  # noqa: E731
BLK = lambda r: r["nb"] >= 1  # noqa: E731


def verdict_p1(s400, s100, sblk, p100, pblk) -> str:
    if np.mean(s400) < 0.9 or np.mean(s100) >= 0.9 or np.mean(sblk) >= 0.9:
        return "not supported"
    if np.mean(s100) <= 0.5 and np.mean(sblk) <= 0.5 and p100 < 0.05 and pblk < 0.05:
        return "supported"
    return "mixed"


def main() -> None:
    from sim.scenes.config import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    nb_, bseed = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    runs = load(args.set)
    seeds = sorted({k[0] for k in runs})
    out: dict[str, Any] = {"set": args.set, "seeds": seeds, "n_runs": len(runs), "main": MAIN, "definition": __doc__}
    L = [f"# P2 bounds ({args.set} seeds {seeds[0]}-{seeds[-1]}, {len(runs)} runs)", "",
         "Unit = seed (10 seeds x 4 runs x 2 UEs x 600 epochs). Main configuration: " + ", ".join(f"{k} {v}" for k, v in MAIN.items()) + ".", ""]
    # ---------------------------------------------------------------- shares and quantiles per condition
    rows = []
    for info in AX["info"]:
        for bw in AX["bw"]:
            for state, fn in (("all", ALL), ("LoS both", LOS2), ("blocked", BLK)):
                ix = idx(info=info, bw=bw)
                s01 = per_seed(runs, lambda rs: share(rs, ix, 0.1, fn))
                s08 = per_seed(runs, lambda rs: share(rs, ix, 0.08, fn))
                s13 = per_seed(runs, lambda rs: share(rs, ix, 0.13, fn))
                med = per_seed(runs, lambda rs: median(rs, ix, fn))
                rows.append({"info": info, "bw": bw, "state": state, "share_le_0.1": float(np.mean(s01)), "ci": boot_ci(s01, nb_, bseed),
                             "share_le_0.08": float(np.mean(s08)), "share_le_0.13": float(np.mean(s13)), "median_peb_m": float(np.median(med))})
    out["shares_main_hw"] = rows
    L += ["## Share of epochs with PEB <= 0.1 m (seed mean [95 % bootstrap CI]) and <= the paper-1 break-even range; main hardware", "",
          "| Info | BW | State | <= 0.1 m | <= 0.08 / 0.13 m | median PEB [m] |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['info']} | {r['bw']} | {r['state']} | {r['share_le_0.1']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['share_le_0.08']:.3f} / {r['share_le_0.13']:.3f} | {r['median_peb_m']:.4f} |")
    # ---------------------------------------------------------------- hardware sweep
    hw = []
    for info in AX["info"]:
        for tm in AX["timing"]:
            for sy in AX["sync"]:
                for ph in AX["phi"]:
                    ix = idx(info=info, timing=tm, sync=sy, phi=ph)
                    s01 = per_seed(runs, lambda rs: share(rs, ix, 0.1, ALL))
                    med = per_seed(runs, lambda rs: median(rs, ix, ALL))
                    hw.append({"info": info, "timing": tm, "sync_ns": sy, "phi_deg": ph, "share_le_0.1": float(np.mean(s01)), "median_peb_m": float(np.median(med))})
    out["hardware_sweep_400MHz_biased"] = hw
    L += ["", "## Hardware sweep (400 MHz, blocked variant b, all epochs): share <= 0.1 m / median PEB [m]", "",
          "| Info | Timing | sync \\ phi | " + " | ".join(f"{p:g} deg" for p in AX["phi"]) + " |", "|---|---|---|" + "---|" * len(AX["phi"])]
    for info in AX["info"]:
        for tm in AX["timing"]:
            for sy in AX["sync"]:
                cells = [next(h for h in hw if h["info"] == info and h["timing"] == tm and h["sync_ns"] == sy and h["phi_deg"] == ph) for ph in AX["phi"]]
                L.append(f"| {info} | {tm} | {sy:g} ns | " + " | ".join(f"{c['share_le_0.1']:.3f} / {c['median_peb_m']:.4f}" for c in cells) + " |")
    # sync constant vs per-epoch: identical in a per-epoch (snapshot) bound
    L += ["", "The per-epoch (snapshot) bound is the same for a sync error drawn once per run (main case) and per epoch (sensitivity); "
          "the two differ only for the estimator + EKF (P2-M2).", ""]
    # ---------------------------------------------------------------- limit analysis
    lim = {}
    for info in AX["info"]:
        base_ix = idx(info=info, bw="200")
        base = per_seed(runs, lambda rs: median(rs, base_ix, ALL))
        relax = {"bandwidth 200 -> 400 MHz": idx(info=info, bw="400"), "SNR x10": idx(info=info, bw="200", snr=10.0),
                 "sigma_sync -> 0": idx(info=info, bw="200", sync=0.0), "sigma_phi -> 0": idx(info=info, bw="200", phi=0.0)}
        lim[info] = {"base_median_m": float(np.median(base))}
        for name, ix in relax.items():
            v = per_seed(runs, lambda rs: median(rs, ix, ALL))
            lim[info][name] = {"median_m": float(np.median(v)), "ratio_to_base": float(np.median(v / base)), "p": wil(np.log(v) - np.log(base))}
        lim[info]["limiting"] = min((k for k in relax), key=lambda k: lim[info][k]["ratio_to_base"])
    out["limit_analysis_200MHz"] = lim
    L += ["## Which limit dominates (200 MHz, main hardware; median PEB when one limit is relaxed alone; seed-level ratio to the base, Wilcoxon p)", ""]
    for info, v in lim.items():
        L.append(f"- {info}: base {v['base_median_m']:.4f} m; " + "; ".join(f"{k}: {v[k]['median_m']:.4f} m (x{v[k]['ratio_to_base']:.2f}, p={v[k]['p']:.3g})"
                                                                          for k in v if isinstance(v[k], dict)) + f" -> limiting: {v['limiting']}")
    # ---------------------------------------------------------------- P1
    p1 = {}
    for info in AX["info"]:
        s400 = per_seed(runs, lambda rs: share(rs, idx(info=info, bw="400"), 0.1, LOS2))
        s100 = per_seed(runs, lambda rs: share(rs, idx(info=info, bw="100"), 0.1, LOS2))
        sblk = per_seed(runs, lambda rs: share(rs, idx(info=info, bw="400"), 0.1, BLK))
        p100, pblk = wil(s400 - s100), wil(s400 - sblk)
        p1[info] = {"share_400_los": float(np.mean(s400)), "share_100_los": float(np.mean(s100)), "share_400_blocked": float(np.mean(sblk)),
                    "ci_400_los": boot_ci(s400, nb_, bseed), "ci_100_los": boot_ci(s100, nb_, bseed), "ci_400_blocked": boot_ci(sblk, nb_, bseed),
                    "p_400_vs_100": p100, "p_los_vs_blocked": pblk, "verdict": verdict_p1(s400, s100, sblk, p100, pblk)}
    out["P1"] = p1
    # ---------------------------------------------------------------- P2
    p2 = {}
    for info in AX["info"]:
        for bv in AX["blocked"]:
            for bw in AX["bw"]:
                ix = idx(info=info, blocked=bv, bw=bw)
                mb = per_seed(runs, lambda rs: median(rs, ix, BLK))
                ml = per_seed(runs, lambda rs: median(rs, ix, LOS2))
                ratio = mb / ml
                p = wil(np.log(mb) - np.log(ml))
                rmed = float(np.median(ratio))
                v = "supported" if (rmed >= 2 and p < 0.05) else ("not supported" if (rmed < 1.25 or (p >= 0.05 and rmed < 2)) else "mixed")
                p2[f"{info}|{bv}|{bw}"] = {"median_ratio": rmed, "ci_ratio": boot_ci(ratio, nb_, bseed), "p": p, "median_blocked_m": float(np.median(mb)),
                                            "median_los_m": float(np.median(ml)), "verdict": v}
    out["P2"] = p2
    # ---------------------------------------------------------------- P3
    p3 = {}
    for bv in AX["blocked"]:
        for bw in AX["bw"]:
            mlos = per_seed(runs, lambda rs: median(rs, idx(info="los", blocked=bv, bw=bw), BLK))
            mmap = per_seed(runs, lambda rs: median(rs, idx(info="map", blocked=bv, bw=bw), BLK))
            mlos_l = per_seed(runs, lambda rs: median(rs, idx(info="los", blocked=bv, bw=bw), LOS2))
            p = wil(np.log(mmap) - np.log(mlos))
            gap = np.log(mlos) - np.log(mlos_l)
            rec = np.where(gap > 0, (np.log(mlos) - np.log(mmap)) / np.where(gap > 0, gap, 1), np.nan)
            lower = float(np.median(mmap / mlos)) < 1
            v = "supported" if (lower and p < 0.05 and np.nanmedian(rec) > 0) else "not supported"
            p3[f"{bv}|{bw}"] = {"median_ratio_map_to_los": float(np.median(mmap / mlos)), "ci": boot_ci(mmap / mlos, nb_, bseed), "p": p,
                                "recovered_share_median": float(np.nanmedian(rec)), "verdict": v}
    out["P3"] = p3
    L += ["", "## Hypotheses (pre-specified rules; seed level)", ""]
    for info, v in p1.items():
        L.append(f"- P1 ({info}): share <= 0.1 m: 400 MHz LoS-both {v['share_400_los']:.3f} {v['ci_400_los']}, 100 MHz LoS-both {v['share_100_los']:.3f}, "
                 f"400 MHz blocked {v['share_400_blocked']:.3f}; p(400 vs 100) {v['p_400_vs_100']:.3g}, p(LoS vs blocked) {v['p_los_vs_blocked']:.3g} -> **{v['verdict']}**")
    for k, v in p2.items():
        if k.endswith("|400"):
            L.append(f"- P2 ({k}): median PEB blocked {v['median_blocked_m']:.4f} vs LoS {v['median_los_m']:.4f} m, ratio {v['median_ratio']:.2f} {v['ci_ratio']}, "
                     f"p {v['p']:.3g} -> **{v['verdict']}**")
    for k, v in p3.items():
        if k.endswith("|400"):
            L.append(f"- P3 ({k}): map/LoS-only median PEB during blockage {v['median_ratio_map_to_los']:.3f} {v['ci']}, p {v['p']:.3g}, "
                     f"recovered share {v['recovered_share_median']:.2f} -> **{v['verdict']}**")
    # ---------------------------------------------------------------- along the sidewalk and splits
    prof = {}
    for info in AX["info"]:
        ix = idx(info=info)
        vals = [[] for _ in range(len(X_BINS) - 1)]
        for r in runs.values():
            v = r["peb"][(slice(None), slice(None)) + ix]
            b = np.clip(np.digitize(r["x"], X_BINS) - 1, 0, len(X_BINS) - 2)
            for k in range(len(X_BINS) - 1):
                vals[k].append(v[b == k])
        prof[info] = [{"x_center_m": float((X_BINS[k] + X_BINS[k + 1]) / 2), "median_m": float(np.median(np.concatenate(vals[k]))) if sum(len(a) for a in vals[k]) else None,
                       "p90_m": float(np.percentile(np.concatenate(vals[k]), 90)) if sum(len(a) for a in vals[k]) else None} for k in range(len(X_BINS) - 1)]
    out["profile_main"] = prof
    split = {}
    for key in ("lamppost", "facade", "low", "high"):
        sub = {k: v for k, v in runs.items() if key in k}
        split[key] = {info: float(np.mean(per_seed(sub, lambda rs: share(rs, idx(info=info), 0.1, ALL)))) for info in AX["info"]}
    out["splits_share_le_0.1_main"] = split
    L += ["", "## Splits (main configuration, share <= 0.1 m)", ""] + [f"- {k}: " + ", ".join(f"{i} {v[i]:.3f}" for i in v) for k, v in split.items()]
    dest = ROOT / "results" / "P2" / f"peb_{args.set}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    (ROOT / "results" / "P2" / f"peb_{args.set}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
