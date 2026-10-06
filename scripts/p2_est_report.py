"""Paper 2 (P2-M2/M3/M4): estimator and closing-experiment statistics, and the pre-specified test of P4.

Unit = SEED (per-seed values over its 4 runs x 2 UEs x 600 epochs; the first
2 s of each run are excluded as EKF start-up). Error = |EKF position - true
UE position| (horizontal). Error-to-PEB ratio per epoch against the LoS-only
bound of the SAME configuration (bandwidth, timing, sigma_sync, sigma_phi,
blocked variant: diffracted <-> biased, kept <-> kept), since the estimator
uses the dominant (LoS) path only.

P4 (pre-specified):
- part 1 "a practical estimator stays well above the bound": at the main
  configuration (400 MHz, TDoA, sync 1 ns per run, phi 2 deg, blocked LoS
  diffracted), the seed medians of the per-epoch error/PEB ratio are > 5
  (exact Wilcoxon of log(ratio / 5) over the 10 seeds, p < 0.05, median
  above 5);
- part 2 "with realistic positioning error, the paper-1 planner does not
  beat A5": with the main estimator's positions (closing experiment), the
  planner is not significantly better than A5 (seed-level Wilcoxon p < 0.05
  with a negative mean) at any of 15-30 dB and the 3GPP reference.
P4 supported if both parts hold, not supported if neither, mixed otherwise.
Literature sanity check: median error at 100/200 MHz in the sub-metre regime.

Writes results/P2/est_<set>.json / .md (and closing statistics if present).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WARMUP = 20
MAIN = "bw400_tdoa_s1_p2_b"
CLAIM_MARGINS = ["15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference"]


def peb_index(cfg: dict) -> tuple:
    from p2_peb_report import idx

    return idx(info="los", bw=cfg["bw"], timing=cfg["timing"], sync=cfg["sync"], phi=cfg["phi"], blocked=cfg["blocked"])


def boot(v, n, seed):
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main() -> None:
    from p2_estimate import configs
    from sim.scenes.config import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--variant", default="v1", choices=["v1", "A", "AB"])
    args = ap.parse_args()
    vtag = "" if args.variant == "v1" else f"_{args.variant}"
    clabel = (lambda n: f"estimator {n}") if args.variant == "v1" else (lambda n: f"estimator {args.variant} {n}")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    nb, bs = int(p2cfg["bootstrap"]["n"]), int(p2cfg["bootstrap"]["seed"])
    base = ROOT / "results" / "P2" / ("est" if args.variant == "v1" else f"est_{args.variant}") / args.set
    mbase = ROOT / "results" / "P2" / "est" / args.set  # measurements (shared by all variants)
    cfgs = configs()
    out = {"set": args.set, "variant": args.variant, "definition": __doc__, "configs": {}}
    L = [f"# P2 estimator ({args.set} seeds)", "", "Seed-level values (mean over seeds of per-seed statistics; 95 % bootstrap CI over seeds). "
         "Error to PEB: LoS-only bound of the same configuration.", "",
         "| Config | median [m] | p90 [m] | RMSE [m] | median error/PEB | median, LoS both | median, blocked | share <= 0.1 m |", "|---|---|---|---|---|---|---|---|"]
    per_seed_err = {}
    for name, c in cfgs.items():
        files = sorted(base.glob(f"{name}/*_track.npz"))
        if not files:
            continue
        per = {}
        for f in files:
            mount, density, seed = f.stem.replace("_track", "").split("_")
            tr = np.load(f)
            ms = np.load(mbase / name / f.name.replace("_track", ""))
            err = np.linalg.norm(tr["xy_ekf"] - ms["ue"][..., :2], axis=-1)[WARMUP:]
            nb_ = ms["blocked"].sum(-1)[WARMUP:]
            pebf = ROOT / "results" / "P2" / "peb" / args.set / f"{mount}_{density}_{seed}.npz"
            peb = np.load(pebf)["peb"][(slice(None), slice(None)) + peb_index(c)][WARMUP:]
            per.setdefault(int(seed), []).append((np.nan_to_num(err, nan=1e3), nb_, peb))
        seeds = sorted(per)
        def stat(fn):
            return np.array([fn(np.concatenate([a[0].ravel() for a in per[s]]), np.concatenate([a[1].ravel() for a in per[s]]),
                                np.concatenate([a[2].ravel() for a in per[s]])) for s in seeds])
        med = stat(lambda e, n, p: np.median(e))
        p90 = stat(lambda e, n, p: np.percentile(e, 90))
        rmse = stat(lambda e, n, p: np.sqrt(np.mean(np.minimum(e, 100.0) ** 2)))
        ratio = stat(lambda e, n, p: np.median(e / np.maximum(p, 1e-6)))
        m_los = stat(lambda e, n, p: np.median(e[n == 0]) if (n == 0).any() else np.nan)
        m_blk = stat(lambda e, n, p: np.median(e[n >= 1]) if (n >= 1).any() else np.nan)
        s01 = stat(lambda e, n, p: np.mean(e <= 0.1))
        per_seed_err[name] = med
        r = {"cfg": c, "seeds": seeds, "median_m": float(np.mean(med)), "median_ci": boot(med, nb, bs), "p90_m": float(np.mean(p90)), "rmse_m": float(np.mean(rmse)),
             "err_to_peb_median": float(np.median(ratio)), "per_seed_err_to_peb": ratio.tolist(), "median_los_m": float(np.nanmean(m_los)),
             "median_blocked_m": float(np.nanmean(m_blk)), "share_le_0.1": float(np.mean(s01)), "per_seed_median_m": med.tolist()}
        out["configs"][name] = r
        L.append(f"| {name} | {r['median_m']:.3f} [{r['median_ci'][0]:.3f}, {r['median_ci'][1]:.3f}] | {r['p90_m']:.2f} | {r['rmse_m']:.2f} | {r['err_to_peb_median']:.1f} | "
                 f"{r['median_los_m']:.3f} | {r['median_blocked_m']:.3f} | {r['share_le_0.1']:.3f} |")
    # paired comparisons (seed level)
    def paired(a, b):
        d = per_seed_err[a] - per_seed_err[b]
        return {"mean_diff_m": float(d.mean()), "p": float(stats.wilcoxon(d).pvalue) if np.count_nonzero(d) else float("nan")}
    comps = {}
    for a, b in (("bw400_tdoa_s1_p2_b", "bw100_tdoa_s1_p2_b"), ("bw400_tdoa_s1_p2_b", "bw200_tdoa_s1_p2_b"), ("bw400_tdoa_s1e_p2_b", "bw400_tdoa_s1_p2_b"),
                 ("bw400_tdoa_s3e_p2_b", "bw400_tdoa_s3_p2_b"), ("bw400_tdoa_s0.3e_p2_b", "bw400_tdoa_s0.3_p2_b"), ("bw400_tdoa_s1_p2_a", "bw400_tdoa_s1_p2_b"),
                 ("bw400_tdoa_s0_p0_b", "bw400_tdoa_s1_p2_b"), ("bw400_toa_s1_p2_b", "bw400_tdoa_s1_p2_b"), ("bw400_aoa_s1_p2_b", "bw400_tdoa_s1_p2_b")):
        if a in per_seed_err and b in per_seed_err:
            comps[f"{a} - {b}"] = paired(a, b)
    out["paired_median_error"] = comps
    L += ["", "Paired seed-level differences of the median error [m] (exact Wilcoxon p):", ""] + [f"- {k}: {v['mean_diff_m']:+.3f} m, p = {v['p']:.3g}" for k, v in comps.items()]
    # P4 part 1
    if MAIN in out["configs"]:
        rat = np.array(out["configs"][MAIN]["per_seed_err_to_peb"])
        p = float(stats.wilcoxon(np.log(rat / 5.0)).pvalue)
        part1 = bool(np.median(rat) > 5 and p < 0.05)
        out["P4_part1"] = {"seed_ratios": rat.tolist(), "median": float(np.median(rat)), "p_vs_5": p, "holds": part1}
        lit = {k: out["configs"][f"bw{k}_tdoa_s1_p2_b"]["median_m"] for k in ("100", "200", "400") if f"bw{k}_tdoa_s1_p2_b" in out["configs"]}
        out["literature_check_median_m"] = lit
        L += ["", f"P4 part 1: median error/PEB per seed {np.round(rat, 1).tolist()}, median {np.median(rat):.1f}, p(>5) {p:.3g} -> holds: {part1}",
              f"Literature check (median error, TDoA, main hardware): {lit} (0.6-0.8 m RMSE reported for street-canyon SRS multi-gNB AoA/ToA)."]
    # closing experiment
    cf = ROOT / "results" / "P2" / f"closing_{args.set}.json"
    if cf.exists():
        cl = json.loads(cf.read_text())["conditions"]
        L += ["", "## Closing experiment: planner (perfect blocker tracks) minus A5 [s/UE-min], seed level (mean [bootstrap CI], exact Wilcoxon p; pointwise)", "",
              "| Condition | " + " | ".join(m.split(" ")[0] if "3GPP" not in m else "Ref" for m in CLAIM_MARGINS) + " |", "|---|" + "---|" * len(CLAIM_MARGINS)]
        for cname, res in cl.items():
            bl = {m["label"]: m["vs_a5"] for m in res["margins"]}
            L.append(f"| {cname} | " + " | ".join(f"{bl[m]['mean_diff']:+.3f} [{bl[m]['ci95_boot'][0]:+.3f}, {bl[m]['ci95_boot'][1]:+.3f}] p={bl[m]['wilcoxon_p_two_sided']:.2g}"
                                                for m in CLAIM_MARGINS) + " |")
        main_c = cl.get(clabel(MAIN))
        if main_c:
            bl = {m["label"]: m["vs_a5"] for m in main_c["margins"]}
            better = [m for m in CLAIM_MARGINS if bl[m]["mean_diff"] < 0 and bl[m]["wilcoxon_p_two_sided"] < 0.05]
            part2 = len(better) == 0
            out["P4_part2"] = {"significantly_better_margins": better, "holds": part2}
            L += ["", f"P4 part 2: planner significantly better than A5 at {better or 'no'} margin(s) -> holds: {part2}"]
        out["closing"] = {k: {m["label"]: m["vs_a5"] for m in v["margins"]} for k, v in cl.items()}
    if "P4_part1" in out and "P4_part2" in out:
        h1, h2 = out["P4_part1"]["holds"], out["P4_part2"]["holds"]
        out["P4"] = "supported" if (h1 and h2) else ("not supported" if not (h1 or h2) else "mixed")
        L += [f"**P4: {out['P4']}**"]
    if cf.exists():
        tl = json.loads(cf.read_text()).get("tails", {})
        out["tails"] = tl
        L += ["", "Tail metrics of the estimator conditions (closing experiment):", ""] + [f"- {k}: " + ", ".join(f"{kk} {vv:.3f}" for kk, vv in v.items()) for k, v in tl.items()]
    (ROOT / "results" / "P2" / f"est_{args.set}{vtag}.json").write_text(json.dumps(out, indent=1) + "\n")
    (ROOT / "results" / "P2" / f"est_{args.set}{vtag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
