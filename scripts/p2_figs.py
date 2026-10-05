"""Paper 2 figures (P2-M5): single column (3.5 in), vector PDF, all text >= 8 pt, into paper2/figs/.

fig_p2_bw.pdf       median PEB (LoS-only, map-aided) and median estimator error vs bandwidth, LoS to both vs blocked
fig_p2_sidewalk.pdf median PEB and estimator error along the sidewalk (UE x, 5 m bins)
fig_p2_cdf.pdf      CDF of PEB (LoS-only, map-aided) and estimator error, main configuration (the shaded paper-1
                    break-even band was removed after the external review; figure-only change)
fig_p2_closing.pdf  closing experiment: outage vs link margin, A5 and the paper-1 planner fed with positions of increasing quality
                    (synthetic bound-level curves: full single-epoch covariance e_t ~ N(0, J_p(t)^-1), scripts/p2_diag_boundlevel.py;
                    switched from the isotropic PEB curves for text v5, figure-only change)
Main configuration: 400 MHz, TDoA, sigma_sync 1 ns (per run), sigma_phi 2 deg, blocked LoS biased/diffracted.
Seed-level points (per-seed medians, mean over seeds) with 95 % bootstrap CIs over seeds where shown.
Run: python scripts/p2_figs.py --set dev|heldout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FIG = ROOT / "paper2" / "figs"
COL = 3.5
WARMUP = 20


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                         "lines.linewidth": 1.1, "lines.markersize": 3.5, "grid.linewidth": 0.3, "pdf.fonttype": 42, "savefig.bbox": "standard"})
    return plt


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    return path


def boot(v, n=10000, seed=20261005):
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def load_runs(tag: str):
    from p2_peb_report import load

    return load(tag)


VARIANT = "v1"


def est_runs(tag: str, cfg: str):
    out = {}
    d_ = "est" if VARIANT == "v1" else f"est_{VARIANT}"
    for f in sorted((ROOT / "results" / "P2" / d_ / tag / cfg).glob("*_track.npz")):
        mount, density, seed = f.stem.replace("_track", "").split("_")
        tr = np.load(f)
        ms = np.load(ROOT / "results" / "P2" / "est" / tag / cfg / f.name.replace("_track", ""))
        err = np.linalg.norm(tr["xy_ekf"] - ms["ue"][..., :2], axis=-1)
        out[(int(seed), mount, density)] = {"err": np.nan_to_num(err, nan=1e3), "nb": ms["blocked"].sum(-1), "x": ms["ue"][..., 0]}
    return out


def seed_medians(runs: dict, getter, mask):
    seeds = sorted({k[0] for k in runs})
    vals = []
    for s in seeds:
        v = np.concatenate([getter(r)[WARMUP:][mask(r)[WARMUP:]] for k, r in runs.items() if k[0] == s])
        vals.append(np.median(v) if v.size else np.nan)
    return np.array(vals)


def main() -> None:
    from p2_peb_report import idx

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--variant", default="v1", choices=["v1", "A", "AB"])
    args = ap.parse_args()
    global VARIANT
    VARIANT = args.variant
    plt = setup()
    runs = load_runs(args.set)
    LOS = lambda r: r["nb"] == 0  # noqa: E731
    BLK = lambda r: r["nb"] >= 1  # noqa: E731
    bws = ["100", "200", "400"]
    # ---------------------------------------------------------------- fig 1: vs bandwidth
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    x = np.arange(3)
    series = [("LoS-only bound", "#1f77b4", "o", lambda bw: (lambda r: r["peb"][(slice(None), slice(None)) + idx(info="los", bw=bw)])),
              ("Map-aided bound", "#2ca02c", "s", lambda bw: (lambda r: r["peb"][(slice(None), slice(None)) + idx(info="map", bw=bw)]))]
    for name, color, mk, get in series:
        for state, fn, ls in (("LoS to both", LOS, "-"), ("blocked", BLK, "--")):
            med = [seed_medians(runs, get(bw), fn) for bw in bws]
            m = np.array([np.mean(v) for v in med])
            ci = np.array([boot(v) for v in med])
            ax.errorbar(x, m, yerr=[m - ci[:, 0], ci[:, 1] - m], color=color, marker=mk, ls=ls, capsize=2, label=f"{name}, {state}")
    for state, fn, ls in (("LoS to both", LOS, "-"), ("blocked", BLK, "--")):
        med = []
        for bw in bws:
            er = est_runs(args.set, f"bw{bw}_tdoa_s1_p2_b")
            med.append(seed_medians(er, lambda r: r["err"], fn) if er else np.full(10, np.nan))
        m = np.array([np.mean(v) for v in med])
        ci = np.array([boot(v) for v in med])
        ax.errorbar(x, m, yerr=[m - ci[:, 0], ci[:, 1] - m], color="#d62728", marker="^", ls=ls, capsize=2, label=f"Estimator, {state}")
    ax.axhline(0.1, color="gray", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} MHz" for b in bws])
    ax.set_ylabel("Median position error [m]")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.12)
    fig.set_size_inches(COL, 2.6)
    # legend outside would widen the figure; place it below instead
    ax.get_legend().remove()
    fig.set_size_inches(COL, 3.2)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.33)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8, handlelength=1.8, columnspacing=0.8)
    print("wrote", save(fig, "fig_p2_bw.pdf"))
    # ---------------------------------------------------------------- fig 2: along the sidewalk
    bins = np.arange(-40.0, 40.01, 5.0)
    centers = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(COL, 2.6))

    def profile(rs, getter):
        out = []
        for k in range(len(bins) - 1):
            per_seed = []
            for s in sorted({kk[0] for kk in rs}):
                v = np.concatenate([getter(r)[WARMUP:][(np.clip(np.digitize(r["x"][WARMUP:], bins) - 1, 0, len(bins) - 2) == k)]
                                    for kk, r in rs.items() if kk[0] == s])
                per_seed.append(np.median(v) if v.size else np.nan)
            out.append(np.nanmean(per_seed))
        return np.array(out)

    ax.plot(centers, profile(runs, lambda r: r["peb"][(slice(None), slice(None)) + idx(info="los")]), color="#1f77b4", marker="o", label="LoS-only bound")
    ax.plot(centers, profile(runs, lambda r: r["peb"][(slice(None), slice(None)) + idx(info="map")]), color="#2ca02c", marker="s", label="Map-aided bound")
    er = est_runs(args.set, "bw400_tdoa_s1_p2_b")
    if er:
        ax.plot(centers, profile(er, lambda r: r["err"]), color="#d62728", marker="^", label="Estimator (EKF)")
    ax.axhline(0.1, color="gray", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("UE position along the street, $x$ [m]")
    ax.set_ylabel("Median error [m]")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.17)
    print("wrote", save(fig, "fig_p2_sidewalk.pdf"))
    # ---------------------------------------------------------------- fig 3: CDF
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    pool = lambda rs, g: np.concatenate([g(r)[WARMUP:].ravel() for r in rs.values()])  # noqa: E731
    for name, color, v in (("LoS-only bound", "#1f77b4", pool(runs, lambda r: r["peb"][(slice(None), slice(None)) + idx(info="los")])),
                           ("Map-aided bound", "#2ca02c", pool(runs, lambda r: r["peb"][(slice(None), slice(None)) + idx(info="map")])),
                           ("Estimator (EKF)", "#d62728", pool(er, lambda r: r["err"]) if er else np.array([np.nan]))):
        v = np.sort(v[np.isfinite(v)])
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 30)
    ax.set_xlabel("Position error [m]")
    ax.set_ylabel("CDF")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.18)
    print("wrote", save(fig, "fig_p2_cdf.pdf"))
    # ---------------------------------------------------------------- fig 4: closing experiment
    cf = ROOT / "results" / "P2" / f"closing_{args.set}.json"
    if cf.exists():
        cl = dict(json.loads(cf.read_text())["conditions"])
        clj = json.loads(cf.read_text())
        # synthetic bound-level curves: full single-epoch covariance, e_t ~ N(0, J_p(t)^-1) (scripts/p2_diag_boundlevel.py; text v5)
        bfj = ROOT / "results" / "P2" / f"diag_boundlevel_{args.set}.json"
        if bfj.exists():
            cl.update(json.loads(bfj.read_text())["conditions"])
        pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())["margins"]
        a5o = clj.get("a5_outage")
        labs = [m["label"] for m in pl if m["label"] != "v1 radio (high margin)"]
        xm = np.array([m["margin_db"] for m in pl if m["label"] != "v1 radio (high margin)"])
        fig, ax = plt.subplots(figsize=(COL, 3.0))
        a5 = np.array([a5o[m["label"]] if a5o else m["a5"]["eval"]["outage_req_s_per_min"]["mean"] for m in pl if m["label"] != "v1 radio (high margin)"])
        est_key = "estimator bw400_tdoa_s1_p2_b" if VARIANT == "v1" else f"estimator {VARIANT} bw400_tdoa_s1_p2_b"
        ax.plot(xm, a5, color="black", marker="v", label="A5 (3GPP)")
        style = {"perfect": ("#2ca02c", "o", "Exact UE position"),
                 "bound-level map-aided, full covariance (synthetic)": ("#17becf", "s", "Map-aided bound (synthetic)"),
                 "bound-level LoS-only, full covariance (synthetic)": ("#1f77b4", "s", "LoS-only bound (synthetic)"),
                 est_key: ("#d62728", "^", "Estimator (400 MHz)"), "white 1 m": ("#7f7f7f", "x", "White 1 m error")}
        for k, (color, mk, lab) in style.items():
            if k not in cl:
                continue
            ys = {m["label"]: m["outage_mean"] for m in cl[k]["margins"]}
            ax.plot(xm, [ys[lb] for lb in labs], color=color, marker=mk, ls="--" if "white" in k else "-", label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("Link margin above SNR$_\\mathrm{req}$ [dB]")
        ax.set_ylabel("Outage at 400 Mbit/s [s/UE-min]")
        ax.grid(True, which="both", alpha=0.4)
        fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.40)
        h, lab = ax.get_legend_handles_labels()
        # planner curves are labelled by the UE position fed to the planner (the caption names the planner)
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8, handlelength=1.8, columnspacing=0.8)
        print("wrote", save(fig, "fig_p2_closing.pdf"))


if __name__ == "__main__":
    main()
