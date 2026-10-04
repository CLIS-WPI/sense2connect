"""Headroom decomposition of A3 outage vs link margin (paper/figs/fig_headroom.pdf).

Stacked shares of A3's outage at the service rate (tuned A3, tau_HO = 20 ms,
evaluation seeds, pooled over jobs) from the causal decomposition in
results/M5/foresight.json (scripts/run_m5_foresight.py), four bands:
- both cells unusable (unrecoverable by any cell choice),
- handover interruption,
- wrong cell, blockage-caused (A3's cell unusable, other cell usable, A3's
  cell usable without blockers) = causal foresight bound,
- wrong cell, distance-caused (A3's cell unusable even unblocked, other
  cell usable).
Line: A3 outage with interruption-free handover (tau_HO = 0, retuned,
results/M3/genie.json) relative to tau_HO = 20 ms.
Checks: interruption and both-unusable equal the M3 decomposition
(genie.json a3_decomp), and the bands sum to A3's outage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import COLUMN_IN, save, setup  # noqa: E402

BANDS = (
    ("c_both_unusable", "Both cells unusable", "#bdbdbd", ""),
    ("b_interruption", "Handover interruption", "#1f77b4", ""),
    ("new_blockage_caused", "Wrong cell, blockage-caused (foresight bound)", "#ff7f0e", "////"),
    ("new_distance_caused", "Wrong cell, distance-caused", "#9ecae1", ""),
)


def data() -> dict:
    m3 = ROOT / "results" / "M3"
    fs = json.loads((ROOT / "results" / "M5" / "foresight.json").read_text())
    gen = json.loads((m3 / "genie.json").read_text())
    tau = f"{gen['tau_ho_default_s']:.3f}"
    rows = [r for r in fs["margins"] if r["label"] != "v1 radio (high margin)"]
    g = {r["label"]: r for r in gen["margins"]}
    for r in rows:
        old = g[r["label"]]["tau_ho"][tau]["a3_decomp"]
        for key in ("c_both_unusable", "b_interruption"):
            if abs(old[key]["mean"] - r[key]["s_per_ue_min"]["mean"]) > 1e-9:
                raise SystemExit(f"{r['label']} {key}: causal decomposition differs from genie.json a3_decomp")
    shares = {key: np.array([r[key]["share_of_a3_outage_pooled"] for r in rows]) for key, *_ in BANDS}
    total = sum(shares.values())
    if np.max(np.abs(total - 1.0)) > 1e-9:
        raise SystemExit("bands do not sum to the A3 outage")
    a3_20 = np.array([g[r["label"]]["tau_ho"][tau]["a3"]["outage_req_s_per_min"]["mean"] for r in rows])
    a3_0 = np.array([g[r["label"]]["tau_ho"]["0.000"]["a3"]["outage_req_s_per_min"]["mean"] for r in rows])
    return {"x": np.array([r["margin_db"] for r in rows]), "labels": [r["label"] for r in rows], "shares": shares, "ratio_tau0": a3_0 / a3_20}


def main() -> None:
    plt = setup()
    dd = data()
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.3))
    pos = np.arange(len(dd["x"]))
    bottom = np.zeros(len(pos))
    for key, label, color, hatch in BANDS:
        share = dd["shares"][key]
        ax.bar(pos, share, bottom=bottom, width=0.7, color=color, edgecolor="black", linewidth=0.3, hatch=hatch, label=label)
        bottom += share
    ax.plot(pos, dd["ratio_tau0"], color="black", marker="o", markerfacecolor="white", lw=1.0,
            label=r"A3 outage with $\tau_\mathrm{HO}=0$ (rel. to 20 ms)")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{v:.0f}" if lab != "3GPP short-range reference" else f"{v:.1f}\n(3GPP)" for v, lab in zip(dd["x"], dd["labels"])])
    ax.set_xlabel("Link margin above SNR$_\\mathrm{req}$ [dB]")
    ax.set_ylabel("Share of A3 outage")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False, fontsize=6.5, handlelength=1.6)
    path = save(fig, "fig_headroom.pdf")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
