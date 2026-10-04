"""Outage at the service rate vs link margin: A3, hybrid, genie, oracle (paper/figs/fig_outage_margin.pdf).

Reads results/M3/metrics.json (A3, oracle), hybrid.json (hybrid, jointly
tuned) and genie.json (genie + A3, first-round policy, no sensing overhead,
tau_HO = 20 ms). Evaluation seeds, mean and 95 % CI over 40 jobs. The
3GPP short-range reference margin is marked; the v1-radio point is not
plotted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import COLORS, COLUMN_IN, save, setup  # noqa: E402

REF = "3GPP short-range reference"


def series() -> dict:
    m3 = ROOT / "results" / "M3"
    main = json.loads((m3 / "metrics.json").read_text())
    hyb = json.loads((m3 / "hybrid.json").read_text())
    gen = json.loads((m3 / "genie.json").read_text())
    labels = main["budget"]["labels"]
    keep = [i for i, lab in enumerate(labels) if lab != "v1 radio (high margin)"]
    tau = f"{gen['tau_ho_default_s']:.3f}"
    x = np.array([main["budget"]["points_db"][i] for i in keep])
    pick = {
        "A3": [main["evaluation"][i]["a3"]["outage_req_s_per_min"] for i in keep],
        "Hybrid (A3 + xApp)": [hyb["margins"][i]["evaluation"]["hybrid_joint"]["outage_req_s_per_min"] for i in keep],
        "Genie + A3": [gen["margins"][i]["tau_ho"][tau]["genie_no_overhead"]["outage_req_s_per_min"] for i in keep],
        "Oracle": [main["evaluation"][i]["oracle"]["outage_req_s_per_min"] for i in keep],
    }
    ref_x = main["budget"]["margin_ref_db"]
    return {"x": x, "series": pick, "ref_x": ref_x, "labels": [labels[i] for i in keep]}


def main() -> None:
    plt = setup()
    data = series()
    style = {
        "A3": (COLORS["a3"], "o", "-"),
        "Hybrid (A3 + xApp)": (COLORS["hybrid"], "s", "--"),
        "Genie + A3": (COLORS["genie"], "^", "-."),
        "Oracle": (COLORS["oracle"], "D", ":"),
    }
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.3))
    x = data["x"]
    for k, (name, rows) in enumerate(data["series"].items()):
        mean = np.array([r["mean"] for r in rows])
        ci = np.array([r["ci"] for r in rows])
        color, marker, ls = style[name]
        dx = (k - 1.5) * 0.35  # small horizontal offset so CI bars do not overlap
        lo = np.maximum(mean - ci, 1e-3)
        ax.errorbar(x + dx, mean, yerr=[mean - lo, mean + ci - mean], color=color, marker=marker, ls=ls,
                    capsize=1.5, elinewidth=0.6, label=name, markerfacecolor="white" if name != "A3" else color)
    ax.axvline(data["ref_x"], color="gray", lw=0.6, ls=":")
    ax.text(data["ref_x"] - 0.4, 0.95, "3GPP\nreference", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.5, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("Link margin above SNR$_\\mathrm{req}$ [dB]")
    ax.set_ylabel("Outage at 400 Mbit/s [s/UE-min]")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, round(data["ref_x"], 1)])
    ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30", f"{data['ref_x']:.1f}"])
    ax.grid(True, which="major", alpha=0.6)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=2.2, columnspacing=1.0)
    path = save(fig, "fig_outage_margin.pdf")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
