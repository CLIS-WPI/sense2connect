"""Headroom decomposition of A3 outage vs link margin (paper/figs/fig_headroom.pdf).

Stacked shares of A3's outage at the service rate (tau_HO = 20 ms, tuned
A3, evaluation seeds, means over 40 jobs), from results/M3/genie.json
(`a3_decomp`):
- both cells unusable (c; equals the oracle outage, unrecoverable),
- handover interruption (b),
- foresight bound: wrong cell inside 10 dB events while the other cell
  was usable (a, in events; equals genie2.json foresight "all"),
- wrong cell outside events (a, outside; path-loss-driven lag).
Line: A3 outage with interruption-free handover (tau_HO = 0, retuned) as a
share of A3 outage at tau_HO = 20 ms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import COLUMN_IN, save, setup  # noqa: E402


def data() -> dict:
    m3 = ROOT / "results" / "M3"
    gen = json.loads((m3 / "genie.json").read_text())
    gen2 = json.loads((m3 / "genie2.json").read_text())
    tau = f"{gen['tau_ho_default_s']:.3f}"
    rows = [r for r in gen["margins"] if r["label"] != "v1 radio (high margin)"]
    labels = [r["label"] for r in rows]
    x = np.array([r["margin_db"] for r in rows])
    d = [r["tau_ho"][tau]["a3_decomp"] for r in rows]
    comp = {
        "Both cells unusable": np.array([v["c_both_unusable"]["mean"] for v in d]),
        "Handover interruption": np.array([v["b_interruption"]["mean"] for v in d]),
        "Foresight bound (wrong cell in events)": np.array([v["a_in_events"]["mean"] for v in d]),
        "Wrong cell outside events": np.array([v["a_outside_events"]["mean"] for v in d]),
    }
    total = sum(comp.values())
    a3_20 = np.array([r["tau_ho"][tau]["a3"]["outage_req_s_per_min"]["mean"] for r in rows])
    a3_0 = np.array([r["tau_ho"]["0.000"]["a3"]["outage_req_s_per_min"]["mean"] for r in rows])
    fs = {r["label"]: r["foresight"]["all"]["s_per_ue_min"]["mean"] for r in gen2["margins"]}
    check = max(abs(fs[lab] - v) for lab, v in zip(labels, comp["Foresight bound (wrong cell in events)"]))
    return {"x": x, "labels": labels, "comp": comp, "total": total, "a3_20": a3_20, "a3_0": a3_0, "foresight_check": check}


def main() -> None:
    plt = setup()
    dd = data()
    if dd["foresight_check"] > 1e-12:
        raise SystemExit(f"foresight bound differs between genie.json and genie2.json by {dd['foresight_check']}")
    if np.max(np.abs(dd["total"] - dd["a3_20"])) > 1e-9:
        raise SystemExit("decomposition does not sum to the A3 outage")
    colors = ["#bdbdbd", "#1f77b4", "#ff7f0e", "#9ecae1"]
    hatches = ["", "", "////", ""]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.3))
    pos = np.arange(len(dd["x"]))
    bottom = np.zeros(len(pos))
    for (name, val), color, hatch in zip(dd["comp"].items(), colors, hatches):
        share = val / dd["total"]
        ax.bar(pos, share, bottom=bottom, width=0.7, color=color, edgecolor="black", linewidth=0.3, hatch=hatch, label=name)
        bottom += share
    ax.plot(pos, dd["a3_0"] / dd["a3_20"], color="black", marker="o", markerfacecolor="white", lw=1.0,
            label=r"A3 outage with $\tau_\mathrm{HO}=0$ (rel. to 20 ms)")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{v:.0f}" if lab != "3GPP short-range reference" else f"{v:.1f}\n(3GPP)" for v, lab in zip(dd["x"], dd["labels"])])
    ax.set_xlabel("Link margin above SNR$_\\mathrm{req}$ [dB]")
    ax.set_ylabel("Share of A3 outage")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=6.5, handlelength=1.6, columnspacing=0.8)
    path = save(fig, "fig_headroom.pdf")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
