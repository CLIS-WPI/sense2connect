"""Value of foresight per margin (paper/figs/fig_value.pdf); replaces fig_headroom.pdf.

For each margin the best-reactive-to-instantaneous-oracle gap (best reactive
= A5 at every margin, results/M5/planner.json) is normalised to 1 and split
into the part closed by perfect foresight -- the cost-aware oracle
(results/M5/dporacle.json: 0.1 s epochs, tau_HO = 20 ms) -- stacked by the
class of the dominant LoS blocker of A5's cell (pedestrian, bus/truck, car;
net steps, pooled over evaluation jobs), and the part that is not closable.
A net-negative car share (at most a few percent) is not drawn; the
not-closable part is drawn as 1 minus the drawn closed parts. Markers: share
of the gap closed by the genie-planner (H = 0.5 s) and by the
sensing-planner (H tuned), (A5 - planner) / (A5 - instantaneous oracle);
values below the axis are drawn at the bottom edge with their value.
The absolute gap [s/UE-min] is printed above each bar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import COLUMN_IN, save, setup  # noqa: E402

V1 = "v1 radio (high margin)"
YMIN = -0.3


def data() -> dict:
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    rows = [r for r in pl["margins"] if r["label"] != V1]
    out = {"labels": [r["label"] for r in rows], "x": [r["margin_db"] for r in rows], "ped": [], "bus": [], "car": [], "closed": [], "gap": [], "genie": [], "sensing": []}
    for r in rows:
        if r["best_reactive"] != "a5":
            raise SystemExit(f"{r['label']}: best reactive is not A5; update the figure labels")
        v = r["value_of_foresight"]
        bc = v["by_class_share_of_gap_pooled"]
        br = r["a5"]["eval"]["outage_req_s_per_min"]["mean"]
        inst = r["oracle_inst"]["outage_req_s_per_min"]["mean"]
        gap = br - inst
        out["ped"].append(bc["pedestrian"])
        out["bus"].append(bc["bus/truck"])
        out["car"].append(max(bc["car"], 0.0))
        out["closed"].append(v["share_of_best_reactive_gap_pooled"])
        out["gap"].append(gap)
        g = r["genie_planner"]["0.5"]["eval"]["outage_req_s_per_min"]["mean"]
        s = r["sensing_planner"][f"{r['sensing_planner_tuned_H']:.1f}"]["eval"]["outage_req_s_per_min"]["mean"]
        out["genie"].append((br - g) / gap if gap > 0 else np.nan)
        out["sensing"].append((br - s) / gap if gap > 0 else np.nan)
    return out


def main() -> None:
    plt = setup()
    d = data()
    pos = np.arange(len(d["x"]))
    ped, bus, car = (np.array(d[k]) for k in ("ped", "bus", "car"))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.5))
    ax.bar(pos, ped, width=0.65, color="#d62728", edgecolor="black", lw=0.3, label="Closable: pedestrian")
    ax.bar(pos, bus, width=0.65, bottom=ped, color="#1f77b4", edgecolor="black", lw=0.3, label="Closable: bus/truck")
    ax.bar(pos, car, width=0.65, bottom=ped + bus, color="#9467bd", edgecolor="black", lw=0.3, label="Closable: car")
    rest = 1.0 - (ped + bus + car)
    ax.bar(pos, rest, width=0.65, bottom=ped + bus + car, color="#e0e0e0", edgecolor="black", lw=0.3, label="Not closable")
    for i, g in enumerate(d["gap"]):
        ax.text(pos[i], 1.02, f"{g:.2f}", ha="center", va="bottom", fontsize=5.8)
    for key, color, marker, label in (("genie", "#ff7f0e", "^", "Genie-planner (H = 0.5 s)"), ("sensing", "#000000", "s", "Sensing-planner")):
        y = np.array(d[key])
        clipped = y < YMIN
        if clipped.any():
            label = label + " (below axis: value)"
        ax.plot(pos[~clipped], y[~clipped], ls="none", marker=marker, color=color, markerfacecolor="white", markersize=4, label=label)
        if clipped.any():
            yb = YMIN + 0.06
            ax.plot(pos[clipped], np.full(clipped.sum(), yb), ls="none", marker=marker, color=color, markerfacecolor="white", markersize=4)
            for i in np.flatnonzero(clipped):
                ax.annotate(f"{y[i]:.1f}", (pos[i], yb), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{v:.0f}" if lab != "3GPP short-range reference" else f"{v:.1f}\n(3GPP)" for v, lab in zip(d["x"], d["labels"])])
    ax.set_xlabel("Link margin above SNR$_\\mathrm{req}$ [dB]")
    ax.set_ylabel("Share of A5-to-oracle gap")
    ax.set_ylim(YMIN, 1.12)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=6.2, handlelength=1.6, columnspacing=0.8)
    print(f"wrote {save(fig, 'fig_value.pdf')}")


if __name__ == "__main__":
    main()
