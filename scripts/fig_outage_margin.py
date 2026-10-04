"""Outage at the service rate vs link margin (paper/figs/fig_outage_margin.pdf).

Schemes (evaluation seeds, mean and 95 % CI over 40 jobs, tau_HO = 20 ms):
- A3: equal-effort wide grid (results/M5/planner.json, a3_wide);
- A5: best reactive scheme at every margin (planner.json, a5);
- sensing-planner: receding-horizon planner on predicted SNR, H tuned on
  the tuning seeds per margin (planner.json, sensing_planner);
- genie-planner: same planner on true future SNR, H = 0.5 s;
- cost-aware oracle: Viterbi with perfect SNR, switches at 0.1 s epochs,
  tau_HO = 20 ms (results/M5/dporacle.json, costaware_epoch);
- instantaneous oracle (planner.json, oracle_inst).
Log y axis; the 3GPP short-range reference margin is marked; the v1-radio
point is not plotted.
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


def series() -> dict:
    pl = json.loads((ROOT / "results" / "M5" / "planner.json").read_text())
    dp = json.loads((ROOT / "results" / "M5" / "dporacle.json").read_text())
    rows = [r for r in pl["margins"] if r["label"] != V1]
    dpm = {r["label"]: r for r in dp["margins"]}
    x = np.array([r["margin_db"] for r in rows])
    out = {
        "A3": [r["a3_wide"]["eval"]["outage_req_s_per_min"] for r in rows],
        "A5 (best reactive)": [r["a5"]["eval"]["outage_req_s_per_min"] for r in rows],
        "Sensing-planner": [r["sensing_planner"][f"{r['sensing_planner_tuned_H']:.1f}"]["eval"]["outage_req_s_per_min"] for r in rows],
        "Genie-planner (H = 0.5 s)": [r["genie_planner"]["0.5"]["eval"]["outage_req_s_per_min"] for r in rows],
        "Cost-aware oracle": [dpm[r["label"]]["0.020"]["costaware_epoch"] for r in rows],
        "Instantaneous oracle": [r["oracle_inst"]["outage_req_s_per_min"] for r in rows],
    }
    ref = next(r["margin_db"] for r in rows if r["label"] == "3GPP short-range reference")
    return {"x": x, "series": out, "ref_x": ref}


def main() -> None:
    plt = setup()
    data = series()
    style = {
        "A3": ("#1f77b4", "o", "-"),
        "A5 (best reactive)": ("#2ca02c", "v", "-"),
        "Sensing-planner": ("#d62728", "s", "--"),
        "Genie-planner (H = 0.5 s)": ("#ff7f0e", "^", "-."),
        "Cost-aware oracle": ("#7f7f7f", "D", ":"),
        "Instantaneous oracle": ("#000000", "x", ":"),
    }
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.5))
    x = data["x"]
    n = len(data["series"])
    for k, (name, rows) in enumerate(data["series"].items()):
        mean = np.array([r["mean"] for r in rows])
        ci = np.array([r["ci"] for r in rows])
        color, marker, ls = style[name]
        dx = (k - (n - 1) / 2) * 0.25
        lo = np.maximum(mean - ci, 1e-3)
        ax.errorbar(x + dx, mean, yerr=[mean - lo, mean + ci - mean], color=color, marker=marker, ls=ls, capsize=1.2, elinewidth=0.5,
                    lw=0.9, markersize=3.0, markerfacecolor="white" if marker not in ("x",) else color, label=name)
    ax.axvline(data["ref_x"], color="gray", lw=0.6, ls=":")
    ax.text(data["ref_x"] - 0.4, 0.97, "3GPP\nreference", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.5, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("Link margin above SNR$_\\mathrm{req}$ [dB]")
    ax.set_ylabel("Outage at 400 Mbit/s [s/UE-min]")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, round(data["ref_x"], 1)])
    ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30", f"{data['ref_x']:.1f}"])
    ax.grid(True, which="major", alpha=0.6)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=6.2, handlelength=2.2, columnspacing=0.8)
    print(f"wrote {save(fig, 'fig_outage_margin.pdf')}")


if __name__ == "__main__":
    main()
