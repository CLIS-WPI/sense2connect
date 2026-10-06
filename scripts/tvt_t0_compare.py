"""TVT T0: headline numbers of papers 1 and 2 on DEVELOPMENT seeds under each service model.

Reads the sandboxes of scripts/tvt_t0_service.sh (results/TVT/T0/<model>/...) and, for the
regression of the power-sum model, the frozen development snapshot results/dev/M5 (paper 1) and
results/P2/closing_dev.json (paper 2). Per margin (15, 20, 25, 30 dB and the 3GPP reference):
paper 1 - A5 outage, cost-aware oracle, value of foresight (best reactive - cost-aware oracle, and
its share of the best-reactive gap), genie-planner and sensing-planner outage (tuned H), perfect-track
planner with a white UE error sigma in {0, 0.1, 0.25, 0.5, 1} m vs A5 (paper-1 positioning
requirement: the largest sigma at which the planner still has a lower mean outage than A5);
paper 2 - closing experiment, planner - A5 for perfect, white 1 m, LoS-only and map-aided PEB-level
and estimator-A positions (seed-level mean difference, exact Wilcoxon p). Also the reference margin
and SNR_req. Writes results/TVT/T0/compare.json and prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MARGINS = ("15 dB", "20 dB", "25 dB", "30 dB", "3GPP short-range reference")
P2_CONDS = ("perfect", "white 1 m", "PEB los-only", "PEB map-aided", "estimator A bw400_tdoa_s1_p2_b")


def paper1(m5: Path, m3: Path) -> dict:
    pl = json.loads((m5 / "planner.json").read_text())
    sw = json.loads((m5 / "review2" / "sweeps.json").read_text())
    met = json.loads((m3 / "metrics.json").read_text())
    out = {"snr_req_db": met["budget"]["snr_req_db"], "margin_ref_db": met["budget"]["margin_ref_db"], "margins": {}}
    sigmas = sorted(float(k.split("ue ")[1]) for k in sw["conditions"])
    for m in pl["margins"]:
        lab = m["label"]
        if lab not in MARGINS:
            continue
        hg, hs = f"{float(m['genie_planner_tuned_H']):.1f}", f"{float(m['sensing_planner_tuned_H']):.1f}"
        a5 = m["a5"]["eval"]["outage_req_s_per_min"]["mean"]
        row = {"a5": a5, "costaware_oracle": m["value_of_foresight"]["costaware"]["mean"],
               "value_of_foresight": m["value_of_foresight"]["best_reactive_minus_costaware_s_per_ue_min"],
               "value_share_of_gap": m["value_of_foresight"]["share_of_best_reactive_gap_pooled"],
               "genie_planner": m["genie_planner"][hg]["eval"]["outage_req_s_per_min"]["mean"],
               "sensing_planner": m["sensing_planner"][hs]["eval"]["outage_req_s_per_min"]["mean"],
               "oracle_inst": m["oracle_inst"]["outage_req_s_per_min"]["mean"] if isinstance(m["oracle_inst"]["outage_req_s_per_min"], dict) else m["oracle_inst"]["outage_req_s_per_min"]}
        planner_vs = {}
        for s in sigmas:
            c = sw["conditions"][f"perfect | ue {s:g}" if f"perfect | ue {s:g}" in sw["conditions"] else f"perfect | ue {s}"]
            mm = {x["label"]: x for x in c["margins"]}[lab]
            planner_vs[s] = mm.get("outage_mean", float(np.mean(list(mm["per_job"].values()))))
        row["perfect_track_planner_by_sigma"] = planner_vs
        ok = [s for s in sigmas if planner_vs[s] < a5]
        row["largest_sigma_beating_a5"] = max(ok) if ok else None
        out["margins"][lab] = row
    return out


def paper2(closing: Path) -> dict:
    d = json.loads(closing.read_text())
    out = {"a5": d["a5_outage"], "conditions": {}}
    for c in P2_CONDS:
        if c not in d["conditions"]:
            continue
        ms = {m["label"]: m for m in d["conditions"][c]["margins"]}
        out["conditions"][c] = {lab: {"mean_diff": ms[lab]["vs_a5"]["mean_diff"], "p": ms[lab]["vs_a5"]["wilcoxon_p_two_sided"]} for lab in MARGINS if lab in ms}
    return out


def regression(m5_a: Path, m5_b: Path) -> float:
    """Max |per-job outage difference| of A5, genie- and sensing-planner between two planner.json files."""
    a = {m["label"]: m for m in json.loads((m5_a / "planner.json").read_text())["margins"]}
    b = {m["label"]: m for m in json.loads((m5_b / "planner.json").read_text())["margins"]}
    worst = 0.0
    for lab in a:
        for k in a[lab]["a5"]["per_job"]:
            worst = max(worst, abs(a[lab]["a5"]["per_job"][k] - b[lab]["a5"]["per_job"][k]))
        for key in ("genie_planner", "sensing_planner"):
            for h, v in a[lab][key].items():
                if "per_job" in v:
                    for k in v["per_job"]:
                        worst = max(worst, abs(v["per_job"][k] - b[lab][key][h]["per_job"][k]))
    return worst


def main() -> None:
    res = {"definition": __doc__, "models": {}}
    t0 = ROOT / "results" / "TVT" / "T0"
    for model in ("power_sum", "best_beam"):
        sb = t0 / model
        if not (sb / "M5" / "planner.json").exists():
            continue
        res["models"][model] = {"paper1": paper1(sb / "M5", sb / "M3"), "paper2": paper2(sb / "closing_dev.json")}
    res["regression_power_sum_vs_frozen_dev_snapshot_max_abs_per_job_outage"] = regression(t0 / "power_sum" / "M5", ROOT / "results" / "dev" / "M5")
    ref2 = paper2(ROOT / "results" / "P2" / "closing_dev.json")
    res["regression_paper2_power_sum_vs_frozen_dev_closing_max_abs_mean_diff"] = max(
        abs(res["models"]["power_sum"]["paper2"]["conditions"][c][lab]["mean_diff"] - ref2["conditions"][c][lab]["mean_diff"])
        for c in ref2["conditions"] if c in res["models"]["power_sum"]["paper2"]["conditions"] for lab in ref2["conditions"][c])
    (t0 / "compare.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    for model, r in res["models"].items():
        p1 = r["paper1"]
        print(f"== {model}: SNR_req {p1['snr_req_db']:.2f} dB, reference margin {p1['margin_ref_db']:.1f} dB")
        print("margin | A5 | cost-aware oracle | value of foresight (share of gap) | genie-planner | sensing planner | largest sigma beating A5 (perfect tracks)")
        for lab, row in p1["margins"].items():
            print(f"{lab[:9]:9s} | {row['a5']:.3f} | {row['costaware_oracle']:.3f} | {row['value_of_foresight']:.3f} ({100 * row['value_share_of_gap']:.0f} %) | "
                  f"{row['genie_planner']:.3f} | {row['sensing_planner']:.3f} | {row['largest_sigma_beating_a5']}")
        print("paper 2 closing (planner - A5 [s/UE-min], p):")
        for c, ms in r["paper2"]["conditions"].items():
            print(f"  {c[:32]:32s} " + "  ".join(f"{lab[:4]}: {v['mean_diff']:+.3f} ({v['p']:.2g})" for lab, v in ms.items()))
    print("regression paper 1 (power_sum sandbox vs results/dev/M5):", res["regression_power_sum_vs_frozen_dev_snapshot_max_abs_per_job_outage"])
    print("regression paper 2 (power_sum sandbox vs results/P2/closing_dev.json):", res["regression_paper2_power_sum_vs_frozen_dev_closing_max_abs_mean_diff"])


if __name__ == "__main__":
    main()
