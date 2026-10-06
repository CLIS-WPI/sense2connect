"""TVT T6: run time per component (Table "complexity") from the stored run records, development seeds.

GPU: one H100 NVL (GPU 1), shared with other jobs of this study while the timings were taken (upper
bounds). Sources: multipath extraction (results/TVT/T2/extract_dyn40.json, per O-RU snapshot, batched);
visibility prediction and tracker update (results/TVT/T4/track/development/<cond>_map0/meta.json, per
tracker epoch = one UE, both O-RUs, 0.1 s); position error bound (results/TVT/T1/bound_summary.json wall
time / (jobs x epochs x UEs), all sweep variants of T1 together); handover schemes (results/TVT/T5/
handover.json per-scheme wall time, tuning + evaluation, per lane-report). Writes results/TVT/T6/complexity.json
and prints a markdown table. Run: python3 scripts/tvt_t6_complexity.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    ex = json.loads((ROOT / "results" / "TVT" / "T2" / "extract_dyn40.json").read_text())
    rows.append(("multipath extraction (T2), per O-RU snapshot, 3168 x 64", ex["runtime_s_per_snapshot"]["median"], "GPU, batched"))
    for cond in ("none", "pred_real", "pred_perfect", "oracle"):
        m = ROOT / "results" / "TVT" / "T4" / "track" / "development" / f"{cond}_map0" / "meta.json"
        if m.exists():
            t = json.loads(m.read_text())["timing"]
            rows.append((f"visibility prediction ({cond}), per tracker epoch", t["visibility_s_per_tracker_epoch"], "GPU, K = 32 samples, batched over trackers"))
            rows.append((f"tracker update ({cond}), per tracker epoch", t["update_s_per_tracker_epoch"], "CPU, NumPy, sequential"))
    b = json.loads((ROOT / "results" / "TVT" / "T1" / "bound_summary.json").read_text())
    n_ep = b["n_jobs"] * 600 * 2
    n_var = len(b["configs"])
    rows.append(("position error bound, per epoch and variant", b["wall_s"] / n_ep / max(n_var, 1), f"GPU, {n_var} variants per epoch"))
    hf = ROOT / "results" / "TVT" / "T5" / "handover.json"
    if hf.exists():
        h = json.loads(hf.read_text())
        lab = next(iter(h["schemes"]))
        for sch, v in h["schemes"][lab].items():
            if " vs " in sch or "wall_s" not in v:
                continue
            rows.append((f"handover scheme {sch} (tuning + evaluation, one margin), total", v["wall_s"], "s per margin"))
    out = {"definition": __doc__, "rows": [{"component": r[0], "seconds": r[1], "note": r[2]} for r in rows]}
    (ROOT / "results" / "TVT" / "T6" / "complexity.json").write_text(json.dumps(out, indent=1) + "\n")
    print("| component | time | note |\n|---|---|---|")
    for r in rows:
        t = r[1]
        s = f"{t * 1e3:.2f} ms" if t < 1 else f"{t:.0f} s"
        print(f"| {r[0]} | {s} | {r[2]} |")


if __name__ == "__main__":
    main()
