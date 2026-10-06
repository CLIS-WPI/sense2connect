"""TVT T6: regression of the residual-SI detections - INR 0 (recomputed in the frozen tree from re-traced sensing
caches) vs the paper-1 detection caches (results/cache/<mount>/<density>/seed_<s>/detections_1024.json), CFAR setting
of the trackers (blind, train 4, pfa 1e-3). Reports the share of frames with identical detection lists and the
max |difference| of the detected range, position and Doppler. Writes results/TVT/T6/si_regression.json.
Run: python3 scripts/tvt_t6_si_regression.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    key = "blind:4:0.001"
    rows = []
    for d in sorted((ROOT / "results" / "TVT" / "T6" / "si" / "inr0").glob("*/*/seed_*/detections_1024.json")):
        mount, dens, sd = d.parts[-4], d.parts[-3], d.parts[-2]
        ref = ROOT / "results" / "cache" / mount / dens / sd / "detections_1024.json"
        a = json.loads(d.read_text())["frames"]
        b = json.loads(ref.read_text())["frames"]
        same, worst = 0, 0.0
        for fa, fb in zip(a, b):
            da, db = fa["detections"][key], fb["detections"][key]
            if len(da) == len(db):
                ok = True
                for x, y in zip(da, db):
                    for k in ("range_m", "x_m", "y_m", "doppler_hz"):
                        if k in x and k in y:
                            worst = max(worst, abs(float(x[k]) - float(y[k])))
                            ok &= abs(float(x[k]) - float(y[k])) < 1e-6
                same += int(ok)
        rows.append({"job": f"{mount}/{dens}/{sd}", "frames": len(a), "identical_frames": same, "max_abs_diff": worst})
    out = {"definition": __doc__, "jobs": rows, "share_identical_frames": float(np.sum([r["identical_frames"] for r in rows]) / max(np.sum([r["frames"] for r in rows]), 1))}
    (ROOT / "results" / "TVT" / "T6" / "si_regression.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "jobs"}), len(rows), "jobs")


if __name__ == "__main__":
    main()
