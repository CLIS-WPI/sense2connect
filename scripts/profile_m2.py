"""Profile one 60 s sensing run and write the timing buckets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.traffic import prepare_scenario
from sim.sensing.trace import trace_job
from sim.scenes.config import load_yaml

def main() -> None:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "M2" / "profile_before.json"
    scenario = prepare_scenario(
        load_yaml(ROOT / "configs" / "m2_scenario.yaml"),
        seed=101,
        mount="lamppost",
        density="low",
        duration_s=0.1 * frames,
        dt_s=0.1,
    )
    summary = trace_job(
        scenario,
        mount="lamppost",
        density="low",
        cache_root=ROOT / "results" / "cache",
        max_frames=frames,
        process=True,
    )
    summary["seconds_per_snapshot"] = sum(summary["buckets_s"].values()) / frames
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
