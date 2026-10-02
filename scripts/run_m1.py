"""Run the M1 scenario and write results/M1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_scenario  # noqa: E402
from sim.scenes.loop import run_scenario  # noqa: E402


def main() -> None:
    scenario = load_scenario(ROOT / "configs" / "m1_scenario.yaml")
    ground_truth = run_scenario(scenario, ROOT / "results" / "M1")
    n_events = len(ground_truth["events"])
    print(f"snapshots={ground_truth['n_snapshots']} events={n_events}")
    for event in ground_truth["events"]:
        print(
            f"  {event['blocker_id']} blocks {event['ue']} {event['path']} "
            f"from {event['start_s']:.3f} s to {event['end_s']:.3f} s"
        )


if __name__ == "__main__":
    main()
