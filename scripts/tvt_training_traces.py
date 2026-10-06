"""TVT T5: ray traces of the TRAINING seeds 5001-5040 (configs/seeds_tvt.yaml) for the learned
blockage-predictor baseline only (ROADMAP_TVT.md: trained on the training seeds only).

New geometry is needed because no cache exists for these seeds. Runs the unchanged paper-1 trace
driver scripts/heldout_traces.py (sensing caches, radar detections, comm traces, comm geometry;
same scene, radios, solvers and deterministic settings) with --seeds = the training seeds. The
seed guard (sim/tvt/seeds.py) is applied first. Logs: results/TVT/T5/logs/training_traces.log.
Run: python scripts/tvt_training_traces.py [--workers 4]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from sim.tvt.seeds import check, load

    seeds = check(load()["training"])
    workers = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "4"
    cmd = [sys.executable, str(ROOT / "scripts" / "heldout_traces.py"), "--workers", workers, "--seeds", *map(str, seeds)]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
