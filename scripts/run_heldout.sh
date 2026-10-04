#!/usr/bin/env bash
# Held-out evaluation (second external review, Block 3), run inside the container
# from the frozen tree (tag v1.1-freeze): traces the held-out seeds 2001-2010 and
# reruns the evaluation pipeline unchanged with S2C_EVAL_SET=heldout. Tuning
# stays on the tuning seeds 101-105 (same data, same code -> same parameters;
# checked afterwards against the development snapshot); the tracker-error
# calibration and the fixed horizons are read from the development results and
# never recomputed. Logs: results/heldout_logs/.
# Before running: copy results/M3 and results/M5 to results/dev (development snapshot).
set -euo pipefail
cd "$(dirname "$0")/.."
test -d results/dev/M5 || { echo "results/dev (development snapshot) missing"; exit 1; }
export S2C_EVAL_SET=heldout
mkdir -p results/heldout_logs
run() { echo "== $* ($(date +%T))"; python "$@" > "results/heldout_logs/$(basename "$1" .py).log" 2>&1; }
run scripts/heldout_traces.py --workers 4
run scripts/fig_blockage_table.py
run scripts/run_m5_tracking.py
run scripts/run_m3.py
run scripts/run_m3_genie.py
run scripts/run_m3_hybrid.py
run scripts/run_m5_dporacle.py
run scripts/run_m5_foresight.py
run scripts/run_m5_planner.py
run scripts/run_m5_paired.py
run scripts/review_grid_oracle.py
run scripts/review_a34.py
run scripts/review_b6_robust.py
run scripts/review_b6_diag.py
run scripts/review_c8_density.py
run scripts/review2_sweeps.py
run scripts/review2_uaplanner.py
run scripts/make_paper.py
echo "== done ($(date +%T))"
