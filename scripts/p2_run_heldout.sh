#!/usr/bin/env bash
# Paper 2 held-out evaluation (P2-M4), run once inside the container from the
# frozen tree (tag p2-freeze), with S2C_EVAL_SET=heldout (seeds 2001-2010).
# Nothing is tuned or calibrated here: the estimator parameters come from
# results/P2/est_tuned.json (tuning seeds), the paper-1 planner from the
# paper-1 results; held-out p2 path caches are traced first.
set -euo pipefail
cd "$(dirname "$0")/.."
export S2C_EVAL_SET=heldout
test -f results/P2/est_tuned.json || { echo "results/P2/est_tuned.json (tuned on tuning seeds) missing"; exit 1; }
mkdir -p results/P2/logs_heldout
run() { echo "== $* ($(date +%T))"; python -W ignore "$@" > "results/P2/logs_heldout/$(basename "$1" .py)_$(echo "${@:2}" | tr ' -' '__').log" 2>&1; }
run scripts/p2_trace.py --workers 4 --sets evaluation
run scripts/p2_peb.py
run scripts/p2_peb_report.py --set heldout
run scripts/p2_estimate.py measure
run scripts/p2_estimate.py evaluate
run scripts/p2_closing.py
run scripts/p2_est_report.py --set heldout
run scripts/p2_figs.py --set heldout
run scripts/p2_make_paper.py --set heldout
echo "== done ($(date +%T))"
