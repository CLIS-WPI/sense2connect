#!/usr/bin/env bash
# Paper 2 held-out evaluation 2 (after the estimator fix), run once inside the container
# from the frozen tree (tag p2-freeze2) on seeds 3001-3010 (configs/p2.yaml heldout2,
# never used before). Nothing is tuned or calibrated: estimator parameters come from
# results/P2/est_tuned*.json (tuning seeds). The 2001-2010 results stay as the pre-fix record.
# Usage: scripts/p2_run_heldout2.sh "<variants>"  (e.g. "v1 A" or "v1 A AB")
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANTS="${1:-v1 A}"
PRIMARY="$(echo $VARIANTS | awk '{print $NF}')"
SEEDS="3001 3002 3003 3004 3005 3006 3007 3008 3009 3010"
mkdir -p results/P2/logs_heldout2
run() { echo "== $* ($(date +%T))"; python -W ignore "$@" > "results/P2/logs_heldout2/$(basename "$1" .py)_$(echo "${@:2}" | tr ' -' '__' | cut -c1-60).log" 2>&1; }
# 1. paper-1 traces of the new seeds (frozen paper-1 driver, run with the paper-1 default seed set; new cache dirs only)
( unset S2C_EVAL_SET; run scripts/heldout_traces.py --workers 4 --seeds $SEEDS )
export S2C_EVAL_SET=heldout2
run scripts/p2_trace.py --workers 4 --sets evaluation
run scripts/p2_peb.py
run scripts/p2_peb_report.py --set heldout2
run scripts/p2_estimate.py measure
for v in $VARIANTS; do run scripts/p2_estimate.py evaluate --variant $v; done
run scripts/p2_closing.py
for v in $VARIANTS; do run scripts/p2_est_report.py --set heldout2 --variant $v; run scripts/p2_diag_tail.py --set heldout2 --variant $v; done
run scripts/p2_figs.py --set heldout2 --variant $PRIMARY
run scripts/p2_make_paper.py --set heldout2 --variant $PRIMARY
echo "== done ($(date +%T))"
