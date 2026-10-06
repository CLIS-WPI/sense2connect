#!/usr/bin/env bash
# TVT T4 + T5 pipeline: tune (4 shards, tuning seeds) -> merge -> tracker runs -> evaluation -> T5 handover.
set -euo pipefail
cd "$(dirname "$0")/.."
L=results/TVT/T4/logs
for k in 0 1 2 3; do docker compose run --rm sionna python scripts/tvt_t4_tune.py --shard $k --nshards 4 > $L/tune_$k.log 2>&1 & done
wait
docker compose run --rm sionna python scripts/tvt_t4_tune.py --merge > $L/tune_merge.log 2>&1
bash scripts/tvt_t4_runs.sh > $L/runs_driver.log 2>&1
docker compose run --rm sionna python scripts/tvt_t4_eval.py > $L/eval.log 2>&1
docker compose run --rm sionna python scripts/tvt_t5_handover.py > results/TVT/T5/logs/handover.log 2>&1
echo pipeline done > $L/pipeline_done.log
