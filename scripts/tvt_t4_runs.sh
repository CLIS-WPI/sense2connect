#!/usr/bin/env bash
# TVT T4: all tracker runs with the tuned parameters (results/TVT/T4/tuned.json), three parallel streams (GPU 1).
# A: development pred_real, none; B: development oracle, pred_perfect; C: tuning pred_real (input of T5 tuning),
# then development pred_real at the finite sigma_map values. Logs: results/TVT/T4/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
L=results/TVT/T4/logs
tr() { echo "python scripts/tvt_t4_track.py $1 > $L/$2.log 2>&1"; }
docker compose run --rm sionna sh -c "$(tr '--set development --cond pred_real' dev_pred_real_map0) && $(tr '--set development --cond none' dev_none_map0)" &
docker compose run --rm sionna sh -c "$(tr '--set development --cond oracle' dev_oracle_map0) && $(tr '--set development --cond pred_perfect' dev_pred_perfect_map0)" &
docker compose run --rm sionna sh -c "$(tr '--set tuning --cond pred_real' tuning_pred_real_map0) && for s in 0.1 0.25 0.5 1.0; do python scripts/tvt_t4_track.py --set development --cond pred_real --sigma-map \$s > $L/dev_pred_real_map\$s.log 2>&1; done" &
wait
echo done
