#!/usr/bin/env bash
# TVT T0: paper-1 and paper-2 pipelines end to end on DEVELOPMENT seeds under one service
# model, isolated from the real results. Run on the host:
#   bash scripts/tvt_t0_service.sh <power_sum|best_beam|mrt> [step ...]   (default: all steps in order)
# Sandbox results/TVT/T0/<model>/{M3,M5} is bind-mounted over results/M3, results/M5 and
# results/dev/M5 inside the container, and results/TVT/T0/<model>/closing_dev.json over
# results/P2/closing_dev.json, so the frozen scripts write only into the sandbox. The two
# inputs that do not depend on the service model (radar tracking metrics and tracker-error
# calibration) are copied from the development snapshot results/dev/M5.
# review_b5_ablation runs only its baseline condition (the perfect-track horizon H per margin used
# downstream) and review2_sweeps only the perfect-track UE-error points (paper-1 positioning
# requirement; paper-2 closing regression), to keep T0 short.
set -euo pipefail
cd "$(dirname "$0")/.."
model="$1"
shift
steps="${*:-run_m3 run_m3_genie run_m3_genie2 run_m5_dporacle run_m5_foresight run_m5_planner run_m5_paired review_b5_ablation review2_sweeps p2_closing}"
sb="results/TVT/T0/$model"
mkdir -p "$sb/M3" "$sb/M5/review2" "results/TVT/T0/logs"
cp results/dev/M5/tracking.json "$sb/M5/tracking.json"
cp results/dev/M5/review2/calibration.json "$sb/M5/review2/calibration.json"
touch "$sb/closing_dev.json"
cat > "$sb/steps.sh" <<INNER
set -e
run() {
  echo "== \$1 (\$(date +%T))"
  case \$1 in
    review_b5_ablation) python scripts/tvt_service.py --model $model scripts/\$1.py --conds baseline > results/TVT/T0/logs/${model}_\$1.log 2>&1 ;;
    review2_sweeps) python scripts/tvt_service.py --model $model scripts/\$1.py --only "perfect | ue 0.0" "perfect | ue 0.1" "perfect | ue 0.25" "perfect | ue 0.5" "perfect | ue 1.0" > results/TVT/T0/logs/${model}_\$1.log 2>&1 ;;
    *) python scripts/tvt_service.py --model $model scripts/\$1.py > results/TVT/T0/logs/${model}_\$1.log 2>&1 ;;
  esac
}
python scripts/tvt_service.py --model $model --precompute > results/TVT/T0/logs/${model}_precompute.log 2>&1
for s in $steps; do run \$s; done
echo done
INNER
docker compose run --rm -e S2C_EVAL_SET=dev \
  -v "$PWD/$sb/M3:/workspace/results/M3" \
  -v "$PWD/$sb/M5:/workspace/results/M5" \
  -v "$PWD/$sb/M5:/workspace/results/dev/M5" \
  -v "$PWD/$sb/closing_dev.json:/workspace/results/P2/closing_dev.json" \
  sionna sh "$sb/steps.sh"
