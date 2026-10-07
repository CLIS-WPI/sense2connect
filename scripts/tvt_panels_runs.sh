#!/usr/bin/env bash
# TVT third round (configs/tvt.yaml panels): re-run of T0-T6 and the J3 diagnosis with the back-to-back TR 38.901
# panels, development seeds (tuning on the tuning seeds, learned baseline on the training seeds).
#   bash scripts/tvt_panels_runs.sh <stage>
# Stages (in dependency order; "radar" stages first, the comm/positioning stages t0 t1 t2 t4meas est can run alongside):
#   radar_trace radar_detect | t0 t1 t2 t4meas est | t3 t4 t5learned t5 | t6pos t6ho | summary
# Logs: results/TVT/logs/panels_<stage>.log; stage completion appended to results/TVT/logs/panels_stages.log.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="$1"
L=results/TVT/logs
mkdir -p "$L" results/TVT/T0/logs results/TVT/T1 results/TVT/T2 results/TVT/T3 results/TVT/T4/logs results/TVT/T5/logs results/TVT/T6/logs results/TVT/J3_diagnosis
run() { echo "== $(date +%H:%M) $1" >> "$L/panels_$2.log"; docker compose run --rm sionna sh -c "$1" >> "$L/panels_$2.log" 2>&1; }
MAIN=bw400_tdoa_s1_p2_b
CALIB="bw400_tdoa_s0_p0_b bw400_tdoa_s1_p5_b bw400_tdoa_s3_p2_b bw100_tdoa_s1_p2_b"
SCH15="A3 A5 CHO trigger_tvt trigger_learned planner_p2 planner_tvt risk_tvt riskneutral_tvt planner_tvt_perfect risk_tvt_perfect riskneutral_tvt_perfect planner_true_perfect planner_white_perfect planner_whitenw_perfect"
case "$stage" in
  radar_trace)  # untilted radar re-trace: canyon tuning/development, O-RU height variants, intersection, training
    run "python scripts/tvt_radar.py trace --sets tuning development --workers 4" radar_trace
    run "python scripts/tvt_radar.py trace --sets development --scenario configs/tvt_sweep_h3.yaml --mounts lamppost_h3 --workers 4" radar_trace
    run "python scripts/tvt_radar.py trace --sets development --scenario configs/tvt_sweep_h10.yaml --mounts lamppost_h10 --workers 4" radar_trace
    run "python scripts/tvt_radar.py trace --sets tuning development --scenario configs/tvt_intersection.yaml --mounts corner --workers 4" radar_trace
    run "python scripts/tvt_radar.py trace --sets training --workers 4" radar_trace
    ;;
  radar_detect)
    run "python scripts/tvt_radar.py detect --sets tuning development --workers 4" radar_detect
    run "python scripts/tvt_radar.py detect --sets training --workers 4" radar_detect
    run "python scripts/tvt_radar.py detect --sets development --inr 10 20 30 40 --workers 4" radar_detect
    run "python scripts/tvt_radar.py detect --sets development --scenario configs/tvt_sweep_h3.yaml --mounts lamppost_h3 --workers 4" radar_detect
    run "python scripts/tvt_radar.py detect --sets development --scenario configs/tvt_sweep_h10.yaml --mounts lamppost_h10 --workers 4" radar_detect
    run "python scripts/tvt_radar.py detect --sets tuning development --scenario configs/tvt_intersection.yaml --mounts corner --workers 4" radar_detect
    ;;
  t0)  # service models (comm only); the papers' power-sum regression does not apply to the new arrays
    bash scripts/tvt_t0_service.sh power_sum >> "$L/panels_t0.log" 2>&1
    bash scripts/tvt_t0_service.sh best_beam >> "$L/panels_t0.log" 2>&1
    run "python scripts/tvt_t0_compare.py" t0
    ;;
  t1)
    run "python scripts/tvt_t1_bound.py" t1
    run "python scripts/tvt_t1_bound.py --va-check --out bound_summary_va.json" t1
    ;;
  t2)
    run "python scripts/tvt_t2_extract.py --draws 200 --tag dyn40 --dyn-range-db 40" t2
    ;;
  t4meas)  # SRS snapshots on both panels: main configuration (tuning, development), estimator-A tuning (bw100), T6 calibration
    run "python scripts/tvt_t4_measure.py --set tuning --cfg $MAIN && python scripts/tvt_t4_measure.py --set development --cfg $MAIN" t4meas
    run "python scripts/tvt_t4_measure.py --set tuning --cfg bw100_tdoa_s1_p2_b" t4meas
    for c in $CALIB; do run "python scripts/tvt_t4_measure.py --set development --cfg $c" t4meas; done
    ;;
  est)  # paper-2 estimator A re-tuned on the panel measurements (tuning seeds) and run
    run "python scripts/tvt_est_tune.py tune --cfgs $MAIN bw100_tdoa_s1_p2_b" est
    run "python scripts/tvt_est_tune.py run --set tuning --cfg $MAIN && python scripts/tvt_est_tune.py run --set development --cfg $MAIN" est
    for c in $CALIB; do run "python scripts/tvt_est_tune.py run --set development --cfg $c" est; done
    ;;
  t3)
    run "python scripts/tvt_t3_visibility.py" t3
    ;;
  t4)
    for k in 0 1 2 3; do docker compose run --rm sionna python scripts/tvt_t4_tune.py --shard $k --nshards 4 > results/TVT/T4/logs/tune_$k.log 2>&1 & done
    wait
    run "python scripts/tvt_t4_tune.py --merge" t4
    bash scripts/tvt_t4_runs.sh >> "$L/panels_t4.log" 2>&1
    run "python scripts/tvt_t4_eval.py" t4
    ;;
  t5learned)
    run "python scripts/tvt_t5_learned.py" t5learned
    ;;
  t5)  # every scheme re-tuned on the tuning seeds under both signaling models; step dumps for the J3 diagnosis
    for m in failure_aware ideal; do
      run "python scripts/tvt_t5_handover.py --signaling $m --schemes $SCH15 --dump-steps results/TVT/J3_diagnosis/steps_$m --out handover_$m.json" t5
    done
    ;;
  t6pos)  # positioning parts of T6: calibration / bandwidth, UE speed and O-RU height variants, intersection, residual SI (visibility)
    for c in $CALIB; do run "for v in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$v --cfg $c; done" t6pos; done
    for v in ueslow uefast; do
      run "python scripts/tvt_t4_measure.py --set development --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v && python scripts/tvt_est_tune.py run --set development --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v && for c in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$c --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v; done" t6pos
    done
    for v in h3 h10; do
      run "python scripts/tvt_t4_measure.py --set development --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v && python scripts/tvt_est_tune.py run --set development --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v && for c in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$c --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v; done" t6pos
    done
    I="--scenario configs/tvt_intersection.yaml"
    run "python scripts/tvt_t4_measure.py --set development $I --p2cfg configs/tvt_intersection_p2.yaml --mounts corner && python scripts/tvt_t4_measure.py --set tuning $I --p2cfg configs/tvt_intersection_p2.yaml --mounts corner" t6pos
    run "python scripts/tvt_t1_bound.py $I --p2cfg configs/tvt_intersection_p2.yaml --mounts corner --map-only --out bound_summary_intersection.json" t6pos
    run "for c in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$c $I --p2cfg configs/tvt_intersection_p2.yaml --faces configs/scenes/tvt_intersection/geometry.json --mounts corner; done && python scripts/tvt_t4_track.py --set tuning --cond pred_real $I --p2cfg configs/tvt_intersection_p2.yaml --faces configs/scenes/tvt_intersection/geometry.json --mounts corner" t6pos
    run "python scripts/tvt_t6_si_eval.py" t6pos
    ;;
  t6ho)  # handover parts of T6 under both signaling models (scripts/tvt_signaling_runs.sh), failure-model sweep
    for m in failure_aware ideal; do
      for st in comm ovh0 si variants intersection; do bash scripts/tvt_signaling_runs.sh $m $st; done
    done
    bash scripts/tvt_signaling_runs.sh failure_aware fsweep
    ;;
  summary)
    run "python scripts/tvt_t6_summary.py > results/TVT/T6/summary_tables.md; python scripts/tvt_t6_complexity.py" summary
    run "python scripts/tvt_j3_diagnosis.py --model failure_aware > results/TVT/J3_diagnosis/tables_failure_aware.md && python scripts/tvt_j3_diagnosis.py --model ideal > results/TVT/J3_diagnosis/tables_ideal.md" summary
    run "python scripts/tvt_signaling_summary.py > results/TVT/T6/signaling_tables.md && python scripts/tvt_panels_compare.py > results/TVT/panels_vs_iso.md" summary
    ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
echo "done $stage $(date +%H:%M)" >> "$L/panels_stages.log"
