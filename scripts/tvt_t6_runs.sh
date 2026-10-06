#!/usr/bin/env bash
# TVT T6: sensitivity, generalisation (development seeds; parameters fixed from T4 tuned.json and T5 handover.json,
# no re-tuning except the second deployment's own-tuning run). Run on the host after T4 and T5:
#   bash scripts/tvt_t6_runs.sh <stage>     stage in: comm | calib | variants | intersection | si
# Logs: results/TVT/T6/logs/. Every stage runs in one container (GPU 1).
set -euo pipefail
cd "$(dirname "$0")/.."
stage="$1"
L=results/TVT/T6/logs
mkdir -p "$L"
FIX=results/TVT/T5/handover.json
SCH="A3 A5 trigger_tvt planner_p2 planner_tvt risk_tvt riskneutral_tvt planner_tvt_perfect risk_tvt_perfect planner_true_perfect"
run() { docker compose run --rm sionna sh -c "$1" > "$L/$2.log" 2>&1; }
case "$stage" in
  comm)  # E2 loop delay and sensing overhead (planner / trigger schemes; A3/A5 do not depend on them)
    for e in 0 10 50 100; do run "python scripts/tvt_t5_handover.py --fixed $FIX --e2-ms $e --schemes $SCH --out handover_e2_${e}ms.json" "e2_${e}ms"; done
    for o in 0 0.5 2; do run "python scripts/tvt_t5_handover.py --fixed $FIX --ovh-scale $o --schemes $SCH --out handover_ovh_${o}.json" "ovh_${o}"; done
    ;;
  calib)  # array calibration / synchronisation / bandwidth: tracker vs paper-2 estimator A
    for c in bw400_tdoa_s0_p0_b bw400_tdoa_s1_p5_b bw400_tdoa_s3_p2_b bw100_tdoa_s1_p2_b; do
      run "python scripts/tvt_t4_measure.py --set development --cfg $c && for v in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$v --cfg $c; done" "calib_$c"
    done
    ;;
  variants)  # UE speed and O-RU height (traced in the frozen tree); UE-speed variants reuse the radar detections of their base mount
    for v in ueslow uefast; do
      docker compose run --rm sionna sh -c "for m in lamppost facade; do for d in low high; do for s in 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010; do t=results/cache/\${m}_$v/\$d/seed_\$s; mkdir -p \$t; for f in detections_1024.json events.json; do [ -e \$t/\$f ] || ln -s ../../../\$m/\$d/seed_\$s/\$f \$t/\$f; done; done; done; done" > /dev/null 2>&1
      run "python scripts/tvt_t4_measure.py --set development --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v && python scripts/tvt_t6_est_variants.py --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v && for c in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$c --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v; done && python scripts/tvt_t5_handover.py --fixed $FIX --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v --schemes $SCH --out handover_$v.json" "variant_$v"
    done
    for v in h3 h10; do
      run "python scripts/tvt_t4_measure.py --set development --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v && python scripts/tvt_t6_est_variants.py --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v && for c in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$c --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v; done && python scripts/tvt_t5_handover.py --fixed $FIX --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v --schemes $SCH --out handover_$v.json" "variant_$v"
    done
    ;;
  intersection)  # second deployment: bound, tracker, handover (canyon parameters; and re-tuned on the intersection tuning seeds)
    I="--scenario configs/tvt_intersection.yaml"
    run "python scripts/tvt_t4_measure.py --set development $I --p2cfg configs/tvt_intersection_p2.yaml --mounts corner && python scripts/tvt_t4_measure.py --set tuning $I --p2cfg configs/tvt_intersection_p2.yaml --mounts corner" "x_measure"
    run "python scripts/tvt_t1_bound.py $I --p2cfg configs/tvt_intersection_p2.yaml --mounts corner --map-only --out bound_summary_intersection.json" "x_bound"
    run "for c in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$c $I --p2cfg configs/tvt_intersection_p2.yaml --faces configs/scenes/tvt_intersection/geometry.json --mounts corner; done && python scripts/tvt_t4_track.py --set tuning --cond pred_real $I --p2cfg configs/tvt_intersection_p2.yaml --faces configs/scenes/tvt_intersection/geometry.json --mounts corner" "x_track"
    run "python scripts/tvt_t5_handover.py --fixed $FIX $I --mounts corner --schemes $SCH --out handover_intersection_fixed.json" "x_handover_fixed"
    run "python scripts/tvt_t5_handover.py $I --mounts corner --tune-own --schemes A3 A5 trigger_tvt planner_tvt risk_tvt riskneutral_tvt planner_tvt_perfect risk_tvt_perfect planner_true_perfect --out handover_intersection_tuned.json" "x_handover_tuned"
    ;;
  si)  # residual radar self-interference: visibility and handover with the tracks from the recomputed detections
    run "python scripts/tvt_t6_si_eval.py" "si_visibility"
    for i in 0 10 20 30 40; do run "python scripts/tvt_t5_handover.py --fixed $FIX --si-inr $i --schemes trigger_tvt planner_tvt risk_tvt riskneutral_tvt A5 --out handover_si_$i.json" "si_handover_$i"; done
    ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
echo "done $stage" >> "$L/stages.log"
