#!/usr/bin/env bash
# TVT (human decision before the freeze): re-run the handover parts of T6 under one signaling model with the
# parameters tuned for that model in T5 (results/TVT/T5/handover_<model>.json), primary metrics wrap-masked.
#   bash scripts/tvt_signaling_runs.sh <ideal|failure_aware> <stage>
#   stage in: comm | si | variants | intersection | ovh0 | fsweep (fsweep: failure_aware only, pre-declared in configs/tvt.yaml)
# Logs: results/TVT/T6/logs/sig_<model>_*.log. Outputs: results/TVT/T5/handover_<what>_<model>.json.
set -euo pipefail
cd "$(dirname "$0")/.."
M="$1"
stage="$2"
L=results/TVT/T6/logs
mkdir -p "$L"
FIX=results/TVT/T5/handover_$M.json
SCH="A3 A5 CHO trigger_tvt trigger_learned planner_p2 planner_tvt risk_tvt riskneutral_tvt planner_tvt_perfect risk_tvt_perfect planner_true_perfect"
run() { docker compose run --rm sionna sh -c "$1" > "$L/sig_${M}_$2.log" 2>&1; }
H="python scripts/tvt_t5_handover.py --signaling $M"
case "$stage" in
  comm)
    for e in 0 10 50 100; do run "$H --fixed $FIX --e2-ms $e --schemes $SCH --out handover_e2_${e}ms_$M.json" "e2_${e}ms"; done
    for o in 0 0.5 2; do run "$H --fixed $FIX --ovh-scale $o --schemes $SCH --out handover_ovh_${o}_$M.json" "ovh_${o}"; done
    ;;
  si)
    for i in 0 10 20 30 40; do run "$H --fixed $FIX --si-inr $i --schemes trigger_tvt planner_tvt risk_tvt riskneutral_tvt A5 CHO --out handover_si_${i}_$M.json" "si_$i"; done
    ;;
  variants)
    for v in ueslow uefast; do run "$H --fixed $FIX --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v facade_$v --schemes $SCH --out handover_${v}_$M.json" "variant_$v"; done
    for v in h3 h10; do run "$H --fixed $FIX --scenario configs/tvt_sweep_$v.yaml --mounts lamppost_$v --schemes $SCH --out handover_${v}_$M.json" "variant_$v"; done
    ;;
  intersection)
    I="--scenario configs/tvt_intersection.yaml --mounts corner"
    run "$H --fixed $FIX $I --schemes $SCH --out handover_intersection_fixed_$M.json" "x_fixed"
    run "$H $I --tune-own --schemes A3 A5 CHO trigger_tvt trigger_learned planner_tvt risk_tvt riskneutral_tvt planner_tvt_perfect risk_tvt_perfect planner_true_perfect --out handover_intersection_tuned_$M.json" "x_tuned"
    ;;
  ovh0)  # human item 5: truth-UE planner with perfect tracks at sensing overhead 0 vs A5, re-tuned at overhead 0
    run "$H --ovh-scale 0 --schemes A5 planner_true_perfect planner_tvt_perfect --out handover_ovh0_tuned_$M.json" "ovh0_tuned"
    ;;
  fsweep)
    [ "$M" = failure_aware ] || { echo "fsweep needs failure_aware"; exit 1; }
    for t in 0.2 0.5; do run "$H --fixed $FIX --t310 $t --schemes $SCH --out handover_fs_t310_${t}_$M.json" "fs_t310_$t"; done
    for q in -3 3; do run "$H --fixed $FIX --q-offset $q --schemes $SCH --out handover_fs_q${q}_$M.json" "fs_q$q"; done
    run "$H --fixed $FIX --tau-re 0.23 --schemes $SCH --out handover_fs_taure_0.23_$M.json" "fs_taure_0.23"
    ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
echo "done $M $stage" >> "$L/sig_stages.log"
