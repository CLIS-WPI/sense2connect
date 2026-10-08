#!/usr/bin/env bash
# TVT fourth round (human decision): the intersection with one TR 38.901 panel per street arm (configs/tvt.yaml
# panels.yaw_deg_by_mount.corner = 0, 180, 90, 270 deg; results/TVT/T6/scene_design.md, committed before this run).
# Re-runs only the intersection parts of T6 (development seeds; tuning seeds for the re-tuning, same grids):
#   radar  per-panel radar detections of mount corner (four panels, untilted third-round trace reused)
#   pos    SRS measurements (four panels), T1 bound, T4 tracker (none / pred_real; pred_real on the tuning seeds)
#   ho     T5 handover at the intersection under both signaling models: canyon parameters and re-tuned
#   all    radar pos ho
# The third-round +-x intersection results are kept in results/TVT/supplement_intersection_pmx/.
# Logs: results/TVT/logs/x4_<stage>.log; stage completion appended to results/TVT/logs/x4_stages.log.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="$1"
L=results/TVT/logs
mkdir -p "$L"
run() { echo "== $(date +%H:%M) $1" >> "$L/x4_$2.log"; docker compose run --rm sionna sh -c "$1" >> "$L/x4_$2.log" 2>&1; }
I="--scenario configs/tvt_intersection.yaml"
P="--p2cfg configs/tvt_intersection_p2.yaml"
F="--faces configs/scenes/tvt_intersection/geometry.json"
case "$stage" in
  radar)
    run "python scripts/tvt_radar.py detect --sets tuning development $I --mounts corner --workers 4" radar
    ;;
  pos)
    run "python scripts/tvt_t4_measure.py --set development $I $P --mounts corner && python scripts/tvt_t4_measure.py --set tuning $I $P --mounts corner" pos
    run "python scripts/tvt_t1_bound.py $I $P --mounts corner --map-only --out bound_summary_intersection.json" pos
    run "for c in none pred_real; do python scripts/tvt_t4_track.py --set development --cond \$c $I $P $F --mounts corner; done && python scripts/tvt_t4_track.py --set tuning --cond pred_real $I $P $F --mounts corner" pos
    ;;
  ho)
    for m in failure_aware ideal; do bash scripts/tvt_signaling_runs.sh $m intersection; done
    ;;
  all)
    for s in radar pos ho; do bash "$0" $s; done
    ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
echo "done $stage $(date +%H:%M)" >> "$L/x4_stages.log"
