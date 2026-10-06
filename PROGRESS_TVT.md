# PROGRESS_TVT - journal extension (ROADMAP_TVT.md, GOAL_TVT.md)

All numbers on DEVELOPMENT seeds 1001-1010 (tuning on 101-105). Held-out 4001-4010 sealed (guard in
sim/tvt/seeds.py; no tvt-freeze tag). Detailed reports: results/TVT/<T>/report.md (not committed).

| milestone | status | GPU 1 wall time | key numbers |
|---|---|---|---|
| T0 harmonization | DONE | ~1.2 h | power_sum sandbox reproduces the frozen dev results of papers 1 and 2 exactly (max diff 0); best_beam: -1.3 dB median vs power sum, reference margin 34.8 dB (35.9); value of foresight 71-80 % of the gap (74-89 %); paper-2 exact-position planner beats A5 only at 25 dB |
| T1 bound | DONE (acceptance as written NOT met, see report) | ~0.1 h | PEB map-aided 3.4 mm -> 5.9 mm for any sigma_map >= 0.1 m (LoS-only 48 mm); element position errors 0.1 mm double the LoS-only PEB |
| T2 extraction | DONE | ~0.1 h | isolated paths at the CRB (RMS err/CRB 1.0); 87-90 % of strong paths unresolvable (err 20-100x CRB); 1.7 ms/snapshot |
| T3 visibility | DONE | ~0.2 h | perfect tracks AUC 1.00 (0 s) / 0.98 (1 s); real tracks AUC 0.74, pedestrians ~0.5; recalibrated Brier 0.19 (raw 0.29, climatology 0.22) |
| T4 tracker | IN PROGRESS (final runs after a face-merge fix; pre-fix results archived, results/TVT/T4/v2) | ~3 h | pre-fix: tracker median 3.8 cm vs paper-2 estimator 12.8 cm (p 0.002), during blockage 6.7 cm vs 1.01 m; visibility prediction from real tracks: no effect (J1 not supported); ~11x above the map-aided PEB |
| T5 planner, baselines | IN PROGRESS (rerun with the fixed tracker; learned baseline waits for the training-seed traces) | ~0.5 h | pre-fix: A5 best practical scheme at every margin; real-track planners 2-3x A5 outage |
| T6 sensitivity | IN PROGRESS (scene design committed bb1f60f before tracing; traces running in frozen clones) | ~4 h so far | blockage-model check done: Sionna first-order diffraction vs model B, 10 dB agreement 88-98 %, Sionna up to 10-14 dB more loss |

## Notes on process
- Provenance: the sensing-trace caches carry whole-tree provenance (commit + hash of every uncommitted
  change). Traces that are followed by detections therefore run from frozen clones of committed trees inside
  results/TVT/ (scripts/tvt_frozen.sh, tvt_frozen_at.sh); restamp_cache.py was NOT used. The first training
  and intersection trace attempts in the working tree were stopped and discarded for this reason.
- GPU 1 memory briefly reached 66 GB (> 60 GB rule) while four trace runs, three tracker streams and T5 ran
  together; the training-seed traces were stopped at once (restartable) and runs are now staggered.

## Open questions for the humans
1. Paper-1 text says "SNR of the best beam"; the paper-1 code (and all published numbers) use the power
   sum over paths with full array gain each (T0 Finding 1). The journal uses best_beam as decided.
2. T1 acceptance: the uncertain-map bound saturates at ~1.75x the perfect-map PEB and never approaches
   the LoS-only bound (facade offsets are self-calibrated within one epoch). Accept as finding, or define
   a per-VA map-error model?
3. Array orientation: the paper arrays face +x along the street; a deployment facing the street would
   change the scenario (not done).
4. CHO cannot be distinguished from A3/A5 in the paper-1 simulator (no command latency / failure model).
