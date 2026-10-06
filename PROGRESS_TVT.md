# PROGRESS_TVT - journal extension (ROADMAP_TVT.md, GOAL_TVT.md)

All numbers on DEVELOPMENT seeds 1001-1010 (tuning on 101-105). Held-out 4001-4010 sealed (guard in
sim/tvt/seeds.py; no tvt-freeze tag). Detailed reports: results/TVT/<T>/report.md (not committed).

| milestone | status | GPU 1 wall time | key numbers |
|---|---|---|---|
| T0 harmonization | DONE | ~1.2 h | power_sum sandbox reproduces the frozen dev results of papers 1 and 2 exactly (max diff 0); best_beam: -1.3 dB median vs power sum, reference margin 34.8 dB (35.9); value of foresight 71-80 % of the gap (74-89 %); paper-2 exact-position planner beats A5 only at 25 dB |
| T1 bound | DONE (acceptance as written NOT met, see report) | ~0.1 h | PEB map-aided 3.4 mm -> 5.9 mm for any sigma_map >= 0.1 m (LoS-only 48 mm); element position errors 0.1 mm double the LoS-only PEB |
| T2 extraction | DONE | ~0.1 h | isolated paths at the CRB (RMS err/CRB 1.0); 87-90 % of strong paths unresolvable (err 20-100x CRB); 1.7 ms/snapshot |
| T3 visibility | DONE | ~0.2 h | perfect tracks AUC 1.00 (0 s) / 0.98 (1 s); real tracks AUC 0.74, pedestrians ~0.5; recalibrated Brier 0.19 (raw 0.29, climatology 0.22) |
| T4 tracker | IN PROGRESS | | |
| T5 planner, baselines | IN PROGRESS (training-seed traces running) | | |
| T6 sensitivity | NOT STARTED | | |

## Open questions for the humans
1. Paper-1 text says "SNR of the best beam"; the paper-1 code (and all published numbers) use the power
   sum over paths with full array gain each (T0 Finding 1). The journal uses best_beam as decided.
2. T1 acceptance: the uncertain-map bound saturates at ~1.75x the perfect-map PEB and never approaches
   the LoS-only bound (facade offsets are self-calibrated within one epoch). Accept as finding, or define
   a per-VA map-error model?
3. Array orientation: the paper arrays face +x along the street; a deployment facing the street would
   change the scenario (not done).
4. CHO cannot be distinguished from A3/A5 in the paper-1 simulator (no command latency / failure model).
