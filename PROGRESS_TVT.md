# PROGRESS_TVT - journal extension (ROADMAP_TVT.md, GOAL_TVT.md)

All numbers on DEVELOPMENT seeds 1001-1010 (tuning on 101-105). Held-out 4001-4010 sealed (guard in
sim/tvt/seeds.py; no tvt-freeze tag). Detailed reports: results/TVT/<T>/report.md (not committed).

| milestone | status | GPU 1 wall time | key numbers |
|---|---|---|---|
| T0 harmonization | DONE | ~1.2 h | power_sum sandbox reproduces the frozen dev results of papers 1 and 2 exactly (max diff 0); best_beam: -1.3 dB median vs power sum, reference margin 34.8 dB (35.9); value of foresight 71-80 % of the gap (74-89 %); paper-2 exact-position planner beats A5 only at 25 dB |
| T1 bound | DONE (acceptance as written NOT met; saturation accepted as a finding by the humans after the free-VA sanity check) | ~0.15 h | PEB map-aided 3.4 mm -> 5.9 mm for any sigma_map >= 0.1 m (LoS-only 48 mm); element position errors 0.1 mm double the LoS-only PEB; sanity check with a free 3-D virtual anchor per NLoS path: 9.9 / 33 / 47 mm at sigma_va 0.01 / 0.1 / 1 m and equal to LoS-only (48 mm) for sigma_va >= 10 m (per epoch within 1.4e-4): no bug; the saturation comes from few shared facade offsets |
| T2 extraction | DONE | ~0.1 h | isolated paths at the CRB (RMS err/CRB 1.0); 87-90 % of strong paths unresolvable (err 20-100x CRB); 1.7 ms/snapshot |
| T3 visibility | DONE | ~0.2 h | perfect tracks AUC 1.00 (0 s) / 0.98 (1 s); real tracks AUC 0.74, pedestrians ~0.5; recalibrated Brier 0.19 (raw 0.29, climatology 0.22) |
| T4 tracker | DONE | ~5 h | tracker median 3.7 cm vs paper-2 estimator 12.8 cm, during blockage 6.4 cm vs 1.01 m (p 0.002); ~11x above the map-aided PEB; predicted visibility (real tracks) slightly WORSE than none (+1.4 mm, p 0.02): J1 not supported; oracle visibility -3.2 mm during blockage (p 0.049) |
| T5 planner, baselines | DONE | ~6 h (incl. ~4.8 h training-seed traces 5001-5040) | A5 best practical scheme at every margin; real-track planners +0.19..+3.7 s/UE-min vs A5 (p 0.002); TVT positions + perfect tracks never beat A5; exact positions + perfect tracks beat A5 only at 25 dB; risk-aware = risk-neutral with real and with perfect inputs; learned trigger (val. AUC 0.70-0.74) is the best sensing scheme with real inputs at 15-25 dB (-0.6..-3.4 vs planner_tvt, p 0.002) but +0.14..+0.24 worse than A5 (p 0.002) |
| T6 sensitivity | DONE | ~9 h | E2 delay 0-100 ms, overhead x0-x2, residual SI up to INR 40 dB: no conclusion changes; tracker beats paper-2 estimator in every calibration/bandwidth/height/speed variant (O-RU 3 m: 4.3 cm vs 1.50 m); A5 best practical scheme in every variant and in the intersection (canyon and re-tuned parameters); perfect-track planners beat A5 only in UE-speed variants at 25 dB; intersection PEB map 4.5 mm / LoS 93 mm, tracker 4.75 cm; Sionna diffraction vs model B 10 dB agreement 88-98 % |
| J3 diagnosis (human request) | DONE | ~0.5 h | value of foresight survives best beam (0.05-0.28 s/UE-min, 71-80 % of the gap; path sum 0.05-0.25, 74-89 %); truth-UE perfect-track planner beats A5 at 25 dB (-0.073, p 0.0098), tracker UE costs +0.05..+0.11 (23-62 % of it in the 1 s after the scenario's UE wrap); white errors with the tracker's RMS are far worse (+0.18..+2.0); remaining gap = sensing overhead + missed switches (mostly pedestrians) vs fewer mistimed switches; results/TVT/J3_diagnosis/report.md |

## Notes on process
- Provenance: the sensing-trace caches carry whole-tree provenance (commit + hash of every uncommitted
  change). Traces that are followed by detections therefore run from frozen clones of committed trees inside
  results/TVT/ (scripts/tvt_frozen_at.sh; the first runs used an equivalent single-clone version); restamp_cache.py was NOT used. The first training
  and intersection trace attempts in the working tree were stopped and discarded for this reason.
- GPU 1 memory briefly reached 66 GB (> 60 GB rule) while four trace runs, three tracker streams and T5 ran
  together; the training-seed traces were stopped at once (restartable) and runs are now staggered.

## Human decisions (recorded)
1. Service model: best-beam SNR for all journal results. **Both conference papers (paper 1, WCNC 2027, and
   paper 2) used the path-power sum** (power of every path with full 64-element array gain, added over
   paths; sim/comm/linkbudget.py, xapp/timeline.py), although the paper-1 text says "SNR of the best beam".
   Under best beam the paper-1 conclusions keep their shape (T0; J3 diagnosis part a): outage rises, the
   reference margin moves from 35.9 to 34.8 dB, the value of foresight stays 71-80 % of the gap.
2. T1 saturation accepted as a finding after the free-VA sanity check (T1 report addendum).
3. Tuned parameters copied into committed files configs/tvt_frozen/ (scripts/tvt_freeze_params.py);
   learned weights referenced by path + sha256; T7 must not re-tune.
4. Point on held-out seeds closed by the humans: nothing was run for 4001-4010.

## Open questions for the humans
1. Array orientation: the paper arrays face +x along the street; a deployment facing the street would
   change the scenario (not done).
2. CHO cannot be distinguished from A3/A5 in the paper-1 simulator (no command latency / failure model).
3. The paper-1 scenario wraps the UEs from one street end to the other (x jumps by 80 m); the tracker
   needs ~0.4 s to re-acquire, which dominates its RMS error and costs the tracker-UE planner 0.01-0.07
   s/UE-min (J3 diagnosis). It is a simulation artifact; the scenario was NOT changed. Keep it for T7 (as
   frozen) or document it as a limitation?
4. The learned trigger's threshold was tuned to the upper grid edge (0.7) at every margin; the grid was not
   extended (equal tuning budget). Accept as is for T7?
5. Narrative choice after T4 ("closes the gap" vs "requirements and achievable region"): both written up,
   not chosen.

## Freeze-readiness checklist (T7 prerequisites; status on development seeds)
- [x] Seed sets and held-out guard (configs/seeds_tvt.yaml, sim/tvt/seeds.py): 4001-4010 refused without the
      tag tvt-freeze; consumed 2001-2010/3001-3010 always refused; no tvt-freeze tag exists.
- [x] Service model fixed (configs/tvt.yaml service.model = best_beam) and regression of papers 1/2 under the
      power-sum model (exact).
- [x] All TVT code on branch tvt; unit tests pass (scripts/test_tvt.py, test_tvt_t1..t5.py; 30 tests).
- [x] Tuned parameters frozen in committed files (configs/tvt_frozen/, written by scripts/tvt_freeze_params.py
      from the result files, sha256 of every source in manifest.json): tracker.json (T4 tracker + visibility
      prior), visibility_calibration.json (T3 isotonic calibration), handover.json (per-margin parameters of
      every scheme incl. trigger_learned and the J3/J4 ablations; duplicates verified identical),
      handover_intersection.json (T6 own tuning), p2_estimator.json (paper-2 estimator A), learned.json
      (results/TVT/T5/learned.pt by path + sha256). T7 runs with `tvt_t4_track.py --params
      configs/tvt_frozen/tracker.json --calibration configs/tvt_frozen/visibility_calibration.json` and
      `tvt_t5_handover.py --fixed configs/tvt_frozen/handover.json --learned-manifest configs/tvt_frozen/learned.json`
      (no tuning). Verified: these reproduce the development results exactly (tracker max |xy diff| 0.0 m on
      one job; handover 20 dB A5 / trigger_learned / planner_tvt / planner_tvt_perfect identical);
      FrozenParamsTest checks the checksums. NOTE: learned.pt itself lives in results/ (not committed); it
      must be kept (or archived with the tag) for T7.
- [ ] Held-out traces: 4001-4010 need sensing caches, detections, comm geometry and path coefficients (as for
      the training seeds: ~1-2 h with 4 workers alone on GPU 1), run from a frozen clone of the tagged tree
      (scripts/tvt_frozen_at.sh) because of the whole-tree provenance of the sensing caches.
- [x] Training-seed traces (5001-5040) and learned baseline done.
- [x] Human decisions on service model, T1 acceptance and parameter freeze recorded above.
- [ ] Remaining human decisions: see "Open questions" (array orientation, CHO model, UE wrap, learned-threshold
      grid edge, narrative after T4); creating the tag tvt-freeze (humans only).

## J1-J5 summary (development seeds 1001-1010 only; seed-level exact Wilcoxon over 10 seeds)
| hypothesis | verdict | evidence |
|---|---|---|
| J1 predicted visibility improves association | NOT SUPPORTED (slightly negative) | Visibility predicted from real tracks: +1.4 mm median error vs no visibility (p 0.02); no gain during blockage or in the second before onset. Oracle visibility: -3.2 mm during blockage (p 0.049), so the mechanism works but the predictor is too weak (T3: real-track AUC 0.74, pedestrians ~0.5). Same picture in every calibration, bandwidth and variant sweep and in the intersection (p 0.56). |
| J2 tracker narrows the gap to the PEB | SUPPORTED (gap narrowed, not closed) | Median 3.7 cm vs 12.8 cm for the paper-2 estimator; during blockage 6.4 cm vs 1.01 m (p 0.002). About 11x the map-aided PEB remains. Causes: 87-90 % of the strong paths cannot be resolved in one snapshot (T2), long tails after reinitialisation, and map error (p90 0.44 -> 0.91 m for sigma_map 0.1 -> 1 m). Both narratives are documented in the T4 report, as instructed; no choice made. |
| J3 planner beats A5 with perfect tracks | NOT SUPPORTED in the reference canyon; MIXED across variants | Reference canyon: planner with TVT positions and perfect tracks never beats A5; exact positions beat A5 at 25 dB only (-0.073, p 0.0098). UE-speed variants: perfect-track planners beat A5 at 25 dB (exact positions at 20-30 dB). Real tracks: +0.19..+3.7 s/UE-min worse than A5 everywhere; the learned trigger too (+0.14..+0.24). CHO = A5 in this simulator. Diagnosis: the headroom exists (value of foresight 71-80 % of the gap under best beam), but the sensing overhead and missed switches (mostly pedestrians) cancel the planner's better timing; the tracker UE adds +0.05..+0.11, partly the scenario's UE wrap. |
| J4 risk-aware beats risk-neutral, same inputs | NOT SUPPORTED | Real inputs: no better at any margin, worse at the reference margin (p 0.006). Perfect tracks (added ablation riskneutral_tvt_perfect, same grid, tuned on the tuning seeds, results/TVT/T5/handover_j4_perfect.json): differences -0.008..0 s/UE-min, p >= 0.45 at every margin. The perfect-track risk-aware planner's advantage over the deterministic planner (15 dB, 30 dB, reference) comes from the sample-based look-ahead, not from the CVaR term (lambda = 0 does as well). |
| J5 conclusions robust | SUPPORTED for the negative/positive pattern above | Map error, calibration, sync, bandwidth, O-RU height, UE speed, density, E2 delay 0-100 ms, sensing overhead x0-x2, residual SI up to 40 dB and a second deployment (intersection) leave the conclusions unchanged: the tracker beats paper-2 everywhere (paper-2 estimator collapses at 3 m O-RU height: 1.50 m vs 4.3 cm), and A5 is the best practical handover scheme everywhere. |

All reports: results/TVT/{T0..T6,J3_diagnosis}/report.md (not committed).
