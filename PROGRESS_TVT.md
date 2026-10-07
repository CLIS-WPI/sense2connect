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
| Signaling round before the freeze (human item 6) | DONE | ~2.4 h (CPU-bound, GPU 1 < 10 GB) | every scheme re-tuned and T5/T6 re-run under "ideal" and "failure_aware" signaling (RLF/T310, HO-command failure, re-establishment tau_RE 1.83 s, proper CHO), primary metrics wrap-masked; failures rare (A5 RLF <= 0.04/min, HOF <= 2 % in the reference canyon); CHO = A5 under ideal signaling, beats A5 at 15 dB under failure-aware (-0.050, p 0.0039); every real-input sensing scheme still significantly worse than A5 at every margin of the reference canyon under both models; pre-declared sweep (T310 0.2/0.5/1 s, Qout +-3 dB, tau_RE 0.23/1.83 s) changes no conclusion; overhead 0: truth-UE perfect-track planner beats A5 at 25 dB only (-0.10, p 0.0039); learned threshold at the new grid edge 0.95 = A3 + overhead |

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
5. Narrative after T4: the humans chose "requirements and achievable region" (reports not reworded).
6. Before the freeze (second round): handover-failure model + proper CHO applied to all schemes, every
   result under "ideal" and "failure_aware" signaling, re-tuned per model with equal budgets; values,
   sources and the T6 sensitivity sweep fixed in configs/tvt.yaml (signaling:) BEFORE any run (commit
   of that config precedes all failure-aware results). Primary metrics exclude the 1 s after each UE
   wrap (configs/tvt.yaml evaluation:), unmasked metrics as supplement. Learned-trigger threshold grid
   0.5 / 0.725 / 0.95 (same 3 points). Array orientation unchanged. Truth-UE planner with perfect tracks at
   sensing overhead 0 vs A5 added to the J3 diagnosis (part d).
7. Array model check (human item 4; orientation NOT changed): the O-RU arrays are 8x8 UPAs of isotropic elements
   in the y-z plane with the default orientation (boresight +x) and no back plane, so they radiate and receive
   equally into the back half-space (u_x < 0) with the same DFT beams (mirror ambiguity). Measured on the
   development seeds: for O-RU 0, 50.2 % of the UE epochs and 50 % of the LoS paths lie behind the array plane
   and 45 % of the path power arrives from behind; for O-RU 1, 9.2 % of the UE epochs and 33 % of the path
   power. A deployed panel with a ground plane would see ~none of this. Recorded as a limitation of the
   scenario inherited from papers 1 and 2 (it applies to every scheme alike); see open question 1.
8. Third round (physical arrays): every O-RU array, the oru-0 radar included, becomes two back-to-back 8x8
   panels (+x / -x) with the TR 38.901 Table 7.3-1 element pattern (8 dBi, 65 deg, 30 dB); link and sensing
   use the panel facing the target, positioning uses both (configs/tvt.yaml `panels:`, fixed with sources
   before any run). The paper-1 radar looked at the street centre, not along +x; the humans chose the same
   +-x panels for the radar too. Pattern applied to the traced path angles (exact; verified against a Sionna
   re-trace), every stage T0-T6 and the J3 diagnosis re-run and every scheme re-tuned; the isotropic results
   are kept in results/TVT_iso for the comparison.
   GPU estimate before running (wall-clock hours of jobs holding GPU 1, upper bound from the logged run
   times of the isotropic rounds): sensing re-trace for provenance ~290 jobs ~8 h; per-panel detections
   (2 panels, incl. residual SI) ~8 h; T0 ~1.2 h; T1-T3 ~0.5 h; T4 ~5 h; T5 incl. learned baseline and the
   estimator-A re-tuning ~3 h; T6 incl. both signaling models ~8 h; J3 ~1 h: ~35 h < 48 h, so it runs.
9. Learned trigger: no further grid change; reported as "tuning switches the predictor off".
10. Primary signaling model: failure_aware; ideal signaling as supplement (configs/tvt.yaml `primary:`).
11. New J3 diagnosis part e: truth-UE planner at overhead 0 with perfect-at-decision blocker states
    extrapolated by the motion model vs the true future trajectories, vs A5, by blocker class.

## Open questions for the humans
1. Array back half-space (decision 7): keep the iso-element arrays without back plane for T7 and state it as a
   limitation, or add a back plane / element pattern (would change the scenario and need a new development
   round before the freeze)? Nothing was changed.
2. Learned trigger: on the extended grid (0.5 / 0.725 / 0.95, same 3 points) the threshold is tuned to the new
   upper edge 0.95 at every margin and under both signaling models; at 0.95 the learned trigger no longer
   fires on its own (HO/min = A3) and the scheme is A3 plus sensing overhead. Report it as "tuning switches the
   learned predictor off" for T7, or allow a further grid change (would be a third tuning of this baseline)?
3. Primary signaling model for the paper's main tables: both are frozen and reported; the rankings agree.
   The failure-aware model is the more defensible primary choice (CHO is only meaningful there).

Resolved in the signaling round: CHO model (now with preparation, execution condition and failure model);
UE wrap (kept, 1 s after each wrap excluded from primary metrics, unmasked as supplement); narrative choice.

## Freeze-readiness checklist (T7 prerequisites; status on development seeds)
- [x] Seed sets and held-out guard (configs/seeds_tvt.yaml, sim/tvt/seeds.py): 4001-4010 refused without the
      tag tvt-freeze; consumed 2001-2010/3001-3010 always refused; no tvt-freeze tag exists.
- [x] Service model fixed (configs/tvt.yaml service.model = best_beam) and regression of papers 1/2 under the
      power-sum model (exact).
- [x] Signaling / failure model, evaluation mask and learned grid fixed in configs/tvt.yaml with sources,
      committed before any run that uses them (65d7a6c); sim/tvt/signaling.py equals the paper-1 simulator with
      signaling off (tested).
- [x] All TVT code on branch tvt; unit tests pass (scripts/test_tvt.py, test_tvt_t1..t5.py; 33 tests).
- [x] Tuned parameters frozen in committed files (configs/tvt_frozen/, written by scripts/tvt_freeze_params.py
      from the result files, sha256 of every source in manifest.json): tracker.json (T4 tracker + visibility
      prior), visibility_calibration.json (T3 isotonic calibration), handover_ideal.json and
      handover_failure_aware.json (per-margin parameters of every scheme incl. CHO, trigger_learned and the
      J3/J4 ablations, tuned per signaling model), handover_intersection_<model>.json (T6 own tuning per model),
      p2_estimator.json (paper-2 estimator A), learned.json (results/TVT/T5/learned.pt by path + sha256). The
      first-round handover.json / handover_intersection.json are superseded and removed. T7 commands (no tuning):
      `tvt_t4_track.py --params configs/tvt_frozen/tracker.json --calibration configs/tvt_frozen/visibility_calibration.json`
      and, for each model in {ideal, failure_aware},
      `tvt_t5_handover.py --signaling <model> --fixed configs/tvt_frozen/handover_<model>.json --learned-manifest configs/tvt_frozen/learned.json`.
      Verified: the frozen failure-aware file reproduces the development results exactly (A5, CHO,
      trigger_learned, planner_tvt, planner_true_perfect, all margins: 125 values, max diff 0); FrozenParamsTest
      checks the checksums. NOTE: learned.pt lives in results/ (not committed); keep or archive it with the tag.
- [ ] Held-out traces: 4001-4010 need sensing caches, detections, comm geometry and path coefficients (as for
      the training seeds: ~1-2 h with 4 workers alone on GPU 1), run from a frozen clone of the tagged tree
      (scripts/tvt_frozen_at.sh) because of the whole-tree provenance of the sensing caches.
- [x] Training-seed traces (5001-5040) and learned baseline done.
- [x] Human decisions 1-7 recorded above.
- [ ] Remaining human decisions: open questions 1-3 (array back half-space, learned-threshold grid edge, primary
      signaling model); creating the tag tvt-freeze (humans only).

## J1-J5 summary (development seeds 1001-1010 only; seed-level exact Wilcoxon over 10 seeds)
| hypothesis | verdict | evidence |
|---|---|---|
| J1 predicted visibility improves association | NOT SUPPORTED (slightly negative) | Visibility predicted from real tracks: +1.4 mm median error vs no visibility (p 0.02); no gain during blockage or in the second before onset. Oracle visibility: -3.2 mm during blockage (p 0.049), so the mechanism works but the predictor is too weak (T3: real-track AUC 0.74, pedestrians ~0.5). Same picture in every calibration, bandwidth and variant sweep and in the intersection (p 0.56). |
| J2 tracker narrows the gap to the PEB | SUPPORTED (gap narrowed, not closed) | Median 3.7 cm vs 12.8 cm for the paper-2 estimator; during blockage 6.4 cm vs 1.01 m (p 0.002). About 11x the map-aided PEB remains. Causes: 87-90 % of the strong paths cannot be resolved in one snapshot (T2), long tails after reinitialisation, and map error (p90 0.44 -> 0.91 m for sigma_map 0.1 -> 1 m). Both narratives are documented in the T4 report, as instructed; no choice made. |
| J3 planner beats A5 with perfect tracks | NOT SUPPORTED in the reference canyon; MIXED across variants | Reference canyon, both signaling models, wrap-masked: planner with TVT positions and perfect tracks never beats A5; exact positions beat A5 at 25 dB only (-0.071, p 0.0098), and still only at 25 dB with zero sensing overhead (-0.10, p 0.0039; x0 re-tuned). UE-speed variants: perfect-track planners beat A5 at 25 dB (exact positions at 20-30 dB). Real tracks: significantly worse than A5 at every margin; the learned trigger tunes itself off (threshold 0.95, = A3 + overhead). CHO = A5 with ideal signaling; with failure-aware signaling CHO beats A5 at 15 dB (-0.050, p 0.0039), so the best practical scheme is a reactive one in both models. Diagnosis: the headroom exists (value of foresight 71-80 % of the gap), but missed switches (mostly pedestrians) and sensing overhead cancel the planner's better timing; tracker UE adds +0.03..+0.07 masked plus up to +0.08 in the wrap windows. |
| J4 risk-aware beats risk-neutral, same inputs | NOT SUPPORTED | Real inputs: no better at any margin, worse at the reference margin (p 0.006). Perfect tracks (added ablation riskneutral_tvt_perfect, same grid, tuned on the tuning seeds, results/TVT/T5/handover_j4_perfect.json): differences -0.008..0 s/UE-min, p >= 0.45 at every margin. The perfect-track risk-aware planner's advantage over the deterministic planner (15 dB, 30 dB, reference) comes from the sample-based look-ahead, not from the CVaR term (lambda = 0 does as well). |
| J5 conclusions robust | SUPPORTED for the negative/positive pattern above | Map error, calibration, sync, bandwidth, O-RU height, UE speed, density, E2 delay 0-100 ms, sensing overhead x0-x2, residual SI up to 40 dB, a second deployment (intersection), ideal vs failure-aware signaling and the pre-declared failure-model sweep (T310 0.2/0.5/1 s, Qout +-3 dB, tau_RE 0.23/1.83 s) leave the conclusions unchanged: the tracker beats paper-2 everywhere (paper-2 estimator collapses at 3 m O-RU height: 1.50 m vs 4.3 cm), and A5 (CHO under failure-aware signaling) is the best practical handover scheme everywhere; no real-input sensing scheme is significantly better than A5 in any setting. |

All reports: results/TVT/{T0..T6,J3_diagnosis}/report.md (not committed).
