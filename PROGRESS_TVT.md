# PROGRESS_TVT - journal extension (ROADMAP_TVT.md, GOAL_TVT.md)

All numbers on DEVELOPMENT seeds 1001-1010 (tuning on 101-105). Held-out 4001-4010 sealed (guard in
sim/tvt/seeds.py; no tvt-freeze tag). **Current scenario: back-to-back TR 38.901 panels (third round, decision 8);
intersection with one panel per street arm (fourth round, decision 12); consolidated report results/TVT/report.md
and results/TVT/panels_vs_iso.md.** The rows below up to the signaling
round are the isotropic-UPA rounds (reports moved to results/TVT_iso/<T>/report.md, not committed).

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
| Signaling round before the freeze (human item 6) | DONE (isotropic) | ~2.4 h (CPU-bound, GPU 1 < 10 GB) | every scheme re-tuned and T5/T6 re-run under "ideal" and "failure_aware" signaling (RLF/T310, HO-command failure, re-establishment tau_RE 1.83 s, proper CHO), primary metrics wrap-masked; failures rare (A5 RLF <= 0.04/min, HOF <= 2 % in the reference canyon); CHO = A5 under ideal signaling, beats A5 at 15 dB under failure-aware (-0.050, p 0.0039); every real-input sensing scheme still significantly worse than A5 at every margin of the reference canyon under both models; pre-declared sweep (T310 0.2/0.5/1 s, Qout +-3 dB, tau_RE 0.23/1.83 s) changes no conclusion; overhead 0: truth-UE perfect-track planner beats A5 at 25 dB only (-0.10, p 0.0039); learned threshold at the new grid edge 0.95 = A3 + overhead |
| Third round: back-to-back TR 38.901 panels (decisions 8-11) | DONE | ~24 h of GPU-1 jobs (~20.5 h elapsed; estimate 35 h) | pattern applied to traced angles (exact vs Sionna, <= 5e-6; radar re-traced once untilted because the tilted trace is not exact); every stage T0-T6 + J3 re-run, every scheme re-tuned under both signaling models. Reference margin 34.8 -> 38.8 dB; PEB LoS 48 -> 37 mm, map 3.4 -> 2.8 mm; radar detections +55-65 %; real-track AUC 0.75 -> 0.73; tracker 3.7 cm unchanged, paper-2 estimator A 12.8 -> 10.1 cm (re-tuned); real-input penalties vs A5 about halved but still significant at every margin; truth-UE perfect-track planner beats A5 at 20 and 30 dB (25 dB p 0.064); learned trigger switched off by tuning at 4/5 margins (fires rarely at 25 dB, no gain); J3 part e: true future = extrapolated (constant-velocity blockers); intersection: +-x panels leave the cross street in a -15 dBi hole (A5 2.0 -> 12.5 s/UE-min at 15 dB) |
| Fourth round: intersection panels per street arm (decisions 12-16) | DONE | ~1.2 h (radar 0.3 h, positioning 0.7 h, handover 0.3 h) | four panels (yaw 0/180/90/270) at both corner O-RUs and the radar, rule committed before the run (09746ff); pattern exact vs Sionna also at 90/270 deg (<= 3.1e-5). Hole closed: UE 1 median unblocked power -84.7 -> -67.3 dB (O-RU 0); A5 outage 12.5 -> 1.95 s/UE-min at 15 dB (re-tuned, failure-aware; isotropic 2.0), reference 0.187 -> 0.033. PEB LoS 77 -> 52 mm, map 3.7 -> 1.7 mm; tracker median 4.1 -> 2.6 cm, p90 0.28 -> 0.12 m (pred_real). Handover: no scheme beats A5 at any margin, both models, canyon and re-tuned parameters; real-input schemes worse at every margin (p <= 0.02, except risk-aware / risk-neutral at the reference margin: +0.013, n.s.); perfect tracks: truth UE +0.007..+0.106 (worse at 20 dB, p 0.049, else n.s.), tracker UE +0.014..+0.157 (worse at 20 and 25 dB, p 0.049) (re-tuned, failure-aware). CHO = A5. +-x results kept in results/TVT/supplement_intersection_pmx |

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
9. Learned trigger: no further grid change; reported as "tuning switches the predictor off" (panels: threshold
   0.95 = never fires, verified by a replay at threshold 1.01, at 4 of 5 margins; 0.725 at 25 dB, fires rarely and
   is 0.004 s/UE-min worse than never firing).
10. Primary signaling model: failure_aware; ideal signaling as supplement (configs/tvt.yaml `primary:`).
11. New J3 diagnosis part e: truth-UE planner at overhead 0 with perfect-at-decision blocker states
    extrapolated by the motion model vs the true future trajectories, vs A5, by blocker class. Result: identical
    (blockers move at exactly constant velocity; only lane-end wraps differ and change no timeline), so the
    perfect-vs-real gap is entirely sensing accuracy and overhead in this scenario.
12. Fourth round, intersection panel orientation (reason: deployment planning, not results): "each O-RU has
    one 38.901 panel facing along each street arm it serves". Both cells serve all four arms, so each corner
    O-RU and the radar at O-RU 0 have four panels, yaw 0 / 180 / 90 / 270 deg (configs/tvt.yaml
    panels.yaw_deg_by_mount.corner; rule and azimuths in results/TVT/T6/scene_design.md and the addendum of
    configs/tvt_intersection_design.md, committed 09746ff BEFORE any run). Only the intersection parts of T6 are
    re-run (both signaling models; re-tuned on the intersection tuning seeds with the same grids). The +-x
    intersection results stay as a supplement ("naive canyon orientation leaves a coverage hole"):
    results/TVT/supplement_intersection_pmx/.
13. Blocker motion: constant velocity on straight lanes kept. **Limitation (favours the sensing schemes):** the
    blockers' future trajectories are perfectly predictable by construction (J3 part e: true future = extrapolated
    state), so the sensing-based predictors face no manoeuvre (turn, stop, speed change) uncertainty; A5 and the
    reactive baselines do not use trajectory prediction and are unaffected. Results are an upper bound on the
    value of blocker tracking with respect to trajectory predictability.
14. Narrative recorded: "requirements and achievable region; solving UE positioning moves the bottleneck to
    blocker-sensing accuracy and overhead". Reports are not reworded.
15. Report of the planner with perfect tracks and UE positions from the tracker vs from truth, vs A5, per margin
    and blocker class with seed-level statistics (panels, failure-aware primary, wrap-masked):
    scripts/tvt_planner_ue_report.py -> results/TVT/J3_diagnosis/planner_ue_report.md.
16. configs/tvt_frozen/manifest.json git_commit filled from the host (scripts/tvt_freeze_stamp.sh stamp, after
    scripts/tvt_freeze_params.py in the container) and checked (scripts/tvt_freeze_stamp.sh check; test_tvt.py
    requires a full hash).

## Open questions for the humans
1. Intersection, perfect tracks + truth UE: with the street-arm panels this planner does NOT beat A5 at any margin
   (canyon: beats A5 at 20 and 30 dB). The intersection's re-tuned grids are the canyon's; nothing was changed.
   Report as is (J5: the planner's headroom is deployment dependent)?

Resolved in the fourth round: intersection orientation (decision 12), blocker motion (decision 13, limitation),
narrative (decision 14).
Resolved in the third round: array back half-space (decision 8 replaced the iso arrays); learned-trigger grid
(decision 9); primary signaling model (decision 10).
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
- [x] All TVT code on branch tvt; unit tests pass (scripts/test_tvt.py, test_tvt_panels.py, test_tvt_t1..t5.py;
      55 tests: pattern values, panel selection incl. four panels (yaw 0/180/90/270), radar transform, tracker panel
      gating and rotated-frame prediction, Sionna exactness on GPU at yaw 0/180/90/270, manifest commit).
- [x] Intersection panel orientation fixed before any run (09746ff; configs/tvt.yaml panels.yaw_deg_by_mount).
- [x] Arrays fixed in configs/tvt.yaml `panels:` with sources before any run (ba97c09); radar re-traced untilted
      (results/TVT/radar/trace, stage-scoped provenance); the T7 held-out run needs the same untilted radar trace
      and per-panel detections (scripts/tvt_radar.py trace/detect), the panel SRS measurements and estimator-A
      tracks (scripts/tvt_t4_measure.py, scripts/tvt_est_tune.py run; parameters from configs/tvt_frozen/p2_estimator.json).
- [x] Tuned parameters frozen in committed files (configs/tvt_frozen/, written by scripts/tvt_freeze_params.py
      from the result files, sha256 of every source in manifest.json): tracker.json (T4 tracker + visibility
      prior), visibility_calibration.json (T3 isotonic calibration), handover_ideal.json and
      handover_failure_aware.json (per-margin parameters of every scheme incl. CHO, trigger_learned and the
      J3/J4 ablations, tuned per signaling model), handover_intersection_<model>.json (T6 own tuning per model),
      p2_estimator.json (estimator A re-tuned on the panel measurements, results/TVT/est_A/est_tuned_A.json;
      manifest "array": back_to_back), learned.json (results/TVT/T5/learned.pt by path + sha256). The
      first-round handover.json / handover_intersection.json are superseded and removed. T7 commands (no tuning):
      `tvt_t4_track.py --params configs/tvt_frozen/tracker.json --calibration configs/tvt_frozen/visibility_calibration.json`
      and, for each model in {ideal, failure_aware},
      `tvt_t5_handover.py --signaling <model> --fixed configs/tvt_frozen/handover_<model>.json --learned-manifest configs/tvt_frozen/learned.json`.
      Regenerated in the fourth round (only handover_intersection_<model>.json changed: per-street-arm panels);
      manifest.json git_commit = e277226 (stamped from the host, scripts/tvt_freeze_stamp.sh stamp; verify with
      scripts/tvt_freeze_stamp.sh check: commit exists, is an ancestor of HEAD, and only configs/tvt_frozen/ and
      *.md changed since). FrozenParamsTest checks the checksums, the array field and the commit field.
      NOTE: learned.pt lives in results/ (not committed); keep or archive it with the tag.
- [ ] Held-out traces: 4001-4010 need sensing caches, detections, comm geometry and path coefficients (as for
      the training seeds: ~1-2 h with 4 workers alone on GPU 1), run from a frozen clone of the tagged tree
      (scripts/tvt_frozen_at.sh) because of the whole-tree provenance of the sensing caches.
- [x] Training-seed traces (5001-5040) and learned baseline done.
- [x] Human decisions 1-16 recorded above.
- [ ] Remaining human decisions: open question 1 (report only); creating the tag tvt-freeze (humans only).

## J1-J5 summary (back-to-back panels, development seeds 1001-1010; primary failure-aware, wrap-masked; exact Wilcoxon over 10 seeds)
| hypothesis | verdict | evidence |
|---|---|---|
| J1 predicted visibility improves association | NOT SUPPORTED | Real-track visibility: +0.3 mm median vs none (p 0.049); oracle visibility during blockage -1.0 mm (p 0.49; isotropic -3.2 mm, p 0.049). T3 real-track AUC 0.73 (isotropic 0.75). Same in every variant. |
| J2 tracker narrows the gap to the PEB | SUPPORTED (gap narrowed, not closed) | Tracker median 3.7 cm vs re-tuned estimator A 10.1 cm (all) and 6.8 cm vs 26 cm during blockage (p 0.002); ~13x the map-aided PEB (2.8 mm). Tracker better in every calibration, bandwidth, height, speed variant and the intersection (2.6 cm there, PEB map 1.7 mm). |
| J3 planner beats A5 with perfect tracks | MIXED: truth UE yes at 20/30 dB, TVT UE no (results/TVT/J3_diagnosis/planner_ue_report.md: tracker UE - truth UE +0.05..+0.07 at 20-30 dB, p 0.002 at 20 and 30 dB, more than half in pedestrian blockages) | Truth UE + perfect tracks: -0.071 (20 dB, p 0.037), -0.044 (30 dB, p 0.002); at overhead 0 also 25 dB and the reference margin. TVT UE + perfect tracks: n.s. at every margin. Real tracks: +0.06..+2.1 worse than A5 (p <= 0.049). Learned trigger: tuning switches it off. Part e: true future = extrapolation, so the gap is sensing accuracy. CHO beats A5 at 15 and 25 dB under failure-aware signaling. |
| J4 risk-aware beats risk-neutral | NOT SUPPORTED | Real inputs: tuned to lambda = 0 (identical to risk-neutral) at 15-25 dB and the reference margin; at 30 dB -0.003 (failure-aware, p 0.52) / -0.023 (ideal, p 0.38). Perfect tracks: risk-aware vs risk-neutral -0.010..+0.006, p >= 0.08; the look-ahead (not CVaR) beats the deterministic planner at 30 dB (-0.024, p 0.004). |
| J5 conclusions robust | SUPPORTED, with one deployment caveat | E2 0-100 ms, overhead x0-x2, SI up to INR 40 dB, UE speed, O-RU height 3/10 m, both signaling models, failure-model sweep: no real-input scheme significantly better than A5 anywhere. Isotropic -> panels changes magnitudes, not the ranking. Intersection with one panel per street arm (decision 12): A5 best, no sensing scheme better even with perfect tracks; supplement: naive canyon orientation (+-x) leaves a coverage hole on the cross street (A5 12.5 vs 1.95 s/UE-min at 15 dB). Limitation: blocker trajectories perfectly predictable (decision 13). |

Isotropic-round J1-J5 (superseded scenario): results/TVT_iso and git history of this file.

All reports: results/TVT/report.md, results/TVT/panels_vs_iso.md (panels); results/TVT_iso/{T0..T6,J3_diagnosis}/report.md (isotropic); not committed.
