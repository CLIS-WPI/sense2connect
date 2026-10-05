# Paper 2: How accurately can mmWave infrastructure localize the user?

## Research question
Paper 1 found that ISAC-assisted proactive handover beats 3GPP A5 only
if the user position is known to about 0.08-0.13 m (one-error sweeps,
15-30 dB margin). Literature: best street-canyon uplink-SRS multi-gNB
AoA/ToA positioning reaches about 0.6-0.8 m RMSE (Expert Syst. Appl.
2026, doi 10.1016/j.eswa.2026.132887); decimeter results exist only for
target sensing/SLAM. Paper 2 asks: can the same two-O-RU 28 GHz
infrastructure localize the UE to the decimeter level, where, when,
and with which bandwidth, and what happens during blockage, exactly
when foresight is needed?
Hypotheses (to be TESTED, report null results honestly):
 P1 Decimeter accuracy is theoretically possible (position error bound
    <= 0.1 m) only with wide bandwidth and LoS to both O-RUs.
 P2 Accuracy degrades sharply during blockage events, i.e. precisely
    when blockage foresight would act.
 P3 Map-aided use of NLoS paths (digital twin) recovers part of the
    loss during blockage.
 P4 A practical estimator stays well above the bound; with realistic
    positioning error, the paper-1 planner does not beat A5.

## Scenario and data (reuse, do not change)
Same street canyon, two O-RUs (mounts lamppost 5 m / facade 8 m), 8x8
UPA at each O-RU, single-antenna pedestrian UEs on the south sidewalk,
traffic densities low/high, model B blockage, exactly as paper 1.
Seeds: tuning 101-105, development 1001-1010, held-out 2001-2010
(per-seed unit, 4 runs per seed). Reuse paper-1 comm geometry caches
(paths O-RU<->UE with gains, delays, departure/arrival angles,
interaction points, model-B losses) READ-ONLY; extend with a new
p2 cache only where needed (e.g. perturbed UE positions).

## Signal model
Uplink SRS from the UE (23 dBm, single antenna) received by both O-RUs
(8x8 UPA, NF 7 dB, kT). Numerology 3 (120 kHz). Bandwidth sweep:
100, 200, 400 MHz (state active subcarriers). SRS repetition every
0.1 s (same epochs as paper 1). Channel per subcarrier and antenna from
the cached multipath with model-B blockage applied. Three timing cases:
(a) synchronized ToA (bound reference), (b) unknown UE clock bias
(TDoA between O-RUs, bias as nuisance parameter), (c) AoA only.

## Bounds (P1-P3)
Position error bound (PEB) per UE and epoch via the equivalent Fisher
information matrix (EFIM) in the position domain, nuisance parameters
(path gains, clock bias) removed by Schur complement:
 - LoS-only information;
 - map-aided: all resolvable paths, with path geometry known from the
   digital twin.
Derivatives of delays/angles w.r.t. UE position: analytic for LoS;
for NLoS use finite differences by re-tracing at UE +/- 1 cm in x and
y (verify against the analytic LoS derivative to 1e-4 relative).
Report: PEB maps along the sidewalk, CDFs, share of time PEB <= 0.1 m
and <= the paper-1 break-even range, split by LoS/blocked state
(10 dB event or not), bandwidth, mount, density, timing case.

## Practical estimator (P4)
Per O-RU: beamspace (2D DFT over the UPA) plus delay estimation with
super-resolution refinement (e.g. ESPRIT or zero-padded FFT with
interpolation) of the dominant path; LoS/NLoS gate; fusion of both
O-RUs by weighted least squares; EKF with a pedestrian motion model.
Tune all estimator parameters on tuning seeds only. Report RMSE,
median, p90, error-to-PEB ratio, and the same splits as the bounds.
Sanity check against the literature (sub-meter regime expected at
moderate bandwidth).

## Closing experiment (P4): link to paper 1
Feed the estimator's UE positions (instead of the 1 m Gaussian error)
into the frozen paper-1 planner with PERFECT blocker tracks, using the
paper-1 code at tag v1.4.1-wcnc2027 read-only (import or a pinned
worktree; never edit it). Paired seed-level comparison vs A5 per
margin. Also with the PEB-limited error (bound-level positioning) as
an upper reference.

## Rules
- Research integrity as in ROADMAP.md: tune on tuning seeds, design and
  debug on development seeds only, freeze code (tag p2-freeze), then a
  single held-out evaluation on 2001-2010. Seed is the statistical
  unit: Wilcoxon signed-rank over 10 seeds, bootstrap CIs over seeds,
  pointwise tests labelled exploratory.
- GPU 1 only, torch on GPU for batched Fisher information, channel
  synthesis and signal processing; up to 4 worker processes for
  tracing; keep peak GPU memory < 60 GB; report runtimes.
- NumPy reference implementations with equality tests for every GPU
  kernel; scripts/test_p2.py (unittest) must pass; determinism checked
  on one job (bitwise or documented tolerance).
- Stage-scoped provenance for every new cache; never restamp caches.
- New paths only: sim/positioning/, scripts/p2_*.py, configs/p2.yaml,
  results/P2/, paper2/. Do not touch paper 1 files or tags.
- Every number for the paper comes from scripts/p2_make_paper.py into
  paper2/numbers2.tex (catalogued with source, config, seeds,
  aggregation in results/P2/numbers_catalog.json).
- Never commit results/ (git-ignored); push the branch `paper2` only.

## Milestones and acceptance criteria
P2-M1 Signal model + PEB: tests pass (LoS derivative check, EFIM vs
  brute-force numeric FIM on toy cases, PEB scales as expected with
  bandwidth and SNR); PEB results on development seeds.
P2-M2 Practical estimator + EKF, tuned on tuning seeds; error-to-PEB
  analysis on development seeds.
P2-M3 Closing experiment with the frozen paper-1 planner on
  development seeds.
P2-M4 Freeze (commit, tag p2-freeze), then the full held-out
  evaluation on 2001-2010, unchanged; report every claim (P1-P4) as
  supported, not supported or mixed, with the statistics.
P2-M5 Figures (single column, vector PDF, >= 8 pt, IEEE ready):
  PEB/RMSE vs bandwidth (LoS vs blocked), PEB map along the sidewalk,
  error CDFs (bound vs estimator), closing-experiment outage vs margin;
  paper2/numbers2.tex; a draft paper2/main.tex (IEEE conference, 6
  pages max, IEEEtran) whose every number is a macro; results/P2/
  report.md summarising all milestones, deviations and open issues.

## Addendum A: realism of the bounds
1. Hardware limits as swept parameters (otherwise the bound is
   unrealistically small at this SNR):
   - inter-O-RU time synchronization error sigma_sync in
     {0, 0.3, 1, 3} ns (Gaussian, per epoch, or constant per run;
     state which), entering ToA/TDoA as a random bias;
   - array calibration error: per-element phase error sigma_phi in
     {0, 2, 5} deg (fixed per O-RU and run), entering AoA as a bias.
   Treat them in the bound as Gaussian priors on the nuisance biases
   (Bayesian EFIM), and in the estimator as actual random draws. Report
   which of bandwidth, SNR, sync or calibration limits the PEB.
2. Blocked LoS realism: when the LoS of a cell has model-B loss >=
   10 dB, evaluate two variants: (a) "geometry kept": the attenuated
   LoS still carries exact delay/angle information (optimistic);
   (b) "biased": the blocked LoS delay/angle carry an unknown bias
   (treated as a nuisance parameter, i.e. its geometric information is
   removed). Report P2 under both; the paper's main claim uses (b),
   with (a) as the optimistic reference.
3. Derivatives: use analytic derivatives (image method for specular
   reflections on the known facades/ground, exact for LoS); keep the
   +/-1 cm finite differences only as a validation test on a subset
   (agreement within 1e-3 relative where the path persists).
