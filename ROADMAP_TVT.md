# ROADMAP_TVT — journal extension (IEEE Transactions on Vehicular Technology)

## Goal
One regular TVT paper that merges paper 1 (value of foresight, tag
v1.5-wcnc2027) and paper 2 (positioning bounds, tag
v1.0-wcnc2027-paper2) and adds a NEW METHOD: a blockage-aware map-aided
multipath tracker in which the ISAC blocker tracks predict the
visibility of every LoS and reflected path and thereby steer data
association. The draft text lives in `paper_tvt/` and is written by the
humans; the agent supplies code, results, figures and numbers.

Scope decision (human): simulation only for now. The O-RAN testbed
(OAI/FlexRIC E2 latency, over-the-air handover with USRPs) is DEFERRED;
the planner keeps the modelled E2 delay of 20 ms and the delay is swept
as a parameter (T6). Do not start any testbed work.

Timeline (proposal, humans adjust): WCNC notification 15 Jan 2027;
TVT submission after that. Feature freeze 4 Jan 2027.

## Research integrity (non-negotiable, as in ROADMAP.md)
- Hypotheses are TESTED, not confirmed. Report null and negative results
  exactly as they are, including if the new tracker does not close the
  gap to the map-aided bound.
- The conference held-out sets (2001-2010, 3001-3010) are CONSUMED and
  must not be used for any journal claim.
- Seeds (`configs/seeds_tvt.yaml`):
  tuning 101-105 (tuning only), development 1001-1010 (design and
  debugging), training 5001-5040 (only for the learned baseline of T5),
  held-out 4001-4010 (NEVER opened before the freeze in T7).
- Never drop runs, seeds or scenarios because results look bad. Never
  change geometry, traffic or impairments to favour a method; scenario
  changes need human approval and a reason that does not come from the
  results.
- Baselines get the same tuning budget and the same information as the
  proposed method. Every gain is attributed by an ablation with identical
  parameters.
- Seed is the statistical unit: exact two-sided Wilcoxon signed-rank over
  10 seeds, bootstrap 95% CIs over seeds; pointwise tests are labelled
  exploratory.
- Provenance: every result and cache records git commit, hash of
  uncommitted changes and config hash; loaders refuse mismatches.

## Hypotheses of the journal paper
- J1 Visibility predicted from ISAC blocker tracks improves multipath
  data association: the proposed tracker has a lower error than the same
  tracker without visibility prediction, especially during blockage and
  in the second before onset.
- J2 The proposed tracker narrows the gap between the paper-2 estimator
  and the map-aided PEB; quantify how much remains and why.
- J3 With the proposed tracker and perfect blocker tracks, the planner
  beats A5 (and CHO) at some margins; with real blocker tracks, report
  what remains.
- J4 The risk-aware planner (CVaR term) improves on the risk-neutral
  planner with the same inputs.
- J5 The conclusions hold under map error, calibration error, traffic
  density and E2 delay sweeps, and in a second deployment.

## Milestones (one at a time; report in results/TVT/<milestone>/report.md, then STOP)

T0 Branch and harmonization
- Branch `tvt` from `main`; merge `paper2` into it (code only; do not
  edit `paper/` or `paper2/`). Report conflicts and resolutions.
- Service model: paper 1 uses the SNR of the best beam, paper 2 the
  total path power with full array gain. Implement both behind one
  config switch, quantify how the headline numbers of both papers change
  on DEVELOPMENT seeds, and STOP: the humans choose one model.
- Create `configs/seeds_tvt.yaml` as above and a guard that refuses the
  held-out seeds before the tag `tvt-freeze` exists.
- Regression: with the chosen service model, reproduce the paper-1 and
  paper-2 pipelines end to end on development seeds.

T1 Physical array model and uncertain-map bound
- Array: element gain errors, phase errors, element-position errors,
  directional element pattern with a back plane (removes the front/back
  ambiguity). Sweep parameters from config.
- PEB with surface offsets d_s ~ N(0, sigma_map^2) per facade as nuisance
  parameters with a Gaussian prior (Bayesian EFIM); analytic VA
  derivatives w.r.t. d_s; finite-difference and brute-force FIM tests.
- Acceptance: tests pass; PEB vs sigma_map in {0, 0.1, 0.25, 0.5, 1} m
  moves from the perfect-map to the LoS-only bound.

T2 Multipath extraction
- Sparse extraction per O-RU and epoch (e.g. OMP on a beam-delay grid
  plus local refinement) returning components with covariances from the
  local Fisher information.
- Acceptance: on fixed geometries, detection rate and false-component
  rate of extracted vs true paths, parameter errors vs CRB; run time.

T3 Visibility prediction
- q_{c,s,t} for LoS and every VA from blocker-track posteriors and the
  predicted UE state (Monte Carlo, K samples), plus a prior for regions
  the radar does not cover.
- Evaluate calibration against ground-truth path blockage (model-B loss
  >= 10 dB): reliability diagram, Brier score, AUC; with perfect tracks
  and with real tracks; split by blocker class.

T4 Blockage-aware map-aided multipath tracker (main method)
- State: UE position, velocity, clock bias, surface offsets. Loopy BP
  for marginal association probabilities (sources incl. clutter),
  PDA-EKF update, map constraints.
- Ablations with identical parameters: (i) no visibility prediction,
  (ii) oracle visibility, (iii) predicted visibility (real tracks);
  sigma_map sweep. Compare with the paper-2 estimator and the PEB;
  windows before onset / during / elsewhere as in paper 2.
- DECISION POINT: report J1/J2 on development seeds and STOP. The humans
  decide whether the narrative is "closes the gap" or "requirements and
  achievable region".

T5 Planner and baselines
- Risk-aware planner (CVaR over K joint samples, candidate plans
  stay/switch-once), fed with the tracker covariance.
- Baselines, same tuning budget on tuning seeds: A3, A5, CHO (prepared
  target, execution condition), sensing trigger in the spirit of Look
  Before Switch, learned blockage predictor in the spirit of RaDaR
  (trained on training seeds 5001-5040 only), paper-1 planner with the
  paper-2 estimator, genie-planner, cost-aware oracle.
- Metrics: outage, handovers/min, ping-pong rate, per blocker class.

T6 Sensitivity, generalization, complexity
- Sweeps: sigma_map, calibration errors, residual radar self-
  interference, traffic density, UE speed, O-RU height, sensing
  overhead, E2 delay (0-100 ms).
- Second deployment (intersection): proposal with geometry and reason,
  STOP for human approval before tracing.
- Blockage model check: model B vs first-order diffraction, if Sionna RT
  2.2 supports it (verify first, report).
- Run time per component (Table "complexity").

T7 Freeze and held-out evaluation
- Commit, tag `tvt-freeze`; then a single evaluation on 4001-4010,
  unchanged code and configs. Report every J-claim as supported, not
  supported or mixed, with statistics.

T8 Figures and numbers
- Vector PDF, IEEE single/double column, >= 8 pt; one script per figure.
- `scripts/tvt_make_paper.py` writes `paper_tvt/numbers_tvt.tex`
  (catalogue with source, config, seeds, aggregation in
  results/TVT/numbers_catalog.json). No number is typed by hand.

## Paths
New code: sim/positioning/ (extend), sim/tvt/, xapp/ (extend, keep
paper-1 behaviour reproducible), scripts/tvt_*.py, configs/tvt*.yaml,
results/TVT/, paper_tvt/ (text by humans). Never edit paper/, paper2/ or
existing tags. Never commit results/.
