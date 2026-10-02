# ROADMAP — goals, milestones, acceptance criteria

Deadline: WoWMoM 2027 regular paper, 1 Dec 2026 23:59 AoE.
Feature freeze: 15 Nov 2026 (after that: experiments, figures, writing only).

## How the agent works with this file
- Work on ONE milestone at a time, in order. The current milestone is the
  first one whose status is not `DONE`.
- At the end of each milestone, write `results/<milestone>/report.md`
  (what was built, how to run it, test output, metrics, open issues,
  deviations from this plan) and STOP for human review.
- Never mark a milestone `DONE` yourself; the human does.
- If a goal turns out infeasible or ambiguous, stop and propose options
  instead of silently changing the plan.

## Research integrity (non-negotiable)
- The hypotheses below are to be TESTED, not confirmed. Report negative
  or null results exactly as they are.
- Seeds: `configs/seeds.yaml` defines disjoint TUNING seeds and
  EVALUATION seeds. Tune thresholds/parameters on tuning seeds only;
  report on evaluation seeds only (>= 10 seeds, mean and 95% CI).
- Never drop runs, seeds or scenarios because results look bad.
- Baselines get the same effort as the proposed method (same tuning
  budget, same information where applicable).

## Research hypotheses (the paper's claims, if they hold)
- H1 Twin-aware clutter removal and ghost handling reduce false alarms and
  ghost tracks versus blind clutter removal, at equal detection probability.
- H2 Sensing-based blockage prediction reduces outage time and throughput
  loss versus reactive beam management and an RSRP-trend predictor.
- H3 The gain survives realistic near-RT RIC loop delays; there is a
  break-even E2 delay beyond which it vanishes (find it).
- H4 There is a sweet spot in the fraction of resources spent on sensing
  that maximises net communication performance.

---

## M0 — Environment (1–4 Oct) — status: TODO
Goal: reproducible GPU environment and a compute budget.
Acceptance:
- `scripts/sanity_check.py --require-gpu`: all PASS, variant `cuda_*`.
- `results/sanity_range_doppler.png`: peak on the expected point.
- Benchmark s/snapshot for 20 and 50 cars, peak GPU memory.
- `results/M0/report.md` incl. an estimate: snapshots/hour and how many
  scenario-seconds can be simulated per day.

## M1 — Scenario, mobility, ground truth, blockage model (5–14 Oct) — TODO
Goal: a configurable, deterministic urban scenario producing per-snapshot
channels and ground truth.
Build:
- `sim/scenes/`: scenario from YAML: Sionna built-in urban scene, O-RU
  positions/heights/arrays (28 GHz, numerology from config), lanes,
  vehicles (`vehicle-multi-sp`) and pedestrians (`human`) with
  trajectories, pedestrian UEs on sidewalks.
- Time loop: every snapshot updates target `position` AND `velocity`
  (velocity is not derived from position), with one batched scene edit.
- Per snapshot, save: sensing paths (`RCSSolver`), background paths
  (`PathSolver`), comm paths per (O-RU, UE), and ground truth: target
  states, and per UE which target blocks which path and when.
- `sim/comm/blockage.py`: 3GPP TR 38.901 blockage model B (knife-edge)
  applied to comm paths using target geometry, because Sionna targets are
  binary absorbers. Comm paths must therefore be computed so that the
  attenuation comes from model B, not from the absorber (design choice to
  be documented in the report).
Acceptance:
- Same config + seed gives bit-identical outputs (test).
- Unit tests for model B against a direct implementation of the formulas;
  plot of loss vs. lateral offset of a blocker is continuous.
- Rendered frames/video of the scenario with targets and UE links.
- Ground truth reports blockage events with start, end, blocker id.

## M2 — Sensing pipeline (15–28 Oct) — TODO
Goal: detections and tracks from the radar channel.
Build (`sim/sensing/`):
- Range-Doppler(-angle) processing with `sionna.phy.isac` (windowing,
  range compensation as configurable steps).
- Clutter removal: (a) blind baseline (e.g., slow-time mean / zero-Doppler
  notch); (b) twin-aware: subtraction of the background predicted by the
  digital twin.
- CFAR detection (2D CA-CFAR, configurable).
- Ghost handling: (a) none; (b) twin-aware, using known building geometry
  (e.g., image-method consistency / map-based rejection or re-mapping).
- Multi-target tracking (Kalman + assignment).
- Monostatic first. Bistatic (O-RU pair) is a stretch goal: implement only
  after human approval at the M2 review.
Metrics: Pd, false-alarm rate, ghost-track rate, position/velocity RMSE
(or OSPA) vs. ground truth.
Acceptance: table of metrics for all (clutter x ghost) variants on
evaluation seeds -> evidence for or against H1.

## M3 — Blockage prediction and xApp logic (29 Oct – 8 Nov) — TODO
Goal: proactive actions from tracks, with the RIC loop modelled.
Build (`xapp/`):
- Predictor: for each UE and horizon T, time-to-blockage and duration
  from tracks and UE/O-RU geometry.
- Actions: proactive beam switch to the best alternative path, handover
  to another O-RU, or no action.
- Interface modelled like E2: report period, loop delay (from M4 or a
  sweep 10 ms – 1 s), control latency; xApp logic independent of the
  simulator (could later run on a real RIC).
- Comm performance per UE over time (SNR, throughput via PHY abstraction,
  outage) including the cost of sensing resources.
- Baselines: reactive beam management (beam-failure detection + recovery
  delay from config), RSRP-trend predictor, oracle.
Acceptance: precision/recall and lead time of predictions; outage time and
throughput vs. baselines -> evidence for H2.

## M4 — O-RAN loop latency (parallel, 25 Oct – 10 Nov) — TODO
Goal: measured, not assumed, E2 loop delays.
Build (`oran_latency/`): OAI gNB + UE in rfsimulator mode with FlexRIC and
a minimal xApp; measure indication -> control round-trip.
Acceptance: latency CDF and summary stats; the values feed M3 sweeps.
Human checkpoint before starting (network/container setup on the server).

## M5 — Experiments and figures (9–20 Nov) — TODO
Goal: final results for H1–H4.
- Sweeps: traffic density, sensing resource fraction, E2 delay, prediction
  horizon, carrier/array config if time allows.
- One command per figure (`scripts/fig_*.py`), from saved results.
- IEEE-ready figures: vector PDF, single-column width, readable fonts.
- A short video of the scenario for the presentation.
Acceptance: every paper figure regenerates from configs with one command.

## M6 — Artefact and submission support (20 Nov – 1 Dec) — TODO
- Clean README for reproduction, pinned environment, license.
- Anonymized mirror check: no names, affiliations, org names, emails or
  identifying paths anywhere (code, notebooks, figures, metadata).
- Writing is done by the humans; the agent supports with numbers/tables.
