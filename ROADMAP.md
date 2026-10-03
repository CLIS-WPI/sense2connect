# ROADMAP — goals, milestones, acceptance criteria

Venue: OPEN — WoWMoM 2027 (regular, 10 pages, double-blind, deadline
1 Dec 2026 23:59 AoE) or WCNC 2027 (6 pages, not blind, EDAS). The
humans decide. Never prepare or support a simultaneous submission of this
work to two venues (double submission is prohibited by IEEE).
Feature freeze: 15 Nov 2026 (after that: experiments, figures, writing only).

## How the agent works with this file
- Work on ONE milestone at a time, in order. The current milestone is the
  first one whose status is not `DONE`.
- At the end of each milestone (or review round), write
  `results/<milestone>/report.md` (what was built, how to run it, test
  output, metrics, open issues, deviations from this plan) and STOP for
  human review.
- Never mark a milestone `DONE` yourself; the human does.
- Never commit or push unless explicitly asked.
- If a goal or constraint turns out infeasible or ambiguous, stop and
  propose options instead of silently changing the plan or the constraint.

## Research integrity (non-negotiable)
- The hypotheses below are to be TESTED, not confirmed. Report negative
  or null results exactly as they are.
- Seeds: `configs/seeds.yaml` defines disjoint TUNING seeds (101-105) and
  EVALUATION seeds (1001-1010). Tune thresholds/parameters on tuning seeds
  only; report on evaluation seeds only (mean and 95% CI).
- Never drop runs, seeds or scenarios because results look bad.
- Never add impairments, change geometry or traffic to make a method look
  better. Changes to the scenario need human approval and a reason that
  does not come from the results.
- Baselines get the same effort as the proposed method (same tuning
  budget, same information where applicable).
- Attribute gains: when variants differ in method AND parameters, run an
  ablation with identical parameters; also compare at equal false-alarm
  budget.
- Provenance: every result and cache entry records git commit, hash of
  uncommitted changes and config hash; loaders refuse mismatches.

## Research hypotheses (the paper's claims, if they hold)
- H1 Digital-twin knowledge improves blocker sensing. Three mechanisms
  are tested and all are reported:
  (a) twin-based static-clutter subtraction vs. blind (expected null:
      static clutter is exactly zero-Doppler in simulation);
  (b) twin-aware ghost handling (image method on known buildings), at
      identical parameters and at equal false-alarm budget;
  (c) map-constrained tracking (vehicles on lanes, pedestrians on
      sidewalks with a free model for crossings) vs. unconstrained EKF.
  Note: H1 was revised from (a)+(b) to (a)+(b)+(c) after the first M2
  results; the paper states this transparently.
- H2 Sensing-based proactive inter-cell handover reduces outage time and
  throughput loss versus reactive 3GPP A3 handover and an RSRP-trend
  predictor, and approaches the oracle cell-selection bound.
- H3 The gain survives realistic near-RT RIC loop delays; there is a
  break-even E2 delay beyond which it vanishes, expected to differ by
  blocker class (bus/truck vs. pedestrian).
- H4 There is a sweet spot in the fraction of resources spent on sensing
  that maximises net communication performance.

## Established findings
M1/M1.5 (tuning seeds; re-run on evaluation seeds in M5):
- Cars never block LoS at O-RU heights 5 m and 8 m; bus/truck and
  pedestrians cause all 10 dB LoS outages.
- Same-cell beam switching cannot recover outages: the blocker also
  shadows the best alternative path in ~100% of events.
- The second O-RU (other cell) is unblocked in ~56-59% of bus/truck
  outages (power ~ -7 dB) but only ~25-29% of pedestrian outages.
- Oracle cell selection cuts the 10 dB outage rate by roughly 50-70%.
M2 (evaluation seeds; PRELIMINARY until the velocity sanity test passes):
- (a) Twin vs. blind clutter subtraction: null (maps differ by ~1e-10).
- (b) Image-method ghost handling: no bus/truck Pd change at identical
  parameters, 14-21% fewer unmatched clusters; at a fixed budget of 4
  FA/CPI, bus/truck track Pd ~0.80 -> ~0.86 (lamppost), ~0.84 -> ~0.90
  (facade); only the image method reaches a 2 FA/CPI budget.
- (c) Map-constrained tracking: bus/truck track Pd 0.82 -> 0.90,
  pedestrian 0.28 -> 0.45; cross-lane velocity RMSE ~2 -> ~0.5 m/s.
- Confirmed track 0.5 s before a 10 dB LoS event (map tracker):
  bus/truck ~82-88%, pedestrian ~54-62%.

---

## M0 — Environment — DONE
Docker image (Sionna 2.2.0, torch 2.9.1+cu128), sanity checks, benchmark.

## M1 — Scenario, mobility, ground truth, blockage model — DONE
Street canyon, lamppost 5 m / facade 8 m, cars/buses/trucks/pedestrians,
3GPP model B on every segment of every comm path, UE-level events
(10 dB, min_gap 0.5 s with nesting), wrap = new identity, theory tests.

## M1.5 — Cross-cell diversity study — DONE
Best available alternative path, second O-RU availability per blocker
class, both-blocked share, oracle cell-selection bound.

## M2 — Sensing pipeline — IN REVIEW
Setup: monostatic radar at oru-0, ideal full duplex (explicit assumption);
TX one element, RX 8x8; numerology 3, one sensing symbol per slot
(PRF 8 kHz), CPI 256 slots (32 ms), overhead 1/14 as a parameter;
1024 subcarriers main (2048 sensitivity only); thermal noise (30 dBm,
NF 7 dB, actual bandwidth); TR 38.901 sensing targets via `RCSSolver`.

Done: 60 s runs with one CPI per 0.1 s snapshot; Doppler-aided EKF;
ghost ablation at identical parameters; clutter null result; event-centric
lead-time table; class gates with sensitivity; false-alarm budgets 2/4/6
with fragment/ghost/other split; map-constrained tracker; channel cache
with provenance; batched GPU angle estimation; 4 parallel trace workers.

Open (required before M2 can be accepted):
1. PRIORITY: velocity sanity test. Bus/truck radial-velocity RMSE ~4 m/s
   is implausible for a Doppler measurement with sigma 0.5 m/s. Unit test
   with one constant-velocity target (approaching, receding, crossing;
   with and without noise; point and 5-point extended target): check
   Doppler sign and scale (Sionna: positive Doppler = approaching,
   f_D = -2 v_r / lambda with v_r > 0 receding), EKF measurement model
   and Jacobian. Fix any bug and rerun M2 evaluation from the caches.
2. ID switches counted once per identity per job; median track lifetime
   per class.
3. Bandwidth consistency: sensing bandwidth must not exceed the comm
   carrier. The comm carrier for M3 is 1024 subcarriers (numerology 3),
   shared by sensing and comm; 2048 is a sensitivity case.
4. Final M2 configuration on evaluation seeds at budgets 2/4/6:
   blind clutter + image-method ghost handling + map-constrained tracker.
   Report track Pd, FA/CPI, velocity RMSE (radial, along/cross lane),
   ID switches, lead-time table per class and mount.
5. Report the three H1 mechanisms and their outcomes plainly, including
   the note that H1 was revised after the first results.
The detector operating point (FA budget) is NOT chosen in M2; it is
chosen in M3 from the false-handover cost.
Stretch (only after human approval): radar at oru-1 too, track fusion.

## M3 — Blockage prediction and xApp logic — TODO
Goal: proactive inter-cell handover from tracks, with the RIC loop modelled.
Inputs: final M2 configuration; comm carrier 1024 subcarriers.
Build (`xapp/`):
- Predictor: per UE, time-to-blockage and duration for the serving and
  the other cell (model B on predicted geometry), horizon H configurable.
- Two O-RUs are two cells. Primary action: xApp-triggered inter-cell
  handover (O-RAN traffic steering) with hold timer against ping-pong.
  Ablation: multi-TRP under one DU (near-zero interruption).
- E2 model: report period, loop delay (sweep 10 ms - 1 s; measured value
  added in M4), handover interruption time from config.
- Comm performance per UE over time: SNR, throughput via PHY
  abstraction, outage, including the cost of sensing resources.
- Operating point: evaluate the detector at FA budgets 2/4/6 and choose
  it on tuning seeds by net outage reduction, accounting for false
  handovers (handovers triggered by a ghost/false track or a wrong
  prediction).
Baselines (equal tuning budget on tuning seeds):
- Reactive 3GPP A3 handover (offset, hysteresis, time-to-trigger).
- RSRP-trend predictor.
- Proactive beam switch on the same cell (expected weak, per M1.5).
- No action. Oracle cell selection (upper bound).
Metrics: prediction precision/recall and lead time; outage time and
throughput; switches per UE-minute; ping-pong rate; false-handover rate;
all split by blocker class. Acceptance: evidence for H2 and H3.

## M4 — O-RAN loop latency — TODO (after M3; human-provided container)
The humans provide a running OAI/FlexRIC container. Connect it to this
project and measure indication -> control round-trip (CDF, summary
stats). Until then, M3 uses the E2-delay sweep. Do not start M4 until the
humans say so.

## M5 — Experiments and figures — TODO
- Sweeps: traffic density, sensing overhead (H4), E2 delay (H3),
  prediction horizon; robustness run with the random components of the
  TR 38.901 models enabled (random_sigma_s, random_phases, random_xpr).
- Re-run the M1/M1.5 characterization on evaluation seeds for the paper.
- One command per figure (`scripts/fig_*.py`), from saved results.
- `paper/numbers.tex`: every number quoted in the paper is generated by a
  script from results (no hand-copied numbers).
- IEEE-ready figures: vector PDF, single-column width, readable fonts.
- A short video of the scenario for the presentation.
Acceptance: every paper figure and number regenerates with one command.

## M6 — Artefact and submission support — TODO
- Clean README for reproduction, pinned environment, license.
- If the venue is double-blind: anonymized mirror; no names, affiliations,
  org names, emails, grant numbers or identifying paths anywhere (code,
  notebooks, figures, PDF metadata).
- Writing is done by the humans; the agent supports with numbers/tables.