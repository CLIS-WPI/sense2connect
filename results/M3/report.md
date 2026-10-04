# M3 rework — proactive handover in a non-degenerate regime

Status: ready for review. Not marked DONE. v1 report kept as `results/M3/report_v1.md`.

## What changed and why

v1 saturated: with 30 dBm, NF 7 dB and the 8x8 array gain the unblocked SNR was ~47 dB (median), so A3, RSRP-trend and the oracle all showed ~0 s/UE-min of SNR outage at 0 dB. **The link-margin sweep below was added after observing this saturation** (it was not part of the original M3 plan). The link budget and the 400 Mbit/s service rate were fixed before the rerun; the numbers below are from the first full rerun with them.

Method changes relative to v1 (all applied to every scheme):

- Model B is evaluated analytically every 10 ms from the blocker poses at that time; no interpolation of blocked power (v1 interpolated linearly between 0.1 s snapshots).
- A handover interrupts the link for tau_HO (rate 0, counted as outage). In v1, tau_HO only delayed the switch.
- Sensing overhead = (1/14) x CPI duty cycle (32 ms / 0.1 s = 0.32) = 0.02286, charged only to the xApp (v1: 1/14).
- The no-action reference is the M1/M1.5 "fixed" cell (largest unblocked power at each step, no cost). RSRP-trend and the xApp run on top of reactive A3 (with that margin's tuned A3 parameters); an xApp handover starts a hold that blocks A3 until the later of hold and the predicted end of the blockage. Reason: UEs walk 60–72 m/min between two O-RUs; at low margins a scheme without distance-driven mobility would be measured on path-loss outage, not blockage.
- Oracle = cell with the larger post-model-B power every 10 ms, no interruption (v1: smaller LoS loss).
- Same-cell beam baseline = analog single beam on the fixed cell: max(LoS-path power, best other path power) after model B (the M1.5 best-alternative bound). With full-array MRT over all paths (used by every other scheme) a same-cell beam switch cannot add power.
- A3 and RSRP-trend decide in the gNB (no E2 delay); the xApp decides at report time + E2 loop delay.
- 10 dB events are recomputed at 10 ms with the M1 rules (min gap 0.5 s, nested in 3 dB) on the fixed cell.
- Each scheme is tuned separately at every margin point (your decision, because the 3GPP reference margin is above 30 dB). This deviates from acceptance criterion 7 (one parameter set evaluated across the sweep).

## Key observations (generated from the tables; H2 conclusions deferred to review)

- RSRP trend + A3 vs A3, outage_req on evaluation seeds (95 % CIs): lower at no margin; overlapping at ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB', '25 dB', '30 dB', '3GPP short-range reference', 'v1 radio (high margin)']; higher at no margin.
- xApp + A3 (A3 held off during xApp hold) vs A3, outage_req on evaluation seeds (95 % CIs): lower at no margin; overlapping at ['0 dB']; higher at ['5 dB', '10 dB', '15 dB', '20 dB', '25 dB', '30 dB', '3GPP short-range reference', 'v1 radio (high margin)'].
- Hybrid (joint) vs A3x, outage_req: lower at no margin; overlapping at ['0 dB', '15 dB', '20 dB']; higher at ['5 dB', '10 dB', '25 dB', '30 dB', '3GPP short-range reference', 'v1 radio (high margin)']. A3 Pareto-front points dominated by a hybrid point, summed over margins: 0 (section 11).
- Genie + A3 (perfect blockage start/end, first-round policy, no sensing overhead) vs A3, outage_req: lower at no margin; overlapping at ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB', '25 dB', '30 dB', '3GPP short-range reference', 'v1 radio (high margin)']; higher at no margin (section 12).
- Interruption-free handover (tau_HO = 0) closes this share of the A3-to-oracle gap: 0 dB 38 %, 5 dB 35 %, 10 dB 42 %, 15 dB 68 %, 20 dB 75 %, 25 dB 87 %, 30 dB 93 %, 3GPP short-range reference 97 %, v1 radio (high margin) 97 % (section 12).
- Policy-independent foresight bound (wrong-cell time inside 10 dB events while the other cell was usable, pooled share of A3 outage_req): 0 dB 0.0 %, 5 dB 1.0 %, 10 dB 5.6 %, 15 dB 3.9 %, 20 dB 4.6 %, 25 dB 5.6 %, 30 dB 8.4 %, 3GPP short-range reference 15.4 %, v1 radio (high margin) 9.1 % (section 13).
- Onset-advance genie (added after the first genie result; no overhead) vs A3: lower at no margin; overlapping at ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB', '25 dB', '30 dB', '3GPP short-range reference', 'v1 radio (high margin)']; higher at no margin (section 13).
- At the 3GPP short-range reference the A3-to-oracle gap is 0.187 s/UE-min; the decomposition in section 11 splits every margin's gap into wrong-cell lag and handover interruption.

## 1. Link budget (fixed before the rerun)

| Item | Value | Source |
|---|---|---|
| gNB (O-RU) transmit power | 33 dBm | 3GPP TR 38.802 V14.2.0, Table A.2.1-1, dense urban, micro layer, above 6 GHz |
| UE noise figure | 13 dB | TR 38.802 V14.2.0, Table A.2.1-1 (baseline; 10 dB = high performance) |
| BS / UE element gain | 0 dBi (isotropic, ray-traced 8x8 array gain included in the channel) | TR 38.802 Table A.2.1-6 gives 8 dBi BS elements; NOT added |
| Cable, body, implementation, shadow-fading losses | 0 dB | TR 38.830 V17.0.0, Table A.3: "reported by companies" (unspecified by 3GPP) |
| Bandwidth | 122.88 MHz (1024 SC x 120 kHz) | M2/M3 carrier |
| Noise | −174 dBm/Hz + 10 log10(B) + NF | |
| Service rate | 400 Mbit/s → SNR_req = 9.32 dB (Shannon, no overhead) | chosen before the rerun |
| Cross-check | TR 38.901 V19.5.0 Table 7.8-1: 35 dBm / 100 MHz at 30 GHz, UT NF 9 dB (calibration table, not used) | |

Reference margin (median best-cell unblocked SNR − SNR_req, tuning jobs) = **35.91 dB**, labelled **3GPP short-range reference**. Lower margins correspond to longer links, unspecified losses, or higher rates. The v1 radio (30 dBm, NF 7 dB) is +3.0 dB from the reference and is kept as the high-margin point (38.91 dB). Each margin point applies a common extra path loss to both cells (0 dB: +35.91 dB, 5 dB: +30.91 dB, 10 dB: +25.91 dB, 15 dB: +20.91 dB, 20 dB: +15.91 dB, 25 dB: +10.91 dB, 30 dB: +5.91 dB, 3GPP short-range reference: +0.00 dB, v1 radio (high margin): -3.00 dB).

Per-UE margin distribution (3GPP reference budget, all 60 jobs; per-UE = median over that UE's 60 s; per-step = every 10 ms):

| Mount | Per-UE median margin p0/p25/p50/p75/p100 [dB] | Per-step margin p5/p25/p50/p75/p95 [dB] | UE-jobs |
|---|---|---|---|
| lamppost | 36.7 / 36.7 / 37.4 / 38.0 / 38.0 | 33.7 / 35.3 / 37.5 / 41.1 / 48.4 | 60 |
| facade | 34.3 / 34.3 / 34.7 / 35.1 / 35.1 | 31.1 / 32.6 / 34.7 / 36.4 / 47.9 | 60 |

The two UEs follow the same configured sidewalk trajectories in every seed, so the per-UE spread is narrow; the per-step spread is the variation along the 80 m walk.

## 2. 10 dB events at 10 ms and onset duration

Model B on every segment of every comm path for every blocker, every 10 ms. Path geometry from a new comm-only trace (`scripts/cache_comm_geometry.py`, PathSolver only, deterministic, 4 workers; sensing caches untouched) held between 0.1 s snapshots with the UE end point at its exact 10 ms position. At the snapshot instants the re-solved model-B powers equal `comm_trace.json` exactly (max relative difference 0 on all 60 jobs).

| Split | Class | Events | Events / UE-min | Actionable share | Onset 10–90 % p10/p50/p90 [s] | Duration p10/p50/p90 [s] |
|---|---|---|---|---|---|---|
| tuning | all | 382 | 9.57 | 0.44 | 0.01 / 0.27 / 1.01 | 0.13 / 0.77 / 2.09 |
| tuning | bus/truck | 213 | 5.33 | 0.58 | 0.11 / 0.52 / 1.05 | 0.58 / 1.09 / 2.27 |
| tuning | pedestrian | 169 | 4.23 | 0.26 | 0.01 / 0.01 / 0.62 | 0.11 / 0.23 / 1.61 |
| tuning | car | 0 | 0.00 | — | — | — |
| evaluation | all | 704 | 8.81 | 0.48 | 0.01 / 0.13 / 0.93 | 0.12 / 0.68 / 2.00 |
| evaluation | bus/truck | 317 | 3.97 | 0.72 | 0.10 / 0.50 / 1.09 | 0.53 / 1.18 / 2.16 |
| evaluation | pedestrian | 387 | 4.85 | 0.29 | 0.01 / 0.01 / 0.37 | 0.11 / 0.23 / 1.58 |
| evaluation | car | 0 | 0.00 | — | — | — |

Onset = time for the LoS loss [dB] on the fixed cell to rise from 10 % to 90 % of the event peak (infinite loss capped at 200 dB). Actionable = the other cell's LoS loss stays below 3 dB for the whole event.

## 3. Tuned parameters per margin (tuning seeds 101–105, 20 jobs, objective: outage time at SNR_req incl. HO interruption; ties → fewer HO)

| Margin | A3 offset / hyst [dB] / TTT [ms] | trend window [s] / drop [dB] / H [s] | xApp budget / H [s] / hold [s] | tuning outage A3 / trend / xApp [s/UE-min] |
|---|---|---|---|---|
| 0 dB | 1 / 1 / 40 | 0.20 / 3 / 2.00 | 2 / 0.5 / 0.2 | 35.417 / 35.391 / 38.816 |
| 5 dB | 1 / 1 / 40 | 0.10 / 3 / 2.00 | 2 / 0.5 / 0.2 | 11.001 / 10.872 / 19.047 |
| 10 dB | 1 / 1 / 40 | 0.10 / 3 / 2.00 | 2 / 0.5 / 0.2 | 4.621 / 4.366 / 7.381 |
| 15 dB | 1 / 1 / 40 | 0.10 / 3 / 2.00 | 2 / 0.5 / 0.2 | 2.807 / 2.703 / 4.193 |
| 20 dB | 1 / 1 / 40 | 0.10 / 5 / 2.00 | 2 / 0.5 / 1.0 | 1.775 / 1.719 / 2.775 |
| 25 dB | 1 / 3 / 40 | 0.10 / 3 / 0.50 | 2 / 0.5 / 1.0 | 0.786 / 0.757 / 1.714 |
| 30 dB | 3 / 3 / 40 | 0.50 / 5 / 0.25 | 2 / 0.5 / 1.0 | 0.313 / 0.328 / 0.647 |
| 3GPP short-range reference | 3 / 3 / 160 | 0.50 / 5 / 0.25 | 2 / 3.0 / 1.0 | 0.197 / 0.243 / 0.432 |
| v1 radio (high margin) | 3 / 3 / 160 | 0.50 / 5 / 0.25 | 2 / 3.0 / 1.0 | 0.179 / 0.241 / 0.407 |

Grid sizes: 24 each (xApp: budget {2,4} x H {0.5,1,2,3} s x hold {0.2,0.5,1.0} s; A3: offset {1,3} x hysteresis {1,2,3} x TTT {40,80,160,320} ms; trend: window {0.1,0.2,0.5} s x drop {3,5} dB x H {0.25,0.5,1,2} s). Full grids are in `metrics.json` (`tuning_grids`).

## 4. Evaluation seeds 1001–1010 (40 jobs, mean ± 95 % CI over jobs)

Outage_req: rate < 400 Mbit/s (incl. sensing overhead for the xApp) or HO interruption. Outage_0: SNR < 0 dB or HO interruption. TP loss = 1 − mean rate / oracle mean rate.

### Margin 0 dB (0.0 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 34.559 ± 2.768 | 7.447 ± 1.331 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 34.559 ± 2.768 | 3.351 ± 0.754 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 37.793 ± 3.079 | 7.977 ± 1.385 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 34.677 ± 2.760 | 3.904 ± 0.790 | 0.92 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | 0.44 ± 0.02 | 0.56 ± 0.02 |
| RSRP trend + A3 | 34.654 ± 2.763 | 3.671 ± 0.775 | 0.71 ± 0.06 | 14.8 ± 1.7 | 0.30 ± 0.04 | 0.72 ± 0.04 | 0.28 ± 0.04 |
| xApp + A3 (A3 held off during xApp hold) | 38.473 ± 2.181 | 7.053 ± 1.175 | 11.33 ± 0.70 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 | 0.53 ± 0.05 |

### Margin 5 dB (5.0 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 10.567 ± 1.483 | 5.903 ± 1.137 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 8.969 ± 1.354 | 1.805 ± 0.453 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 15.739 ± 1.817 | 6.401 ± 1.193 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 9.405 ± 1.376 | 2.186 ± 0.482 | 0.88 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | 0.44 ± 0.02 | 0.56 ± 0.02 |
| RSRP trend + A3 | 9.293 ± 1.372 | 2.106 ± 0.469 | 0.66 ± 0.06 | 14.8 ± 1.7 | 0.30 ± 0.04 | 0.70 ± 0.04 | 0.30 ± 0.04 |
| xApp + A3 (A3 held off during xApp hold) | 18.382 ± 1.620 | 3.588 ± 0.565 | 10.65 ± 0.68 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 | 0.53 ± 0.05 |

### Margin 10 dB (10.0 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 7.187 ± 1.305 | 4.075 ± 0.900 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 2.871 ± 0.676 | 1.012 ± 0.277 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 7.664 ± 1.351 | 4.565 ± 0.960 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 3.426 ± 0.711 | 1.372 ± 0.306 | 0.85 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | 0.44 ± 0.02 | 0.56 ± 0.02 |
| RSRP trend + A3 | 3.196 ± 0.693 | 1.319 ± 0.296 | 0.65 ± 0.06 | 14.8 ± 1.7 | 0.30 ± 0.04 | 0.70 ± 0.04 | 0.30 ± 0.04 |
| xApp + A3 (A3 held off during xApp hold) | 6.247 ± 1.063 | 2.612 ± 0.365 | 9.94 ± 0.62 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 | 0.53 ± 0.05 |

### Margin 15 dB (15.0 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 5.640 ± 1.103 | 1.421 ± 0.424 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 1.672 ± 0.409 | 0.353 ± 0.160 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 6.218 ± 1.167 | 2.092 ± 0.510 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 2.044 ± 0.440 | 0.666 ± 0.181 | 0.80 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | 0.44 ± 0.02 | 0.56 ± 0.02 |
| RSRP trend + A3 | 1.972 ± 0.424 | 0.659 ± 0.176 | 0.63 ± 0.06 | 14.8 ± 1.7 | 0.30 ± 0.04 | 0.70 ± 0.04 | 0.30 ± 0.04 |
| xApp + A3 (A3 held off during xApp hold) | 3.463 ± 0.538 | 1.872 ± 0.232 | 8.55 ± 0.60 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 | 0.53 ± 0.05 |

### Margin 20 dB (20.0 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 3.798 ± 0.847 | 0.564 ± 0.281 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 0.902 ± 0.251 | 0.116 ± 0.092 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 4.300 ± 0.913 | 0.628 ± 0.298 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 1.256 ± 0.282 | 0.401 ± 0.107 | 0.65 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | 0.44 ± 0.02 | 0.56 ± 0.02 |
| RSRP trend + A3 | 1.211 ± 0.271 | 0.414 ± 0.106 | 0.53 ± 0.06 | 14.8 ± 1.7 | 0.30 ± 0.04 | 0.73 ± 0.04 | 0.27 ± 0.04 |
| xApp + A3 (A3 held off during xApp hold) | 2.448 ± 0.396 | 1.095 ± 0.173 | 6.12 ± 0.54 | 39.6 ± 2.4 | 0.37 ± 0.02 | 0.47 ± 0.05 | 0.53 ± 0.05 |

### Margin 25 dB (25.0 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 1.114 ± 0.381 | 0.412 ± 0.225 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 0.300 ± 0.152 | 0.021 ± 0.016 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 1.751 ± 0.458 | 0.453 ± 0.242 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 0.584 ± 0.171 | 0.267 ± 0.041 | 0.55 ± 0.08 | 11.8 ± 1.5 | 0.25 ± 0.04 | 0.45 ± 0.02 | 0.55 ± 0.02 |
| RSRP trend + A3 | 0.587 ± 0.165 | 0.297 ± 0.039 | 0.48 ± 0.06 | 13.7 ± 1.6 | 0.27 ± 0.04 | 0.74 ± 0.04 | 0.26 ± 0.04 |
| xApp + A3 (A3 held off during xApp hold) | 1.493 ± 0.245 | 0.803 ± 0.093 | 4.39 ± 0.28 | 35.3 ± 2.4 | 0.35 ± 0.02 | 0.47 ± 0.05 | 0.53 ± 0.05 |

### Margin 30 dB (30.0 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 0.545 ± 0.275 | 0.289 ± 0.167 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 0.099 ± 0.086 | 0.010 ± 0.009 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 0.602 ± 0.293 | 0.332 ± 0.183 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 0.308 ± 0.104 | 0.196 ± 0.030 | 0.45 ± 0.07 | 9.1 ± 1.3 | 0.19 ± 0.03 | 0.46 ± 0.02 | 0.54 ± 0.02 |
| RSRP trend + A3 | 0.331 ± 0.104 | 0.228 ± 0.032 | 0.41 ± 0.06 | 10.9 ± 1.5 | 0.23 ± 0.03 | 0.74 ± 0.04 | 0.26 ± 0.04 |
| xApp + A3 (A3 held off during xApp hold) | 0.736 ± 0.173 | 0.499 ± 0.063 | 3.52 ± 0.19 | 23.4 ± 2.6 | 0.30 ± 0.02 | 0.47 ± 0.05 | 0.53 ± 0.05 |

### Margin 3GPP short-range reference (35.9 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 0.369 ± 0.203 | 0.105 ± 0.072 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 0.016 ± 0.013 | 0.002 ± 0.003 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 0.406 ± 0.220 | 0.152 ± 0.099 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 0.203 ± 0.042 | 0.156 ± 0.023 | 0.34 ± 0.06 | 7.7 ± 1.1 | 0.13 ± 0.03 | 0.44 ± 0.03 | 0.56 ± 0.03 |
| RSRP trend + A3 | 0.232 ± 0.034 | 0.214 ± 0.029 | 0.38 ± 0.05 | 10.6 ± 1.4 | 0.21 ± 0.03 | 0.72 ± 0.03 | 0.28 ± 0.03 |
| xApp + A3 (A3 held off during xApp hold) | 0.506 ± 0.084 | 0.404 ± 0.039 | 3.31 ± 0.17 | 19.5 ± 1.7 | 0.33 ± 0.02 | 0.39 ± 0.05 | 0.61 ± 0.05 |

### Margin v1 radio (high margin) (38.9 dB)

| Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | HO precision | False-HO rate |
|---|---|---|---|---|---|---|---|
| no action (fixed cell) | 0.296 ± 0.171 | 0.025 ± 0.024 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| oracle cell selection | 0.011 ± 0.010 | 0.000 ± 0.001 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| same-cell analog beam | 0.340 ± 0.187 | 0.045 ± 0.036 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — | — |
| A3 (reactive) | 0.182 ± 0.032 | 0.154 ± 0.022 | 0.31 ± 0.05 | 7.7 ± 1.1 | 0.13 ± 0.03 | 0.44 ± 0.03 | 0.56 ± 0.03 |
| RSRP trend + A3 | 0.226 ± 0.032 | 0.212 ± 0.028 | 0.37 ± 0.05 | 10.6 ± 1.4 | 0.21 ± 0.03 | 0.72 ± 0.03 | 0.28 ± 0.03 |
| xApp + A3 (A3 held off during xApp hold) | 0.456 ± 0.057 | 0.397 ± 0.038 | 3.14 ± 0.13 | 19.5 ± 1.7 | 0.33 ± 0.02 | 0.39 ± 0.05 | 0.61 ± 0.05 |

Oracle and no-action outage_req are identical at margin 0 dB: whenever the fixed cell is below the service rate there, the other cell is below it too, so no cell choice helps (see the decomposition in section 11: (a) is zero).

HO precision/false-HO: share of the scheme's own trigger HOs (A3: all HOs; trend/xApp: trend/xApp-triggered HOs) that leave the fixed cell within [event start − 2 s, event end] of a 10 dB event of that UE.

## 5. Recall per blocker class, all vs actionable events (evaluation seeds)

Recall = share of 10 dB events with a matching HO (any time in [start − 2 s, end]); proactive recall = matching HO before the event start. Interruption per event = time with outage_req in [start − 0.2 s, end].

### Margin 3GPP short-range reference

| Scheme | Class | Events (all / actionable) | Recall all | Recall actionable | Proactive recall all | Proactive recall actionable | Interruption / event all [s] | Interruption / event actionable [s] |
|---|---|---|---|---|---|---|---|---|
| no action (fixed cell) | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.033 ± 0.016 | 0.031 ± 0.022 |
| no action (fixed cell) | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.050 ± 0.028 | 0.045 ± 0.032 |
| no action (fixed cell) | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.016 ± 0.009 | 0.000 ± 0.000 |
| oracle cell selection | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.002 ± 0.001 | 0.000 ± 0.000 |
| oracle cell selection | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.001 ± 0.001 | 0.000 ± 0.000 |
| oracle cell selection | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.002 ± 0.002 | 0.000 ± 0.000 |
| A3 (reactive) | all | 704 / 338 | 0.41 ± 0.05 | 0.54 ± 0.06 | 0.02 ± 0.01 | 0.03 ± 0.02 | 0.013 ± 0.003 | 0.011 ± 0.001 |
| A3 (reactive) | bus/truck | 317 / 227 | 0.65 ± 0.08 | 0.65 ± 0.08 | 0.02 ± 0.02 | 0.02 ± 0.02 | 0.016 ± 0.004 | 0.013 ± 0.002 |
| A3 (reactive) | pedestrian | 387 / 111 | 0.18 ± 0.05 | 0.35 ± 0.10 | 0.02 ± 0.01 | 0.03 ± 0.02 | 0.010 ± 0.004 | 0.007 ± 0.002 |
| RSRP trend + A3 | all | 704 / 338 | 0.53 ± 0.04 | 0.68 ± 0.05 | 0.06 ± 0.02 | 0.07 ± 0.03 | 0.015 ± 0.002 | 0.015 ± 0.001 |
| RSRP trend + A3 | bus/truck | 317 / 227 | 0.72 ± 0.06 | 0.68 ± 0.07 | 0.07 ± 0.03 | 0.06 ± 0.03 | 0.017 ± 0.002 | 0.014 ± 0.002 |
| RSRP trend + A3 | pedestrian | 387 / 111 | 0.35 ± 0.06 | 0.66 ± 0.11 | 0.04 ± 0.02 | 0.07 ± 0.04 | 0.013 ± 0.003 | 0.015 ± 0.002 |
| xApp + A3 (A3 held off during xApp hold) | all | 704 / 338 | 0.44 ± 0.03 | 0.38 ± 0.05 | 0.33 ± 0.03 | 0.34 ± 0.05 | 0.015 ± 0.005 | 0.003 ± 0.001 |
| xApp + A3 (A3 held off during xApp hold) | bus/truck | 317 / 227 | 0.56 ± 0.06 | 0.47 ± 0.07 | 0.45 ± 0.06 | 0.43 ± 0.07 | 0.010 ± 0.007 | 0.002 ± 0.001 |
| xApp + A3 (A3 held off during xApp hold) | pedestrian | 387 / 111 | 0.33 ± 0.05 | 0.23 ± 0.09 | 0.22 ± 0.04 | 0.18 ± 0.08 | 0.018 ± 0.009 | 0.005 ± 0.002 |

### Margin 20 dB

| Scheme | Class | Events (all / actionable) | Recall all | Recall actionable | Proactive recall all | Proactive recall actionable | Interruption / event all [s] | Interruption / event actionable [s] |
|---|---|---|---|---|---|---|---|---|
| no action (fixed cell) | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.414 ± 0.074 | 0.417 ± 0.096 |
| no action (fixed cell) | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.636 ± 0.131 | 0.598 ± 0.117 |
| no action (fixed cell) | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.194 ± 0.049 | 0.047 ± 0.016 |
| oracle cell selection | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.095 ± 0.023 | 0.000 ± 0.000 |
| oracle cell selection | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.063 ± 0.034 | 0.000 ± 0.000 |
| oracle cell selection | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.121 ± 0.032 | 0.000 ± 0.000 |
| A3 (reactive) | all | 704 / 338 | 0.66 ± 0.04 | 0.90 ± 0.03 | 0.15 ± 0.02 | 0.19 ± 0.04 | 0.115 ± 0.023 | 0.021 ± 0.001 |
| A3 (reactive) | bus/truck | 317 / 227 | 0.92 ± 0.04 | 0.92 ± 0.04 | 0.23 ± 0.05 | 0.24 ± 0.06 | 0.088 ± 0.035 | 0.020 ± 0.001 |
| A3 (reactive) | pedestrian | 387 / 111 | 0.41 ± 0.06 | 0.82 ± 0.09 | 0.08 ± 0.03 | 0.08 ± 0.05 | 0.137 ± 0.033 | 0.023 ± 0.004 |
| RSRP trend + A3 | all | 704 / 338 | 0.68 ± 0.03 | 0.91 ± 0.03 | 0.39 ± 0.05 | 0.63 ± 0.05 | 0.112 ± 0.023 | 0.020 ± 0.001 |
| RSRP trend + A3 | bus/truck | 317 / 227 | 0.91 ± 0.03 | 0.91 ± 0.04 | 0.62 ± 0.07 | 0.77 ± 0.06 | 0.084 ± 0.034 | 0.019 ± 0.001 |
| RSRP trend + A3 | pedestrian | 387 / 111 | 0.46 ± 0.06 | 0.88 ± 0.08 | 0.18 ± 0.04 | 0.35 ± 0.10 | 0.135 ± 0.033 | 0.020 ± 0.002 |
| xApp + A3 (A3 held off during xApp hold) | all | 704 / 338 | 0.69 ± 0.03 | 0.70 ± 0.05 | 0.62 ± 0.03 | 0.67 ± 0.06 | 0.160 ± 0.031 | 0.017 ± 0.003 |
| xApp + A3 (A3 held off during xApp hold) | bus/truck | 317 / 227 | 0.84 ± 0.04 | 0.84 ± 0.05 | 0.77 ± 0.05 | 0.83 ± 0.05 | 0.110 ± 0.049 | 0.010 ± 0.002 |
| xApp + A3 (A3 held off during xApp hold) | pedestrian | 387 / 111 | 0.52 ± 0.06 | 0.42 ± 0.11 | 0.46 ± 0.06 | 0.38 ± 0.11 | 0.199 ± 0.042 | 0.028 ± 0.007 |

### Margin 10 dB

| Scheme | Class | Events (all / actionable) | Recall all | Recall actionable | Proactive recall all | Proactive recall actionable | Interruption / event all [s] | Interruption / event actionable [s] |
|---|---|---|---|---|---|---|---|---|
| no action (fixed cell) | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.778 ± 0.106 | 0.813 ± 0.128 |
| no action (fixed cell) | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.162 ± 0.160 | 1.116 ± 0.144 |
| no action (fixed cell) | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.411 ± 0.079 | 0.218 ± 0.057 |
| oracle cell selection | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.304 ± 0.057 | 0.032 ± 0.022 |
| oracle cell selection | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.321 ± 0.095 | 0.052 ± 0.036 |
| oracle cell selection | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.288 ± 0.064 | 0.000 ± 0.000 |
| A3 (reactive) | all | 704 / 338 | 0.66 ± 0.04 | 0.90 ± 0.03 | 0.15 ± 0.02 | 0.19 ± 0.04 | 0.344 ± 0.057 | 0.093 ± 0.025 |
| A3 (reactive) | bus/truck | 317 / 227 | 0.92 ± 0.04 | 0.92 ± 0.04 | 0.23 ± 0.05 | 0.24 ± 0.06 | 0.377 ± 0.093 | 0.112 ± 0.038 |
| A3 (reactive) | pedestrian | 387 / 111 | 0.41 ± 0.06 | 0.82 ± 0.09 | 0.08 ± 0.03 | 0.08 ± 0.05 | 0.311 ± 0.065 | 0.052 ± 0.006 |
| RSRP trend + A3 | all | 704 / 338 | 0.68 ± 0.03 | 0.91 ± 0.03 | 0.39 ± 0.04 | 0.63 ± 0.05 | 0.322 ± 0.058 | 0.056 ± 0.024 |
| RSRP trend + A3 | bus/truck | 317 / 227 | 0.92 ± 0.03 | 0.91 ± 0.04 | 0.62 ± 0.07 | 0.77 ± 0.06 | 0.345 ± 0.095 | 0.074 ± 0.037 |
| RSRP trend + A3 | pedestrian | 387 / 111 | 0.46 ± 0.05 | 0.88 ± 0.08 | 0.18 ± 0.04 | 0.35 ± 0.10 | 0.300 ± 0.065 | 0.024 ± 0.003 |
| xApp + A3 (A3 held off during xApp hold) | all | 704 / 338 | 0.70 ± 0.03 | 0.72 ± 0.06 | 0.64 ± 0.03 | 0.69 ± 0.06 | 0.370 ± 0.063 | 0.097 ± 0.035 |
| xApp + A3 (A3 held off during xApp hold) | bus/truck | 317 / 227 | 0.87 ± 0.04 | 0.87 ± 0.05 | 0.80 ± 0.05 | 0.86 ± 0.05 | 0.387 ± 0.103 | 0.106 ± 0.054 |
| xApp + A3 (A3 held off during xApp hold) | pedestrian | 387 / 111 | 0.52 ± 0.06 | 0.42 ± 0.11 | 0.46 ± 0.06 | 0.38 ± 0.11 | 0.349 ± 0.069 | 0.092 ± 0.049 |

### Margin 0 dB

| Scheme | Class | Events (all / actionable) | Recall all | Recall actionable | Proactive recall all | Proactive recall actionable | Interruption / event all [s] | Interruption / event actionable [s] |
|---|---|---|---|---|---|---|---|---|
| no action (fixed cell) | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.999 ± 0.117 | 1.005 ± 0.140 |
| no action (fixed cell) | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.406 ± 0.168 | 1.337 ± 0.149 |
| no action (fixed cell) | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.613 ± 0.096 | 0.357 ± 0.092 |
| oracle cell selection | all | 704 / 338 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.999 ± 0.117 | 1.005 ± 0.140 |
| oracle cell selection | bus/truck | 317 / 227 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.406 ± 0.168 | 1.337 ± 0.149 |
| oracle cell selection | pedestrian | 387 / 111 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.613 ± 0.096 | 0.357 ± 0.092 |
| A3 (reactive) | all | 704 / 338 | 0.66 ± 0.04 | 0.90 ± 0.03 | 0.15 ± 0.02 | 0.19 ± 0.04 | 1.000 ± 0.117 | 1.005 ± 0.140 |
| A3 (reactive) | bus/truck | 317 / 227 | 0.92 ± 0.04 | 0.92 ± 0.04 | 0.23 ± 0.05 | 0.24 ± 0.06 | 1.407 ± 0.168 | 1.337 ± 0.149 |
| A3 (reactive) | pedestrian | 387 / 111 | 0.41 ± 0.06 | 0.82 ± 0.09 | 0.08 ± 0.03 | 0.08 ± 0.05 | 0.614 ± 0.096 | 0.357 ± 0.092 |
| RSRP trend + A3 | all | 704 / 338 | 0.68 ± 0.03 | 0.91 ± 0.03 | 0.39 ± 0.04 | 0.63 ± 0.05 | 1.000 ± 0.117 | 1.005 ± 0.140 |
| RSRP trend + A3 | bus/truck | 317 / 227 | 0.92 ± 0.03 | 0.91 ± 0.04 | 0.62 ± 0.07 | 0.77 ± 0.06 | 1.407 ± 0.168 | 1.337 ± 0.149 |
| RSRP trend + A3 | pedestrian | 387 / 111 | 0.46 ± 0.05 | 0.88 ± 0.08 | 0.18 ± 0.04 | 0.35 ± 0.10 | 0.613 ± 0.096 | 0.358 ± 0.092 |
| xApp + A3 (A3 held off during xApp hold) | all | 704 / 338 | 0.70 ± 0.03 | 0.72 ± 0.06 | 0.64 ± 0.03 | 0.69 ± 0.06 | 1.018 ± 0.119 | 1.026 ± 0.142 |
| xApp + A3 (A3 held off during xApp hold) | bus/truck | 317 / 227 | 0.87 ± 0.04 | 0.87 ± 0.05 | 0.80 ± 0.05 | 0.86 ± 0.05 | 1.427 ± 0.173 | 1.359 ± 0.154 |
| xApp + A3 (A3 held off during xApp hold) | pedestrian | 387 / 111 | 0.52 ± 0.06 | 0.42 ± 0.11 | 0.46 ± 0.06 | 0.38 ± 0.11 | 0.629 ± 0.094 | 0.377 ± 0.092 |

## 6. H3: xApp outage vs E2 loop delay per blocker class

### Margin: 3GPP short-range reference (xApp parameters tuned at that margin)

| tau_E2 [s] | Outage_req [s/UE-min] | Interruption/event bus/truck [s] | Interruption/event pedestrian [s] | Interruption/event car [s] | Proactive recall bus/truck | Proactive recall pedestrian | Proactive recall car |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.498 ± 0.080 | 0.010 ± 0.007 | 0.018 ± 0.009 | — | 0.45 ± 0.06 | 0.22 ± 0.04 | — |
| 0.02 | 0.506 ± 0.084 | 0.010 ± 0.007 | 0.018 ± 0.009 | — | 0.45 ± 0.06 | 0.22 ± 0.04 | — |
| 0.05 | 0.498 ± 0.081 | 0.010 ± 0.007 | 0.016 ± 0.008 | — | 0.45 ± 0.06 | 0.23 ± 0.04 | — |
| 0.10 | 0.485 ± 0.072 | 0.009 ± 0.005 | 0.014 ± 0.006 | — | 0.45 ± 0.05 | 0.23 ± 0.04 | — |
| 0.20 | 0.498 ± 0.074 | 0.009 ± 0.006 | 0.014 ± 0.006 | — | 0.42 ± 0.06 | 0.24 ± 0.05 | — |
| 0.50 | 0.526 ± 0.076 | 0.009 ± 0.005 | 0.017 ± 0.009 | — | 0.46 ± 0.05 | 0.28 ± 0.04 | — |
| 1.00 | 0.563 ± 0.070 | 0.010 ± 0.004 | 0.016 ± 0.008 | — | 0.51 ± 0.06 | 0.29 ± 0.05 | — |

A3 at the same margin (no E2 dependence): outage_req 0.203 ± 0.042 s/UE-min; interruption/event bus/truck 0.016 ± 0.004 s, pedestrian 0.010 ± 0.004 s.

### Margin: 15 dB (xApp parameters tuned at that margin)

| tau_E2 [s] | Outage_req [s/UE-min] | Interruption/event bus/truck [s] | Interruption/event pedestrian [s] | Interruption/event car [s] | Proactive recall bus/truck | Proactive recall pedestrian | Proactive recall car |
|---|---|---|---|---|---|---|---|
| 0.01 | 3.472 ± 0.537 | 0.193 ± 0.071 | 0.256 ± 0.050 | — | 0.80 ± 0.05 | 0.46 ± 0.06 | — |
| 0.02 | 3.463 ± 0.538 | 0.193 ± 0.071 | 0.256 ± 0.050 | — | 0.80 ± 0.05 | 0.46 ± 0.06 | — |
| 0.05 | 3.583 ± 0.541 | 0.194 ± 0.070 | 0.255 ± 0.051 | — | 0.79 ± 0.05 | 0.46 ± 0.06 | — |
| 0.10 | 3.520 ± 0.526 | 0.194 ± 0.070 | 0.254 ± 0.051 | — | 0.78 ± 0.06 | 0.46 ± 0.06 | — |
| 0.20 | 3.641 ± 0.518 | 0.195 ± 0.070 | 0.253 ± 0.051 | — | 0.75 ± 0.06 | 0.46 ± 0.05 | — |
| 0.50 | 3.789 ± 0.497 | 0.196 ± 0.068 | 0.249 ± 0.050 | — | 0.74 ± 0.06 | 0.43 ± 0.06 | — |
| 1.00 | 3.791 ± 0.496 | 0.193 ± 0.070 | 0.246 ± 0.049 | — | 0.69 ± 0.06 | 0.41 ± 0.05 | — |

A3 at the same margin (no E2 dependence): outage_req 2.044 ± 0.440 s/UE-min; interruption/event bus/truck 0.177 ± 0.064 s, pedestrian 0.223 ± 0.046 s.

## 7. Handover interruption sweep (3GPP short-range reference margin)

| tau_HO [s] | A3 outage_req | trend outage_req | xApp outage_req | A3 HO/min | xApp HO/min |
|---|---|---|---|---|---|
| 0.00 | 0.049 ± 0.026 | 0.020 ± 0.014 | 0.118 ± 0.067 | 7.7 ± 1.1 | 19.5 ± 1.7 |
| 0.02 | 0.203 ± 0.042 | 0.232 ± 0.034 | 0.506 ± 0.084 | 7.7 ± 1.1 | 19.5 ± 1.7 |
| 0.05 | 0.433 ± 0.072 | 0.549 ± 0.074 | 1.053 ± 0.119 | 7.7 ± 1.1 | 19.5 ± 1.7 |
| 0.08 | 0.663 ± 0.103 | 0.864 ± 0.114 | 1.554 ± 0.155 | 7.7 ± 1.1 | 19.5 ± 1.7 |

## 8. GPU efficiency (GPU 1 only)

Before (v1 pipeline, `scripts/profile_m3.py`, one tuning seed x 4 jobs, extrapolated to the full v1 run):

| Stage | Wall |
|---|---|
| simulate none | 1.13 s / job |
| simulate oracle | 1.14 s / job |
| simulate beam | 1.14 s / job |
| simulate a3 | 1.14 s / job |
| simulate trend | 1.21 s / job |
| simulate xapp | 8.18 s / job |
| load_all_jobs_s | 0.7 min |
| tune_s | 59.5 min |
| evaluate_s | 9.3 min |
| delay_sweep_s | 38.2 min |
| ho_sweep_s | 21.8 min |
| total_s | 129.5 min |
| GPU 1 SM during profiling | mean 0 %, max 0 % (61 samples) — v1 M3 is CPU-only |

After (this run):

| Stage | Wall [s] | Peak torch GPU memory [GB] | SM mean / max [%] (dmon samples) |
|---|---|---|---|
| comm geometry trace (one-off, 60 jobs, 4 workers) | 580 | 3.9 (device FB, all workers) | 17 / 32 (288) |
| build | 83.7 | 4.39 | 1 / 22 (82) |
| budget | 0.0 | 0.00 | 0 / 0 (1) |
| snr_all_margins | 0.1 | 0.12 | 0 / 0 (1) |
| tune | 46.6 | 0.00 | 0 / 0 (47) |
| evaluate | 10.2 | 0.00 | 0 / 0 (11) |
| sweeps | 8.7 | 0.00 | 0 / 0 (10) |

Full v1 run ≈ 130 min (CPU, 6 schemes, smaller grids, no margin sweep); this run: 2.5 min for 9 margins x (3 tuned schemes x 24-point grids) + evaluation + sweeps.

Why the SM target (>50 %) is not reached:

- Model B and the predictions are the only dense kernels. They run as one batched float64 pass per job (T x U x C x P x S x B ≈ 6000 x 2 x 2 x 8 x 3 x 32 screens, chunked) and finish in well under a second per job; the build stage is dominated by the CPU part (analytic poses every 10 ms via `states_at`, map-tracker replay from the M2 detections) in 4 worker processes. The GPU is busy only in short bursts between CPU jobs.
- The handover state machines are 6000 dependent 10 ms steps over 10^2–10^3 lanes; each step is ~30 small vector operations. On the GPU that is kernel-launch bound; NumPy on the CPU is as fast, so they run on the CPU (the plan allowed this).
- The comm-only re-trace is limited by Dr.Jit kernel compilation and scene edits (as measured in M2), not by GPU compute.

## 9. Tests and determinism

`python scripts/test_m3.py` (in the container):

```
test_a3_needs_ttt (__main__.BaselineTest.test_a3_needs_ttt) ... ok
test_l3_and_trend (__main__.BaselineTest.test_l3_and_trend) ... ok
test_service_snr_and_margin_shift (__main__.LinkBudgetTest.test_service_snr_and_margin_shift) ... ok
test_sensing_overhead_is_one_fourteenth (__main__.PhyTest.test_sensing_overhead_is_one_fourteenth) ... ok
test_shannon_caps_at_nr_mcs27 (__main__.PhyTest.test_shannon_caps_at_nr_mcs27) ... ok
test_snr_uses_tx_and_noise_figure (__main__.PhyTest.test_snr_uses_tx_and_noise_figure) ... ok
test_steering_is_unit_norm (__main__.PhyTest.test_steering_is_unit_norm) ... ok
test_handover_needs_the_actionable_window (__main__.PolicyTest.test_handover_needs_the_actionable_window) ... ok
test_return_after_hold (__main__.PolicyTest.test_return_after_hold) ... ok
test_blockage_window (__main__.PredictTest.test_blockage_window) ... ok
test_lane_track_stays_on_the_line (__main__.PredictTest.test_lane_track_stays_on_the_line) ... ok
test_vectorised_matches_scalar (__main__.SchemeEqualityTest.test_vectorised_matches_scalar) ... ok
test_gpu_matches_numpy (__main__.TimelineEqualityTest.test_gpu_matches_numpy) ... ok
test_prediction_gpu_matches_numpy (__main__.TimelineEqualityTest.test_prediction_gpu_matches_numpy) ... ok
test_snapshot_steps_reproduce_the_trace (__main__.TimelineEqualityTest.test_snapshot_steps_reproduce_the_trace) ... ok
test_interpolation_and_loss_lerp (__main__.TimelineTest.test_interpolation_and_loss_lerp) ... ok

----------------------------------------------------------------------
Ran 16 tests in 3.451s

OK
```

Equality tests (tol 1e-5): batched GPU model B vs NumPy `path_blocker_loss` (powers, LoS loss, dominant blocker); GPU prediction vs `xapp.predict.los_loss_db`; SNR for all margins on GPU vs NumPy; rate vs `sim.comm.phy.shannon_bps`; vectorised schemes vs the scalar reference (identical handover lists and outage masks). The GPU timeline at the snapshot instants matches the traced powers to 1e-4 relative (float32 device UE position in the trace vs float64 analytic position here; ≤ 2.5e-5 observed).

Determinism: two full runs in separate processes, the second rebuilding every timeline from scratch: 60 m3_timeline.npz caches bitwise identical; metrics.json (excluding stage timings) identical.

## 10. Deviations, findings to note, open issues

- Margin sweep added after observing saturation in v1 (stated above). Per-margin tuning instead of one fixed parameter set (your decision).
- Comm geometry cache uses a scoped provenance (content hash of the producing modules + configs) instead of the whole-tree dirty hash, so later xApp edits do not invalidate a 10-minute trace; `m3_timeline` caches likewise hash their own sources and parameters. M2 detections are replayed as in v1; their (older) provenance is recorded, not enforced.
- xApp hold: in the smoke run (seed 101 and evaluation seed 1001) the hold could expire before the predicted blockage, letting A3 hand back at once. The hold now lasts until the later of hold and the predicted end of the blockage window. This was a policy logic fix, decided before any full evaluation; it changed the smoke results very little because most xApp triggers are false alarms (next item). Reviewed and kept: any leak from seed 1001 could only favour the xApp, which still loses to A3, so the conclusion is conservative.
- The xApp predictor fires often on blockages that do not happen (precision table, section 4). Candidate causes, not changed here: class-agnostic sizes (free tracks get a 12 m bus box), ghost/false tracks, along-lane velocity error (v1: p90 ~6 m/s). Changing the predictor is a method change and needs your decision.
- Same-cell beam baseline redefined as an analog single-beam bound (MRT over all paths already includes every surviving path).
- Events and actionability are defined on the fixed (largest unblocked power) cell; oracle and fixed are cost-free references.
- 3GPP gives no numbers for cable/body/implementation/shadow-fading losses in the FR2 template; they are 0 dB in the reference and covered by the margin sweep.

## 11. Follow-up after review: hybrid A3 + xApp, Pareto fronts, outage decomposition

**The hybrid scheme was added after the first M3 rework results**, at the reviewer's request, because in O-RAN an xApp complements the RAN's native mobility instead of replacing it. Hybrid = A3 always active + xApp proactive handovers on top: the xApp only advances a handover when it predicts a blockage of the serving cell (other cell predicted clear, other cell's filtered SNR > serving − 10 dB); it sets no hold, so A3 handles everything else, including returns, and may also hand straight back.

Naming: the scheme reported above as "xApp + A3 (A3 held off during xApp hold)" is the first-round xApp and is kept unchanged. It already ran on top of A3, but it suspended A3 until the later of its hold and the predicted end of the blockage. A truly xApp-alone scheme (no A3) is the v1 design in `report_v1.md`.

Grids: A3x = offset {1,3} dB x hysteresis {0,1,2,3,5} dB x TTT {40,80,160,320,640} ms (50 points). Hybrid = the same A3 grid x xApp budget {2,4} x H {0.5,1,2,3} s (400 points). "Hybrid (joint)" is tuned over all 400 points; "hybrid (A3 fixed)" keeps the A3x-tuned A3 parameters and tunes only the 8 xApp points (same A3 as the A3x baseline, so any difference is the xApp's). The joint hybrid has an 8x larger tuning grid than A3x; the Pareto fronts below compare whole grids rather than single tuned points. Tuning: tuning seeds, per margin, same objective. Pareto points: every grid point on the evaluation seeds (descriptive; nothing is selected from them).

### Tuned parameters (tuning seeds)

| Margin | A3x offset/hyst/TTT | Hybrid joint offset/hyst/TTT, budget, H | Hybrid A3-fixed budget, H | Tuning outage A3x / hybrid joint / hybrid A3-fixed |
|---|---|---|---|---|
| 0 dB | 1/0/40 | 1/1/40, 2, 0.5 | 2, 0.5 | 35.411 / 37.646 / 37.648 |
| 5 dB | 1/0/40 | 1/2/40, 2, 0.5 | 2, 0.5 | 10.964 / 15.530 / 15.872 |
| 10 dB | 1/0/40 | 3/5/40, 4, 0.5 | 2, 0.5 | 4.568 / 6.562 / 8.312 |
| 15 dB | 1/0/40 | 3/5/40, 2, 0.5 | 2, 0.5 | 2.798 / 3.372 / 6.333 |
| 20 dB | 1/1/40 | 3/5/40, 2, 0.5 | 2, 0.5 | 1.775 / 2.330 / 5.065 |
| 25 dB | 1/3/40 | 3/5/640, 2, 0.5 | 2, 0.5 | 0.786 / 1.376 / 3.563 |
| 30 dB | 3/5/40 | 3/5/640, 2, 0.5 | 2, 0.5 | 0.267 / 0.463 / 0.799 |
| 3GPP short-range reference | 3/5/80 | 3/5/640, 2, 0.5 | 2, 0.5 | 0.147 / 0.282 / 0.617 |
| v1 radio (high margin) | 3/5/160 | 3/5/640, 2, 0.5 | 2, 0.5 | 0.129 / 0.270 / 0.452 |

### Evaluation seeds (mean ± 95 % CI over 40 jobs)

| Margin | Scheme | Outage_req [s/UE-min] | Outage_0 [s/UE-min] | TP loss vs oracle [%] | HO / UE-min | Ping-pong | xApp-HO precision |
|---|---|---|---|---|---|---|---|
| 0 dB | oracle | 34.559 ± 2.768 | 3.351 ± 0.754 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 0 dB | A3 (24-pt grid) | 34.677 ± 2.760 | 3.904 ± 0.790 | 0.92 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | — |
| 0 dB | A3x (50-pt grid) | 34.669 ± 2.761 | 3.855 ± 0.786 | 0.74 ± 0.07 | 14.6 ± 1.7 | 0.31 ± 0.03 | — |
| 0 dB | xApp + A3, A3 held off | 38.473 ± 2.181 | 7.053 ± 1.175 | 11.33 ± 0.70 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 |
| 0 dB | hybrid (joint) | 37.178 ± 2.481 | 7.631 ± 0.893 | 8.88 ± 0.46 | 196.0 ± 14.2 | 0.94 ± 0.00 | 0.48 ± 0.05 |
| 0 dB | hybrid (A3 fixed) | 37.179 ± 2.482 | 7.797 ± 0.889 | 8.94 ± 0.47 | 207.0 ± 14.6 | 0.94 ± 0.00 | 0.49 ± 0.05 |
| 5 dB | oracle | 8.969 ± 1.354 | 1.805 ± 0.453 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 5 dB | A3 (24-pt grid) | 9.405 ± 1.376 | 2.186 ± 0.482 | 0.88 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | — |
| 5 dB | A3x (50-pt grid) | 9.371 ± 1.376 | 2.187 ± 0.481 | 0.73 ± 0.08 | 14.6 ± 1.7 | 0.31 ± 0.03 | — |
| 5 dB | xApp + A3, A3 held off | 18.382 ± 1.620 | 3.588 ± 0.565 | 10.65 ± 0.68 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 |
| 5 dB | hybrid (joint) | 14.453 ± 1.471 | 5.477 ± 0.523 | 9.14 ± 0.49 | 184.1 ± 14.0 | 0.93 ± 0.00 | 0.48 ± 0.05 |
| 5 dB | hybrid (A3 fixed) | 14.675 ± 1.455 | 5.887 ± 0.516 | 9.32 ± 0.51 | 207.0 ± 14.6 | 0.94 ± 0.00 | 0.49 ± 0.05 |
| 10 dB | oracle | 2.871 ± 0.676 | 1.012 ± 0.277 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 10 dB | A3 (24-pt grid) | 3.426 ± 0.711 | 1.372 ± 0.306 | 0.85 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | — |
| 10 dB | A3x (50-pt grid) | 3.362 ± 0.706 | 1.376 ± 0.305 | 0.73 ± 0.08 | 14.6 ± 1.7 | 0.31 ± 0.03 | — |
| 10 dB | xApp + A3, A3 held off | 6.247 ± 1.063 | 2.612 ± 0.365 | 9.94 ± 0.62 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 |
| 10 dB | hybrid (joint) | 5.319 ± 1.059 | 1.996 ± 0.382 | 12.37 ± 0.60 | 36.2 ± 6.2 | 0.63 ± 0.05 | 0.43 ± 0.06 |
| 10 dB | hybrid (A3 fixed) | 7.390 ± 0.817 | 5.143 ± 0.409 | 9.66 ± 0.53 | 207.0 ± 14.6 | 0.94 ± 0.00 | 0.49 ± 0.05 |
| 15 dB | oracle | 1.672 ± 0.409 | 0.353 ± 0.160 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 15 dB | A3 (24-pt grid) | 2.044 ± 0.440 | 0.666 ± 0.181 | 0.80 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | — |
| 15 dB | A3x (50-pt grid) | 2.048 ± 0.437 | 0.681 ± 0.181 | 0.72 ± 0.08 | 14.6 ± 1.7 | 0.31 ± 0.03 | — |
| 15 dB | xApp + A3, A3 held off | 3.463 ± 0.538 | 1.872 ± 0.232 | 8.55 ± 0.60 | 60.4 ± 4.0 | 0.80 ± 0.01 | 0.47 ± 0.05 |
| 15 dB | hybrid (joint) | 2.581 ± 0.531 | 1.320 ± 0.274 | 10.14 ± 0.59 | 30.9 ± 6.4 | 0.60 ± 0.06 | 0.48 ± 0.08 |
| 15 dB | hybrid (A3 fixed) | 5.805 ± 0.496 | 4.516 ± 0.337 | 9.77 ± 0.55 | 207.0 ± 14.6 | 0.94 ± 0.00 | 0.49 ± 0.05 |
| 20 dB | oracle | 0.902 ± 0.251 | 0.116 ± 0.092 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 20 dB | A3 (24-pt grid) | 1.256 ± 0.282 | 0.401 ± 0.107 | 0.65 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | — |
| 20 dB | A3x (50-pt grid) | 1.256 ± 0.282 | 0.401 ± 0.107 | 0.65 ± 0.08 | 13.6 ± 1.6 | 0.30 ± 0.04 | — |
| 20 dB | xApp + A3, A3 held off | 2.448 ± 0.396 | 1.095 ± 0.173 | 6.12 ± 0.54 | 39.6 ± 2.4 | 0.37 ± 0.02 | 0.47 ± 0.05 |
| 20 dB | hybrid (joint) | 1.840 ± 0.381 | 0.827 ± 0.209 | 4.92 ± 0.44 | 30.9 ± 6.4 | 0.60 ± 0.06 | 0.48 ± 0.08 |
| 20 dB | hybrid (A3 fixed) | 4.871 ± 0.398 | 4.047 ± 0.306 | 8.97 ± 0.51 | 196.0 ± 14.2 | 0.94 ± 0.00 | 0.48 ± 0.05 |
| 25 dB | oracle | 0.300 ± 0.152 | 0.021 ± 0.016 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 25 dB | A3 (24-pt grid) | 0.584 ± 0.171 | 0.267 ± 0.041 | 0.55 ± 0.08 | 11.8 ± 1.5 | 0.25 ± 0.04 | — |
| 25 dB | A3x (50-pt grid) | 0.584 ± 0.171 | 0.267 ± 0.041 | 0.55 ± 0.08 | 11.8 ± 1.5 | 0.25 ± 0.04 | — |
| 25 dB | xApp + A3, A3 held off | 1.493 ± 0.245 | 0.803 ± 0.093 | 4.39 ± 0.28 | 35.3 ± 2.4 | 0.35 ± 0.02 | 0.47 ± 0.05 |
| 25 dB | hybrid (joint) | 1.091 ± 0.249 | 0.360 ± 0.094 | 3.88 ± 0.30 | 12.1 ± 2.0 | 0.33 ± 0.05 | 0.46 ± 0.08 |
| 25 dB | hybrid (A3 fixed) | 3.686 ± 0.324 | 3.329 ± 0.280 | 7.78 ± 0.46 | 164.9 ± 13.6 | 0.92 ± 0.01 | 0.49 ± 0.05 |
| 30 dB | oracle | 0.099 ± 0.086 | 0.010 ± 0.009 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 30 dB | A3 (24-pt grid) | 0.308 ± 0.104 | 0.196 ± 0.030 | 0.45 ± 0.07 | 9.1 ± 1.3 | 0.19 ± 0.03 | — |
| 30 dB | A3x (50-pt grid) | 0.264 ± 0.108 | 0.137 ± 0.023 | 0.44 ± 0.08 | 6.1 ± 0.8 | 0.13 ± 0.03 | — |
| 30 dB | xApp + A3, A3 held off | 0.736 ± 0.173 | 0.499 ± 0.063 | 3.52 ± 0.19 | 23.4 ± 2.6 | 0.30 ± 0.02 | 0.47 ± 0.05 |
| 30 dB | hybrid (joint) | 0.554 ± 0.172 | 0.282 ± 0.053 | 3.32 ± 0.19 | 12.1 ± 2.0 | 0.33 ± 0.05 | 0.46 ± 0.08 |
| 30 dB | hybrid (A3 fixed) | 0.803 ± 0.204 | 0.635 ± 0.132 | 3.56 ± 0.24 | 30.9 ± 6.4 | 0.60 ± 0.06 | 0.48 ± 0.08 |
| 3GPP short-range reference | oracle | 0.016 ± 0.013 | 0.002 ± 0.003 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| 3GPP short-range reference | A3 (24-pt grid) | 0.203 ± 0.042 | 0.156 ± 0.023 | 0.34 ± 0.06 | 7.7 ± 1.1 | 0.13 ± 0.03 | — |
| 3GPP short-range reference | A3x (50-pt grid) | 0.159 ± 0.036 | 0.118 ± 0.017 | 0.29 ± 0.05 | 5.8 ± 0.8 | 0.12 ± 0.02 | — |
| 3GPP short-range reference | xApp + A3, A3 held off | 0.506 ± 0.084 | 0.404 ± 0.039 | 3.31 ± 0.17 | 19.5 ± 1.7 | 0.33 ± 0.02 | 0.39 ± 0.05 |
| 3GPP short-range reference | hybrid (joint) | 0.328 ± 0.075 | 0.251 ± 0.043 | 2.96 ± 0.13 | 12.1 ± 2.0 | 0.33 ± 0.05 | 0.46 ± 0.08 |
| 3GPP short-range reference | hybrid (A3 fixed) | 0.623 ± 0.134 | 0.587 ± 0.122 | 3.40 ± 0.22 | 29.2 ± 6.1 | 0.58 ± 0.06 | 0.48 ± 0.08 |
| v1 radio (high margin) | oracle | 0.011 ± 0.010 | 0.000 ± 0.001 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | — |
| v1 radio (high margin) | A3 (24-pt grid) | 0.182 ± 0.032 | 0.154 ± 0.022 | 0.31 ± 0.05 | 7.7 ± 1.1 | 0.13 ± 0.03 | — |
| v1 radio (high margin) | A3x (50-pt grid) | 0.133 ± 0.026 | 0.102 ± 0.016 | 0.24 ± 0.05 | 5.0 ± 0.8 | 0.07 ± 0.03 | — |
| v1 radio (high margin) | xApp + A3, A3 held off | 0.456 ± 0.057 | 0.397 ± 0.038 | 3.14 ± 0.13 | 19.5 ± 1.7 | 0.33 ± 0.02 | 0.39 ± 0.05 |
| v1 radio (high margin) | hybrid (joint) | 0.288 ± 0.055 | 0.247 ± 0.042 | 2.84 ± 0.11 | 12.1 ± 2.0 | 0.33 ± 0.05 | 0.46 ± 0.08 |
| v1 radio (high margin) | hybrid (A3 fixed) | 0.456 ± 0.085 | 0.427 ± 0.079 | 3.08 ± 0.15 | 21.3 ± 3.9 | 0.51 ± 0.05 | 0.48 ± 0.08 |

### Pareto fronts: does hybrid dominate A3?

Fronts over all grid points (evaluation seeds, means over jobs). Plots: `results/M3/pareto/pareto_<i>_<margin>.png|pdf` (left: outage vs HO/UE-min, right: outage vs ping-pong; circles = tuned points).

| Margin | vs HO/UE-min: A3-front points dominated by hybrid | hybrid-front points dominated by A3 | HO range where hybrid front lower | HO range where hybrid front higher | vs ping-pong: A3-front dominated | ping-pong range where hybrid lower |
|---|---|---|---|---|---|---|
| 0 dB | 0/29 | 33/33 | — (0/44 levels) | 3.25–196.03 (44/44) | 0/26 | — (0/42) |
| 5 dB | 0/27 | 31/31 | — (0/37 levels) | 3.25–184.08 (37/37) | 0/29 | — (0/38) |
| 10 dB | 0/16 | 8/8 | — (0/20 levels) | 3.25–36.20 (20/20) | 0/18 | — (0/22) |
| 15 dB | 0/13 | 6/6 | — (0/16 levels) | 3.25–30.93 (16/16) | 0/13 | — (0/17) |
| 20 dB | 0/13 | 4/4 | — (0/14 levels) | 3.25–30.93 (14/14) | 0/13 | — (0/15) |
| 25 dB | 0/7 | 2/2 | — (0/8 levels) | 3.25–15.69 (8/8) | 0/9 | — (0/9) |
| 30 dB | 0/5 | 1/1 | — (0/6 levels) | 3.25–12.07 (6/6) | 0/5 | — (0/6) |
| 3GPP short-range reference | 0/4 | 1/1 | — (0/5 levels) | 3.25–12.07 (5/5) | 0/4 | — (0/5) |
| v1 radio (high margin) | 0/2 | 1/1 | — (0/3 levels) | 3.25–12.07 (3/3) | 0/2 | — (0/3) |

Front comparison: at each x level (HO rate or ping-pong of any front point) the best outage reachable with x' ≤ x is compared; differences are means without a CI, so small gaps between fronts are not significant (compare with the CIs in the tables).

### Headroom decomposition: where A3 loses against the oracle

Every 10 ms step in outage (rate < SNR_req rate or interruption) is assigned to exactly one class: (c) both cells below the service rate (unrecoverable by any cell choice; equals the oracle outage), else (b) handover interruption, else (a) serving cell below the rate while the other cell was usable (detection, L3 filter and TTT lag). (a) is split into steps inside 10 dB events of that UE (blockage onset) and outside (path-loss driven cell changes). Residual vs oracle = (a) + (b). A proactive scheme can at best remove (a) and must not add (b). Units: s/UE-min, mean ± 95 % CI over evaluation jobs. For the xApp and hybrid rows the service-rate test includes their sensing overhead, so their (c) is slightly above the oracle outage; that difference is the cost of the sensing resources.

| Margin | Scheme | (a) wrong cell | (a) in 10 dB events | (a) outside events | (b) interruption | (c) both unusable | oracle outage |
|---|---|---|---|---|---|---|---|
| 0 dB | a3 (main, 24-pt grid) | 0.073 ± 0.011 | 0.004 ± 0.003 | 0.069 ± 0.010 | 0.045 ± 0.007 | 34.559 ± 2.768 | 34.559 ± 2.768 |
| 0 dB | a3x (50-pt grid) | 0.067 ± 0.009 | 0.004 ± 0.003 | 0.063 ± 0.009 | 0.043 ± 0.007 | 34.559 ± 2.768 | 34.559 ± 2.768 |
| 0 dB | xapp (main, A3 held off) | 2.205 ± 0.619 | 0.013 ± 0.012 | 2.193 ± 0.614 | 0.221 ± 0.058 | 36.047 ± 2.771 | 34.559 ± 2.768 |
| 0 dB | hybrid (joint) | 0.404 ± 0.113 | 0.003 ± 0.003 | 0.401 ± 0.112 | 0.727 ± 0.217 | 36.047 ± 2.771 | 34.559 ± 2.768 |
| 0 dB | hybrid (A3 fixed) | 0.403 ± 0.112 | 0.003 ± 0.003 | 0.400 ± 0.111 | 0.729 ± 0.217 | 36.047 ± 2.771 | 34.559 ± 2.768 |
| 5 dB | a3 (main, 24-pt grid) | 0.284 ± 0.033 | 0.091 ± 0.013 | 0.193 ± 0.024 | 0.152 ± 0.015 | 8.969 ± 1.354 | 8.969 ± 1.354 |
| 5 dB | a3x (50-pt grid) | 0.242 ± 0.029 | 0.082 ± 0.013 | 0.160 ± 0.020 | 0.160 ± 0.015 | 8.969 ± 1.354 | 8.969 ± 1.354 |
| 5 dB | xapp (main, A3 held off) | 7.476 ± 0.616 | 0.140 ± 0.041 | 7.336 ± 0.610 | 0.952 ± 0.060 | 9.955 ± 1.421 | 8.969 ± 1.354 |
| 5 dB | hybrid (joint) | 1.543 ± 0.125 | 0.081 ± 0.016 | 1.462 ± 0.124 | 2.955 ± 0.221 | 9.955 ± 1.421 | 8.969 ± 1.354 |
| 5 dB | hybrid (A3 fixed) | 1.450 ± 0.116 | 0.073 ± 0.013 | 1.378 ± 0.114 | 3.270 ± 0.240 | 9.955 ± 1.421 | 8.969 ± 1.354 |
| 10 dB | a3 (main, 24-pt grid) | 0.322 ± 0.047 | 0.193 ± 0.031 | 0.129 ± 0.023 | 0.234 ± 0.024 | 2.871 ± 0.676 | 2.871 ± 0.676 |
| 10 dB | a3x (50-pt grid) | 0.246 ± 0.036 | 0.139 ± 0.022 | 0.107 ± 0.019 | 0.246 ± 0.026 | 2.871 ± 0.676 | 2.871 ± 0.676 |
| 10 dB | xapp (main, A3 held off) | 2.091 ± 0.518 | 0.239 ± 0.056 | 1.851 ± 0.499 | 1.126 ± 0.077 | 3.030 ± 0.703 | 2.871 ± 0.676 |
| 10 dB | hybrid (joint) | 1.628 ± 0.496 | 0.172 ± 0.038 | 1.456 ± 0.480 | 0.661 ± 0.116 | 3.030 ± 0.703 | 2.871 ± 0.676 |
| 10 dB | hybrid (A3 fixed) | 0.490 ± 0.102 | 0.123 ± 0.019 | 0.368 ± 0.093 | 3.869 ± 0.285 | 3.030 ± 0.703 | 2.871 ± 0.676 |
| 15 dB | a3 (main, 24-pt grid) | 0.120 ± 0.025 | 0.081 ± 0.017 | 0.039 ± 0.012 | 0.252 ± 0.027 | 1.672 ± 0.409 | 1.672 ± 0.409 |
| 15 dB | a3x (50-pt grid) | 0.108 ± 0.022 | 0.070 ± 0.015 | 0.038 ± 0.011 | 0.268 ± 0.029 | 1.672 ± 0.409 | 1.672 ± 0.409 |
| 15 dB | xapp (main, A3 held off) | 0.589 ± 0.154 | 0.234 ± 0.068 | 0.355 ± 0.114 | 1.151 ± 0.079 | 1.722 ± 0.424 | 1.672 ± 0.409 |
| 15 dB | hybrid (joint) | 0.274 ± 0.062 | 0.192 ± 0.049 | 0.082 ± 0.022 | 0.584 ± 0.122 | 1.722 ± 0.424 | 1.672 ± 0.409 |
| 15 dB | hybrid (A3 fixed) | 0.136 ± 0.029 | 0.084 ± 0.018 | 0.053 ± 0.015 | 3.946 ± 0.288 | 1.722 ± 0.424 | 1.672 ± 0.409 |
| 20 dB | a3 (main, 24-pt grid) | 0.090 ± 0.023 | 0.058 ± 0.017 | 0.032 ± 0.009 | 0.264 ± 0.030 | 0.902 ± 0.251 | 0.902 ± 0.251 |
| 20 dB | a3x (50-pt grid) | 0.090 ± 0.023 | 0.058 ± 0.017 | 0.032 ± 0.009 | 0.264 ± 0.030 | 0.902 ± 0.251 | 0.902 ± 0.251 |
| 20 dB | xapp (main, A3 held off) | 0.731 ± 0.171 | 0.393 ± 0.103 | 0.338 ± 0.102 | 0.772 ± 0.048 | 0.945 ± 0.264 | 0.902 ± 0.251 |
| 20 dB | hybrid (joint) | 0.288 ± 0.055 | 0.238 ± 0.051 | 0.050 ± 0.011 | 0.606 ± 0.126 | 0.945 ± 0.264 | 0.902 ± 0.251 |
| 20 dB | hybrid (A3 fixed) | 0.110 ± 0.024 | 0.076 ± 0.020 | 0.034 ± 0.009 | 3.815 ± 0.280 | 0.945 ± 0.264 | 0.902 ± 0.251 |
| 25 dB | a3 (main, 24-pt grid) | 0.052 ± 0.022 | 0.032 ± 0.019 | 0.020 ± 0.007 | 0.232 ± 0.029 | 0.300 ± 0.152 | 0.300 ± 0.152 |
| 25 dB | a3x (50-pt grid) | 0.052 ± 0.022 | 0.032 ± 0.019 | 0.020 ± 0.007 | 0.232 ± 0.029 | 0.300 ± 0.152 | 0.300 ± 0.152 |
| 25 dB | xapp (main, A3 held off) | 0.477 ± 0.126 | 0.374 ± 0.087 | 0.102 ± 0.056 | 0.696 ± 0.048 | 0.320 ± 0.155 | 0.300 ± 0.152 |
| 25 dB | hybrid (joint) | 0.532 ± 0.120 | 0.428 ± 0.092 | 0.105 ± 0.047 | 0.239 ± 0.039 | 0.320 ± 0.155 | 0.300 ± 0.152 |
| 25 dB | hybrid (A3 fixed) | 0.102 ± 0.027 | 0.079 ± 0.024 | 0.022 ± 0.007 | 3.265 ± 0.270 | 0.320 ± 0.155 | 0.300 ± 0.152 |
| 30 dB | a3 (main, 24-pt grid) | 0.029 ± 0.014 | 0.026 ± 0.013 | 0.003 ± 0.003 | 0.180 ± 0.025 | 0.099 ± 0.086 | 0.099 ± 0.086 |
| 30 dB | a3x (50-pt grid) | 0.045 ± 0.022 | 0.043 ± 0.022 | 0.002 ± 0.002 | 0.120 ± 0.016 | 0.099 ± 0.086 | 0.099 ± 0.086 |
| 30 dB | xapp (main, A3 held off) | 0.165 ± 0.081 | 0.130 ± 0.057 | 0.035 ± 0.035 | 0.465 ± 0.051 | 0.106 ± 0.089 | 0.099 ± 0.086 |
| 30 dB | hybrid (joint) | 0.207 ± 0.078 | 0.164 ± 0.059 | 0.043 ± 0.033 | 0.240 ± 0.039 | 0.106 ± 0.089 | 0.099 ± 0.086 |
| 30 dB | hybrid (A3 fixed) | 0.081 ± 0.034 | 0.077 ± 0.033 | 0.004 ± 0.003 | 0.616 ± 0.127 | 0.106 ± 0.089 | 0.099 ± 0.086 |
| 3GPP short-range reference | a3 (main, 24-pt grid) | 0.033 ± 0.017 | 0.031 ± 0.017 | 0.002 ± 0.002 | 0.154 ± 0.022 | 0.016 ± 0.013 | 0.016 ± 0.013 |
| 3GPP short-range reference | a3x (50-pt grid) | 0.027 ± 0.015 | 0.026 ± 0.015 | 0.001 ± 0.001 | 0.116 ± 0.016 | 0.016 ± 0.013 | 0.016 ± 0.013 |
| 3GPP short-range reference | xapp (main, A3 held off) | 0.101 ± 0.056 | 0.071 ± 0.041 | 0.030 ± 0.027 | 0.389 ± 0.035 | 0.017 ± 0.014 | 0.016 ± 0.013 |
| 3GPP short-range reference | hybrid (joint) | 0.070 ± 0.035 | 0.059 ± 0.032 | 0.011 ± 0.012 | 0.241 ± 0.039 | 0.017 ± 0.014 | 0.016 ± 0.013 |
| 3GPP short-range reference | hybrid (A3 fixed) | 0.023 ± 0.019 | 0.020 ± 0.019 | 0.004 ± 0.005 | 0.583 ± 0.122 | 0.017 ± 0.014 | 0.016 ± 0.013 |
| v1 radio (high margin) | a3 (main, 24-pt grid) | 0.018 ± 0.011 | 0.017 ± 0.011 | 0.001 ± 0.001 | 0.154 ± 0.022 | 0.011 ± 0.010 | 0.011 ± 0.010 |
| v1 radio (high margin) | a3x (50-pt grid) | 0.021 ± 0.010 | 0.019 ± 0.010 | 0.002 ± 0.003 | 0.101 ± 0.015 | 0.011 ± 0.010 | 0.011 ± 0.010 |
| v1 radio (high margin) | xapp (main, A3 held off) | 0.057 ± 0.030 | 0.039 ± 0.023 | 0.017 ± 0.019 | 0.389 ± 0.035 | 0.011 ± 0.010 | 0.011 ± 0.010 |
| v1 radio (high margin) | hybrid (joint) | 0.036 ± 0.018 | 0.033 ± 0.017 | 0.003 ± 0.004 | 0.241 ± 0.039 | 0.011 ± 0.010 | 0.011 ± 0.010 |
| v1 radio (high margin) | hybrid (A3 fixed) | 0.020 ± 0.013 | 0.017 ± 0.013 | 0.003 ± 0.004 | 0.425 ± 0.079 | 0.011 ± 0.010 | 0.011 ± 0.010 |

Follow-up wall time: 15.7 min (CPU, vectorised lanes; per margin up to 36 000 lanes x 5991 steps).

## 12. Second follow-up: genie bound, interruption-free handover, onset figure

Added after the M3 review (H2 not supported, accepted). The xApp predictor is unchanged.

**Genie + A3.** The first-round xApp policy (same trigger rule: serving-cell LoS loss ≥ 10 dB within H, other cell < 3 dB over that window, other cell's filtered SNR > serving − 10 dB; A3 held off until the later of hold and the predicted end; same E2 loop delay 20 ms, report period 0.1 s and tau_HO) driven by ground truth: the "prediction" at each report is the actual model-B LoS loss of both cells over the next 3 s (10 ms timeline), so start and end are perfect. Grid H {0.5,1,2,3} s x hold {0.2,0.5,1.0} s (12 points), A3 underlay = the A3 parameters tuned at that margin, tuned on tuning seeds per margin. "genie" pays the xApp's sensing overhead (bound for a sensing-based predictor); "genie, no overhead" does not. This bounds what any predictor could add to A3 under this policy.

### Genie bound per margin (evaluation seeds, tau_HO = 20 ms)

| Margin | Genie H / hold [s] | Oracle | A3 | genie + A3 | genie + A3, no overhead | xApp + A3 (held off) | hybrid (joint) | genie HO / UE-min | genie ping-pong |
|---|---|---|---|---|---|---|---|---|---|
| 0 dB | 0.5 / 0.2 | 34.559 ± 2.768 | 34.677 ± 2.760 | 36.586 ± 2.698 | 35.161 ± 2.694 | 38.473 ± 2.181 | 37.178 ± 2.481 | 13.9 ± 1.6 | 0.26 ± 0.04 |
| 5 dB | 0.5 / 0.2 | 8.969 ± 1.354 | 9.405 ± 1.376 | 11.298 ± 1.506 | 10.309 ± 1.464 | 18.382 ± 1.620 | 14.453 ± 1.471 | 13.9 ± 1.6 | 0.26 ± 0.04 |
| 10 dB | 0.5 / 0.2 | 2.871 ± 0.676 | 3.426 ± 0.711 | 3.653 ± 0.779 | 3.477 ± 0.742 | 6.247 ± 1.063 | 5.319 ± 1.059 | 13.9 ± 1.6 | 0.26 ± 0.04 |
| 15 dB | 0.5 / 0.2 | 1.672 ± 0.409 | 2.044 ± 0.440 | 2.145 ± 0.458 | 2.093 ± 0.444 | 3.463 ± 0.538 | 2.581 ± 0.531 | 13.9 ± 1.6 | 0.26 ± 0.04 |
| 20 dB | 0.5 / 0.2 | 0.902 ± 0.251 | 1.256 ± 0.282 | 1.346 ± 0.293 | 1.303 ± 0.282 | 2.448 ± 0.396 | 1.840 ± 0.381 | 13.9 ± 1.6 | 0.26 ± 0.04 |
| 25 dB | 0.5 / 0.2 | 0.300 ± 0.152 | 0.584 ± 0.171 | 0.621 ± 0.174 | 0.597 ± 0.170 | 1.493 ± 0.245 | 1.091 ± 0.249 | 12.3 ± 1.5 | 0.22 ± 0.04 |
| 30 dB | 1.0 / 0.2 | 0.099 ± 0.086 | 0.308 ± 0.104 | 0.333 ± 0.107 | 0.324 ± 0.102 | 0.736 ± 0.173 | 0.554 ± 0.172 | 9.9 ± 1.3 | 0.15 ± 0.03 |
| 3GPP short-range reference | 1.0 / 1.0 | 0.016 ± 0.013 | 0.203 ± 0.042 | 0.234 ± 0.042 | 0.233 ± 0.041 | 0.506 ± 0.084 | 0.328 ± 0.075 | 9.3 ± 1.2 | 0.07 ± 0.02 |
| v1 radio (high margin) | 1.0 / 1.0 | 0.011 ± 0.010 | 0.182 ± 0.032 | 0.214 ± 0.033 | 0.213 ± 0.033 | 0.456 ± 0.057 | 0.288 ± 0.055 | 9.3 ± 1.2 | 0.07 ± 0.02 |

Outage_req in s/UE-min, mean ± 95 % CI over 40 evaluation jobs.

### Genie bound per blocker class (interruption per 10 dB event [s], all events; proactive recall)

| Margin | Class | Events | Oracle | A3 | genie + A3 | genie + A3, no overhead | A3 proactive recall | genie proactive recall |
|---|---|---|---|---|---|---|---|---|
| 0 dB | bus/truck | 317 | 1.406 ± 0.168 | 1.407 ± 0.168 | 1.433 ± 0.174 | 1.432 ± 0.174 | 0.23 ± 0.05 | 0.70 ± 0.06 |
| 0 dB | pedestrian | 387 | 0.613 ± 0.096 | 0.614 ± 0.096 | 0.635 ± 0.095 | 0.632 ± 0.094 | 0.08 ± 0.03 | 0.26 ± 0.04 |
| 5 dB | bus/truck | 317 | 1.043 ± 0.138 | 1.057 ± 0.138 | 1.124 ± 0.140 | 1.108 ± 0.137 | 0.23 ± 0.05 | 0.70 ± 0.06 |
| 5 dB | pedestrian | 387 | 0.417 ± 0.092 | 0.435 ± 0.091 | 0.463 ± 0.091 | 0.451 ± 0.091 | 0.08 ± 0.03 | 0.26 ± 0.04 |
| 10 dB | bus/truck | 317 | 0.321 ± 0.095 | 0.377 ± 0.093 | 0.367 ± 0.101 | 0.343 ± 0.097 | 0.23 ± 0.05 | 0.70 ± 0.06 |
| 10 dB | pedestrian | 387 | 0.288 ± 0.064 | 0.311 ± 0.065 | 0.323 ± 0.065 | 0.316 ± 0.065 | 0.08 ± 0.03 | 0.26 ± 0.04 |
| 15 dB | bus/truck | 317 | 0.152 ± 0.063 | 0.177 ± 0.064 | 0.170 ± 0.067 | 0.163 ± 0.064 | 0.23 ± 0.05 | 0.70 ± 0.06 |
| 15 dB | pedestrian | 387 | 0.204 ± 0.046 | 0.223 ± 0.046 | 0.234 ± 0.047 | 0.229 ± 0.046 | 0.08 ± 0.03 | 0.26 ± 0.04 |
| 20 dB | bus/truck | 317 | 0.063 ± 0.034 | 0.088 ± 0.035 | 0.081 ± 0.037 | 0.076 ± 0.036 | 0.23 ± 0.05 | 0.70 ± 0.06 |
| 20 dB | pedestrian | 387 | 0.121 ± 0.032 | 0.137 ± 0.033 | 0.148 ± 0.032 | 0.144 ± 0.032 | 0.08 ± 0.03 | 0.26 ± 0.04 |
| 25 dB | bus/truck | 317 | 0.018 ± 0.023 | 0.038 ± 0.023 | 0.027 ± 0.024 | 0.026 ± 0.023 | 0.07 ± 0.03 | 0.66 ± 0.06 |
| 25 dB | pedestrian | 387 | 0.038 ± 0.017 | 0.050 ± 0.017 | 0.052 ± 0.018 | 0.049 ± 0.017 | 0.04 ± 0.02 | 0.24 ± 0.04 |
| 30 dB | bus/truck | 317 | 0.008 ± 0.013 | 0.024 ± 0.015 | 0.015 ± 0.015 | 0.015 ± 0.015 | 0.05 ± 0.03 | 0.56 ± 0.06 |
| 30 dB | pedestrian | 387 | 0.009 ± 0.009 | 0.021 ± 0.009 | 0.020 ± 0.010 | 0.019 ± 0.009 | 0.03 ± 0.01 | 0.18 ± 0.04 |
| 3GPP short-range reference | bus/truck | 317 | 0.001 ± 0.001 | 0.016 ± 0.004 | 0.007 ± 0.003 | 0.007 ± 0.003 | 0.02 ± 0.02 | 0.55 ± 0.06 |
| 3GPP short-range reference | pedestrian | 387 | 0.002 ± 0.002 | 0.010 ± 0.004 | 0.010 ± 0.004 | 0.010 ± 0.004 | 0.02 ± 0.01 | 0.18 ± 0.04 |
| v1 radio (high margin) | bus/truck | 317 | 0.001 ± 0.001 | 0.015 ± 0.003 | 0.006 ± 0.002 | 0.006 ± 0.002 | 0.02 ± 0.02 | 0.55 ± 0.06 |
| v1 radio (high margin) | pedestrian | 387 | 0.001 ± 0.001 | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.02 ± 0.01 | 0.18 ± 0.04 |

### Where the genie's outage comes from (same decomposition as section 11, s/UE-min)

| Margin | Scheme | (a) wrong cell | (a) in 10 dB events | (a) outside events | (b) interruption | (c) both unusable |
|---|---|---|---|---|---|---|
| 0 dB | A3 | 0.073 ± 0.011 | 0.004 ± 0.003 | 0.069 ± 0.010 | 0.045 ± 0.007 | 34.559 ± 2.768 |
| 0 dB | genie + A3 | 0.475 ± 0.112 | 0.007 ± 0.008 | 0.468 ± 0.109 | 0.063 ± 0.012 | 36.047 ± 2.771 |
| 0 dB | genie + A3, no overhead | 0.532 ± 0.117 | 0.009 ± 0.009 | 0.523 ± 0.114 | 0.070 ± 0.013 | 34.559 ± 2.768 |
| 5 dB | A3 | 0.284 ± 0.033 | 0.091 ± 0.013 | 0.193 ± 0.024 | 0.152 ± 0.015 | 8.969 ± 1.354 |
| 5 dB | genie + A3 | 1.144 ± 0.147 | 0.082 ± 0.032 | 1.062 ± 0.137 | 0.199 ± 0.021 | 9.955 ± 1.421 |
| 5 dB | genie + A3, no overhead | 1.137 ± 0.147 | 0.088 ± 0.033 | 1.048 ± 0.133 | 0.204 ± 0.020 | 8.969 ± 1.354 |
| 10 dB | A3 | 0.322 ± 0.047 | 0.193 ± 0.031 | 0.129 ± 0.023 | 0.234 ± 0.024 | 2.871 ± 0.676 |
| 10 dB | genie + A3 | 0.379 ± 0.097 | 0.095 ± 0.029 | 0.285 ± 0.080 | 0.243 ± 0.026 | 3.030 ± 0.703 |
| 10 dB | genie + A3, no overhead | 0.362 ± 0.090 | 0.107 ± 0.034 | 0.255 ± 0.070 | 0.244 ± 0.026 | 2.871 ± 0.676 |
| 15 dB | A3 | 0.120 ± 0.025 | 0.081 ± 0.017 | 0.039 ± 0.012 | 0.252 ± 0.027 | 1.672 ± 0.409 |
| 15 dB | genie + A3 | 0.166 ± 0.037 | 0.069 ± 0.027 | 0.097 ± 0.023 | 0.257 ± 0.028 | 1.722 ± 0.424 |
| 15 dB | genie + A3, no overhead | 0.163 ± 0.036 | 0.068 ± 0.027 | 0.095 ± 0.023 | 0.258 ± 0.028 | 1.672 ± 0.409 |
| 20 dB | A3 | 0.090 ± 0.023 | 0.058 ± 0.017 | 0.032 ± 0.009 | 0.264 ± 0.030 | 0.902 ± 0.251 |
| 20 dB | genie + A3 | 0.132 ± 0.027 | 0.060 ± 0.018 | 0.071 ± 0.018 | 0.269 ± 0.030 | 0.945 ± 0.264 |
| 20 dB | genie + A3, no overhead | 0.131 ± 0.027 | 0.060 ± 0.018 | 0.071 ± 0.018 | 0.270 ± 0.030 | 0.902 ± 0.251 |
| 25 dB | A3 | 0.052 ± 0.022 | 0.032 ± 0.019 | 0.020 ± 0.007 | 0.232 ± 0.029 | 0.300 ± 0.152 |
| 25 dB | genie + A3 | 0.060 ± 0.021 | 0.034 ± 0.019 | 0.026 ± 0.008 | 0.242 ± 0.030 | 0.320 ± 0.155 |
| 25 dB | genie + A3, no overhead | 0.055 ± 0.022 | 0.032 ± 0.019 | 0.023 ± 0.008 | 0.242 ± 0.030 | 0.300 ± 0.152 |
| 30 dB | A3 | 0.029 ± 0.014 | 0.026 ± 0.013 | 0.003 ± 0.003 | 0.180 ± 0.025 | 0.099 ± 0.086 |
| 30 dB | genie + A3 | 0.029 ± 0.014 | 0.027 ± 0.013 | 0.002 ± 0.002 | 0.197 ± 0.025 | 0.106 ± 0.089 |
| 30 dB | genie + A3, no overhead | 0.028 ± 0.014 | 0.026 ± 0.013 | 0.002 ± 0.002 | 0.197 ± 0.025 | 0.099 ± 0.086 |
| 3GPP short-range reference | A3 | 0.033 ± 0.017 | 0.031 ± 0.017 | 0.002 ± 0.002 | 0.154 ± 0.022 | 0.016 ± 0.013 |
| 3GPP short-range reference | genie + A3 | 0.031 ± 0.016 | 0.028 ± 0.016 | 0.003 ± 0.004 | 0.186 ± 0.024 | 0.017 ± 0.014 |
| 3GPP short-range reference | genie + A3, no overhead | 0.030 ± 0.016 | 0.027 ± 0.015 | 0.003 ± 0.004 | 0.186 ± 0.024 | 0.016 ± 0.013 |
| v1 radio (high margin) | A3 | 0.018 ± 0.011 | 0.017 ± 0.011 | 0.001 ± 0.001 | 0.154 ± 0.022 | 0.011 ± 0.010 |
| v1 radio (high margin) | genie + A3 | 0.016 ± 0.011 | 0.014 ± 0.009 | 0.002 ± 0.003 | 0.186 ± 0.024 | 0.011 ± 0.010 |
| v1 radio (high margin) | genie + A3, no overhead | 0.016 ± 0.011 | 0.014 ± 0.009 | 0.002 ± 0.003 | 0.186 ± 0.024 | 0.011 ± 0.010 |

Reading the decomposition (genie without overhead vs A3, means): the genie has less wrong-cell time inside 10 dB events at 5 dB, 10 dB, 15 dB, 25 dB, 3GPP short-range reference, v1 radio (high margin); it has more wrong-cell time outside events at 0 dB, 5 dB, 10 dB, 15 dB, 20 dB, 25 dB, 3GPP short-range reference, v1 radio (high margin). The second effect comes from the policy's hold, which keeps A3 off and so delays path-loss-driven cell changes; handover interruption (b) is not reduced by prediction. Under this policy even perfect prediction does not add to A3 within the CIs; the policy, not the predictor, sets the bound here. A different policy would be a method change and is not tested.

### Interruption-free handover (tau_HO = 0, multi-TRP / DAPS-like)

A3 and genie + A3 retuned on tuning seeds at tau_HO = 0 (same grids and objective); the oracle has no interruption in either case. Gap = scheme − oracle outage_req [s/UE-min]; closed = 1 − gap(tau_HO = 0) / gap(tau_HO = 20 ms).

| Margin | Oracle | A3 (20 ms) | A3 (0) | A3 gap closed | genie + A3 (20 ms) | genie + A3 (0) | genie gap (0) | A3 tau_HO=0 params offset/hyst/TTT |
|---|---|---|---|---|---|---|---|---|
| 0 dB | 34.559 ± 2.768 | 34.677 ± 2.760 | 34.632 ± 2.763 | 38 % | 36.586 ± 2.698 | 36.547 ± 2.700 | 1.988 | 1/1/40 |
| 5 dB | 8.969 ± 1.354 | 9.405 ± 1.376 | 9.253 ± 1.370 | 35 % | 11.298 ± 1.506 | 11.156 ± 1.502 | 2.187 | 1/1/40 |
| 10 dB | 2.871 ± 0.676 | 3.426 ± 0.711 | 3.193 ± 0.701 | 42 % | 3.653 ± 0.779 | 3.421 ± 0.771 | 0.550 | 1/1/40 |
| 15 dB | 1.672 ± 0.409 | 2.044 ± 0.440 | 1.792 ± 0.428 | 68 % | 2.145 ± 0.458 | 1.890 ± 0.448 | 0.218 | 1/1/40 |
| 20 dB | 0.902 ± 0.251 | 1.256 ± 0.282 | 0.992 ± 0.266 | 75 % | 1.346 ± 0.293 | 1.077 ± 0.279 | 0.176 | 1/1/40 |
| 25 dB | 0.300 ± 0.152 | 0.584 ± 0.171 | 0.337 ± 0.157 | 87 % | 0.621 ± 0.174 | 0.367 ± 0.159 | 0.067 | 1/1/40 |
| 30 dB | 0.099 ± 0.086 | 0.308 ± 0.104 | 0.114 ± 0.089 | 93 % | 0.333 ± 0.107 | 0.120 ± 0.093 | 0.021 | 1/1/40 |
| 3GPP short-range reference | 0.016 ± 0.013 | 0.203 ± 0.042 | 0.023 ± 0.017 | 97 % | 0.234 ± 0.042 | 0.026 ± 0.018 | 0.010 | 1/3/40 |
| v1 radio (high margin) | 0.011 ± 0.010 | 0.182 ± 0.032 | 0.017 ± 0.011 | 97 % | 0.214 ± 0.033 | 0.016 ± 0.011 | 0.005 | 1/2/80 |

### Onset figure

`results/M3/onset/onset_events.png|pdf` (data in `onset_events.json`). Selection rule, fixed before plotting: per class, the evaluation-seed 10 dB event whose duration is closest to the class median duration (ties: lowest seed, mount, density, UE, start). Curves: model-B LoS loss of the cell the event occurs on (the fixed cell at event start) and of the other cell, 10 ms resolution, capped at 60 dB for display. Vertical lines: handover switch times of A3 and genie + A3 (tuned parameters) at the 3GPP reference margin (solid) and at 10 dB (dashed); the A3 decision is 10 ms before its switch.

| Class | Job / UE | Event [s] | Onset 10–90 % [s] | Blocker | Handovers (switch times [s]) |
|---|---|---|---|---|---|
| bus/truck | seed 1003 facade/low, UE 0 | 16.84–18.01 | 0.70 | truck-5#2 | A3 @ 3GPP ref.: none; genie+A3 @ 3GPP ref.: 15.93, 18.22; A3 @ 10 dB: 16.91, 18.08; genie+A3 @ 10 dB: 16.43, 18.08 |
| pedestrian | seed 1001 lamppost/high, UE 1 | 24.10–24.32 | 0.01 | ped-0#-1 | A3 @ 3GPP ref.: 21.13, 22.22; genie+A3 @ 3GPP ref.: 22.22; A3 @ 10 dB: 22.08, 25.92; genie+A3 @ 10 dB: 22.08, 25.92 |

Second follow-up wall time: 0.8 min.

## 13. Third follow-up (genie bound only): foresight bound and an onset-advance genie policy

This concerns the genie bound only; the sensing xApp and its predictor are unchanged.

### Policy-independent foresight bound

With the tuned A3 (tau_HO = 20 ms, main-run parameters) on the evaluation seeds: wrong-cell time inside 10 dB events of the UE, i.e. steps where the serving cell is below the service rate while the other cell is at or above it, per blocker class (class of the event), in s/UE-min and as a share of A3's total outage_req (per job, mean ± 95 % CI; and pooled over all jobs). This is the most that any blockage prediction can recover from A3's outage under any policy: prediction cannot remove handover interruption (b) or time when both cells are unusable (c). Wrong-cell time outside events (path-loss driven cell changes) is not blockage foresight and is not counted.

| Margin | A3 outage_req | Class | Foresight-recoverable [s/UE-min] | Share of A3 outage (per job) | Share (pooled) |
|---|---|---|---|---|---|
| 0 dB | 34.677 ± 2.760 | all | 0.004 ± 0.003 | 0.0 ± 0.0 % | 0.0 % |
| 0 dB |  | bus/truck | 0.002 ± 0.002 | 0.0 ± 0.0 % | 0.0 % |
| 0 dB |  | pedestrian | 0.002 ± 0.002 | 0.0 ± 0.0 % | 0.0 % |
| 5 dB | 9.405 ± 1.376 | all | 0.091 ± 0.013 | 1.1 ± 0.2 % | 1.0 % |
| 5 dB |  | bus/truck | 0.031 ± 0.011 | 0.3 ± 0.1 % | 0.3 % |
| 5 dB |  | pedestrian | 0.060 ± 0.012 | 0.8 ± 0.2 % | 0.6 % |
| 10 dB | 3.426 ± 0.711 | all | 0.193 ± 0.031 | 7.3 ± 1.3 % | 5.6 % |
| 10 dB |  | bus/truck | 0.117 ± 0.029 | 4.3 ± 1.2 % | 3.4 % |
| 10 dB |  | pedestrian | 0.076 ± 0.017 | 3.1 ± 0.9 % | 2.2 % |
| 15 dB | 2.044 ± 0.440 | all | 0.081 ± 0.017 | 4.6 ± 1.0 % | 3.9 % |
| 15 dB |  | bus/truck | 0.025 ± 0.010 | 1.3 ± 0.6 % | 1.2 % |
| 15 dB |  | pedestrian | 0.056 ± 0.012 | 3.3 ± 0.8 % | 2.7 % |
| 20 dB | 1.256 ± 0.282 | all | 0.058 ± 0.017 | 5.3 ± 1.7 % | 4.6 % |
| 20 dB |  | bus/truck | 0.023 ± 0.013 | 2.4 ± 1.6 % | 1.8 % |
| 20 dB |  | pedestrian | 0.035 ± 0.013 | 2.9 ± 1.1 % | 2.8 % |
| 25 dB | 0.584 ± 0.171 | all | 0.032 ± 0.019 | 4.9 ± 2.2 % | 5.6 % |
| 25 dB |  | bus/truck | 0.010 ± 0.010 | 1.4 ± 1.0 % | 1.8 % |
| 25 dB |  | pedestrian | 0.022 ± 0.012 | 3.6 ± 1.9 % | 3.8 % |
| 30 dB | 0.308 ± 0.104 | all | 0.026 ± 0.013 | 7.0 ± 3.0 % | 8.4 % |
| 30 dB |  | bus/truck | 0.008 ± 0.009 | 1.1 ± 1.2 % | 2.5 % |
| 30 dB |  | pedestrian | 0.018 ± 0.009 | 6.0 ± 2.9 % | 5.9 % |
| 3GPP short-range reference | 0.203 ± 0.042 | all | 0.031 ± 0.017 | 8.2 ± 4.0 % | 15.4 % |
| 3GPP short-range reference |  | bus/truck | 0.010 ± 0.011 | 2.1 ± 2.4 % | 4.8 % |
| 3GPP short-range reference |  | pedestrian | 0.022 ± 0.011 | 6.1 ± 3.0 % | 10.7 % |
| v1 radio (high margin) | 0.182 ± 0.032 | all | 0.017 ± 0.011 | 5.2 ± 3.1 % | 9.1 % |
| v1 radio (high margin) |  | bus/truck | 0.005 ± 0.009 | 1.1 ± 2.0 % | 2.7 % |
| v1 radio (high margin) |  | pedestrian | 0.012 ± 0.008 | 4.1 ± 2.6 % | 6.5 % |

No 10 dB events are caused by cars, so the car class is omitted.

### Genie, onset-advance policy

**This policy was added after the first genie result** (section 12), to test whether a different use of perfect prediction helps. A3 is always active and there is no global hold. The genie (ground-truth LoS loss) only advances a handover when the predicted blockage of the serving cell starts at least 30 ms after the report (E2 loop delay + 10 ms, so the switch lands before onset) and the other cell is predicted clear (< 3 dB) for the predicted blockage; the other cell's filtered SNR must exceed serving − 10 dB as before. After such a handover A3's hand-back is blocked only until the predicted end of the blockage. Grid H {0.5,1,2,3} s (4 points), A3 underlay = the A3 tuned at that margin and tau_HO, tuned on tuning seeds per margin.

| Margin | tau_HO | A3 | genie, 1st policy (no ovh.) | onset genie (no ovh.) | onset genie (with ovh.) | onset H [s] | onset HO/UE-min | onset precision | onset proactive recall |
|---|---|---|---|---|---|---|---|---|---|
| 0 dB | 20 ms | 34.677 ± 2.760 | 35.161 ± 2.694 | 35.121 ± 2.692 | 36.546 ± 2.696 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 0 dB | 0 ms | 34.632 ± 2.763 | 35.118 ± 2.697 | 35.081 ± 2.695 | 36.510 ± 2.698 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 5 dB | 20 ms | 9.405 ± 1.376 | 10.309 ± 1.464 | 10.279 ± 1.466 | 11.262 ± 1.506 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 5 dB | 0 ms | 9.253 ± 1.370 | 10.162 ± 1.459 | 10.136 ± 1.460 | 11.124 ± 1.501 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 10 dB | 20 ms | 3.426 ± 0.711 | 3.477 ± 0.742 | 3.474 ± 0.742 | 3.650 ± 0.779 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 10 dB | 0 ms | 3.193 ± 0.701 | 3.243 ± 0.734 | 3.246 ± 0.734 | 3.425 ± 0.771 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 15 dB | 20 ms | 2.044 ± 0.440 | 2.093 ± 0.444 | 2.089 ± 0.444 | 2.141 ± 0.459 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 15 dB | 0 ms | 1.792 ± 0.428 | 1.837 ± 0.433 | 1.838 ± 0.433 | 1.892 ± 0.448 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 20 dB | 20 ms | 1.256 ± 0.282 | 1.303 ± 0.282 | 1.298 ± 0.282 | 1.341 ± 0.293 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 20 dB | 0 ms | 0.992 ± 0.266 | 1.033 ± 0.267 | 1.034 ± 0.267 | 1.078 ± 0.279 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 25 dB | 20 ms | 0.584 ± 0.171 | 0.597 ± 0.170 | 0.591 ± 0.169 | 0.615 ± 0.174 | 0.5 | 12.0 ± 1.5 | 0.85 ± 0.05 | 0.44 ± 0.05 |
| 25 dB | 0 ms | 0.337 ± 0.157 | 0.341 ± 0.155 | 0.342 ± 0.155 | 0.367 ± 0.159 | 0.5 | 13.6 ± 1.6 | 0.89 ± 0.04 | 0.47 ± 0.05 |
| 30 dB | 20 ms | 0.308 ± 0.104 | 0.324 ± 0.102 | 0.313 ± 0.103 | 0.322 ± 0.107 | 1.0 | 9.4 ± 1.2 | 0.80 ± 0.04 | 0.35 ± 0.04 |
| 30 dB | 0 ms | 0.114 ± 0.089 | 0.112 ± 0.089 | 0.112 ± 0.089 | 0.121 ± 0.093 | 1.0 | 13.4 ± 1.5 | 0.84 ± 0.05 | 0.46 ± 0.05 |
| 3GPP short-range reference | 20 ms | 0.203 ± 0.042 | 0.233 ± 0.041 | 0.218 ± 0.040 | 0.219 ± 0.041 | 1.0 | 8.5 ± 1.1 | 0.81 ± 0.04 | 0.35 ± 0.04 |
| 3GPP short-range reference | 0 ms | 0.023 ± 0.017 | 0.025 ± 0.018 | 0.022 ± 0.017 | 0.023 ± 0.018 | 0.5 | 12.0 ± 1.5 | 0.85 ± 0.05 | 0.44 ± 0.05 |
| v1 radio (high margin) | 20 ms | 0.182 ± 0.032 | 0.213 ± 0.033 | 0.198 ± 0.032 | 0.199 ± 0.032 | 1.0 | 8.5 ± 1.1 | 0.81 ± 0.04 | 0.35 ± 0.04 |
| v1 radio (high margin) | 0 ms | 0.017 ± 0.011 | 0.016 ± 0.011 | 0.016 ± 0.011 | 0.017 ± 0.012 | 1.0 | 12.2 ± 1.5 | 0.83 ± 0.05 | 0.44 ± 0.05 |

Outage_req in s/UE-min (mean ± 95 % CI, 40 evaluation jobs).

Decomposition at tau_HO = 20 ms (s/UE-min, no overhead):

| Margin | Scheme | (a) in 10 dB events | (a) outside events | (b) interruption | (c) both unusable |
|---|---|---|---|---|---|
| 0 dB | A3 | 0.004 ± 0.003 | 0.069 ± 0.010 | 0.045 ± 0.007 | 34.559 ± 2.768 |
| 0 dB | genie, 1st policy | 0.009 ± 0.009 | 0.523 ± 0.114 | 0.070 ± 0.013 | 34.559 ± 2.768 |
| 0 dB | onset genie | 0.009 ± 0.009 | 0.487 ± 0.117 | 0.067 ± 0.013 | 34.559 ± 2.768 |
| 5 dB | A3 | 0.091 ± 0.013 | 0.193 ± 0.024 | 0.152 ± 0.015 | 8.969 ± 1.354 |
| 5 dB | genie, 1st policy | 0.088 ± 0.033 | 1.048 ± 0.133 | 0.204 ± 0.020 | 8.969 ± 1.354 |
| 5 dB | onset genie | 0.089 ± 0.033 | 1.022 ± 0.134 | 0.199 ± 0.020 | 8.969 ± 1.354 |
| 10 dB | A3 | 0.193 ± 0.031 | 0.129 ± 0.023 | 0.234 ± 0.024 | 2.871 ± 0.676 |
| 10 dB | genie, 1st policy | 0.107 ± 0.034 | 0.255 ± 0.070 | 0.244 ± 0.026 | 2.871 ± 0.676 |
| 10 dB | onset genie | 0.109 ± 0.035 | 0.256 ± 0.070 | 0.238 ± 0.025 | 2.871 ± 0.676 |
| 15 dB | A3 | 0.081 ± 0.017 | 0.039 ± 0.012 | 0.252 ± 0.027 | 1.672 ± 0.409 |
| 15 dB | genie, 1st policy | 0.068 ± 0.027 | 0.095 ± 0.023 | 0.258 ± 0.028 | 1.672 ± 0.409 |
| 15 dB | onset genie | 0.069 ± 0.027 | 0.096 ± 0.023 | 0.252 ± 0.027 | 1.672 ± 0.409 |
| 20 dB | A3 | 0.058 ± 0.017 | 0.032 ± 0.009 | 0.264 ± 0.030 | 0.902 ± 0.251 |
| 20 dB | genie, 1st policy | 0.060 ± 0.018 | 0.071 ± 0.018 | 0.270 ± 0.030 | 0.902 ± 0.251 |
| 20 dB | onset genie | 0.060 ± 0.018 | 0.071 ± 0.018 | 0.264 ± 0.029 | 0.902 ± 0.251 |
| 25 dB | A3 | 0.032 ± 0.019 | 0.020 ± 0.007 | 0.232 ± 0.029 | 0.300 ± 0.152 |
| 25 dB | genie, 1st policy | 0.032 ± 0.019 | 0.023 ± 0.008 | 0.242 ± 0.030 | 0.300 ± 0.152 |
| 25 dB | onset genie | 0.032 ± 0.019 | 0.023 ± 0.008 | 0.235 ± 0.029 | 0.300 ± 0.152 |
| 30 dB | A3 | 0.026 ± 0.013 | 0.003 ± 0.003 | 0.180 ± 0.025 | 0.099 ± 0.086 |
| 30 dB | genie, 1st policy | 0.026 ± 0.013 | 0.002 ± 0.002 | 0.197 ± 0.025 | 0.099 ± 0.086 |
| 30 dB | onset genie | 0.026 ± 0.013 | 0.002 ± 0.002 | 0.186 ± 0.024 | 0.099 ± 0.086 |
| 3GPP short-range reference | A3 | 0.031 ± 0.017 | 0.002 ± 0.002 | 0.154 ± 0.022 | 0.016 ± 0.013 |
| 3GPP short-range reference | genie, 1st policy | 0.027 ± 0.015 | 0.003 ± 0.004 | 0.186 ± 0.024 | 0.016 ± 0.013 |
| 3GPP short-range reference | onset genie | 0.029 ± 0.017 | 0.003 ± 0.004 | 0.171 ± 0.022 | 0.016 ± 0.013 |
| v1 radio (high margin) | A3 | 0.017 ± 0.011 | 0.001 ± 0.001 | 0.154 ± 0.022 | 0.011 ± 0.010 |
| v1 radio (high margin) | genie, 1st policy | 0.014 ± 0.009 | 0.002 ± 0.003 | 0.186 ± 0.024 | 0.011 ± 0.010 |
| v1 radio (high margin) | onset genie | 0.015 ± 0.011 | 0.002 ± 0.003 | 0.171 ± 0.022 | 0.011 ± 0.010 |

Onset genie (no overhead) vs A3 at tau_HO = 20 ms (95 % CIs): lower at no margin; overlapping at ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB', '25 dB', '30 dB', '3GPP short-range reference', 'v1 radio (high margin)']; higher at no margin.

Third follow-up wall time: 28 s.

### Regression check: the follow-up code changes did not change the existing schemes

The follow-ups changed `xapp/schemes.py` (the simulator also returns the interruption mask) and `scripts/run_m3.py` (`make_lanes(..., hybrid=False)`; the prediction source may be the string "genie"). After the last change the main run was repeated with the changed code on the same timeline caches and the same configuration, and every metric was compared with the earlier `metrics.json` for exact equality (no tolerance). "xApp" here is the first-round scheme (xApp + A3, A3 held off during the xApp hold).

| Scheme | Evaluation metrics, all margins | Tuned parameters + objective, all margins | Full tuning grid |
|---|---|---|---|
| no action (fixed cell) | identical | n/a (not tuned) | n/a (not tuned) |
| oracle cell selection | identical | n/a (not tuned) | n/a (not tuned) |
| same-cell analog beam | identical | n/a (not tuned) | n/a (not tuned) |
| A3 (reactive) | identical | identical | identical |
| RSRP trend + A3 | identical | identical | identical |
| xApp + A3 (A3 held off during xApp hold) | identical | identical | identical |

H3 and tau_HO sweeps: identical. Budget, margin distribution, event statistics: budget identical, margin_distribution identical, event_stats identical. All identical.

