# Second external review, Block 3: held-out evaluation (seeds 2001-2010)

Frozen code: tag v1.1-freeze (de179bb). Pipeline: scripts/run_heldout.sh with S2C_EVAL_SET=heldout (logs: results/heldout_logs/).
Traces: 40 held-out jobs (sensing caches, detections, comm traces, comm geometry; geometry check vs comm_trace.json: max relative difference 0).
Nothing was recalibrated or retuned on held-out seeds. Tuning reruns on the tuning seeds 101-105 reproduced every tuned parameter of the development run exactly
(A5 / A3-wide / genie / sensing-planner H and budget, best reactive, B6 guarded planner, uncertainty-aware planner K/alpha/theta, M3 schemes); the sweeps' fixed H
and the tracker-error calibration were read from the development results. Development values: results/dev/ (snapshot) and 'dev_value' in results/M5/numbers_catalog.json.
Not rerun on held-out (not used by the paper any more): review_a1.json, review_b5.json (its tuned H is read by the sweeps), genie2.json.

## Macros that change beyond their rounding (85 of the catalogued macros)

| Macro | development | held-out |
|---|---|---|
| numBusShare | 44--47\% | 41--42\% |
| numPedShare | 53--56\% | 58--59\% |
| numSameCellLoss | 26--34 | 26--33 |
| numOtherCellBus | 68--71\% | 58--61\% |
| numOtherCellPed | 27--34\% | 29--30\% |
| numOnsetBus | 0.49 | 0.47 |
| numPedPd | 0.46 | 0.48 |
| numFA | 3.9 | 3.8 |
| numLeadBus | 84--90\% | 83--90\% |
| numLeadPed | 53--63\% | 53--65\% |
| numMapBus | 0.82 to 0.90 | 0.83 to 0.91 |
| numMapCross | 2.0 to 0.5 | 2.1 to 0.4 |
| numGhostGain | 0.82 to 0.88 | 0.83 to 0.88 |
| numAthreeTen | 3.43 | 3.58 |
| numOracleTen | 2.87 | 3.04 |
| numGenieTen | 3.48 | 3.64 |
| numHybridTen | 5.32 | 6.04 |
| numAthreeRef | 0.20 | 0.21 |
| numOracleRef | 0.02 | 0.03 |
| numGenieRef | 0.23 | 0.24 |
| numHybridRef | 0.33 | 0.36 |
| numXappTen | 6.25 | 6.71 |
| numGenieMatch | 87\% | 79\% |
| numAthreeMatch | 44\% | 43\% |
| numGeniePreBus | 70\% | 65\% |
| numAthreePreBus | 23\% | 22\% |
| numForesightRange | 2--9\% | 2--11\% |
| numForesightRef | 16\% | 10\% |
| numForesightMaxAbs | 0.32 | 0.31 |
| numTauZeroLow | 35--42\% | 35--41\% |
| numTauZeroHigh | 68--93\% | 61--95\% |
| numAfiveTen | 3.29 | 3.51 |
| numCostOracleTen | 2.95 | 3.13 |
| numGeniePlanTen | 2.99 | 3.17 |
| numSensePlanTen | 9.07 | 10.34 |
| numAfiveRef | 0.08 | 0.09 |
| numCostOracleRef | 0.02 | 0.03 |
| numGeniePlanRef | 0.03 | 0.04 |
| numSensePlanRef | 0.13 | 0.19 |
| numTrueLossRef | 0.04 | 0.05 |
| numValueRange | 74--89\% | 73--88\% |
| numValuePedRange | 47--64\% | 48--65\% |
| numValueBusRange | 12--31\% | 10--29\% |
| numValuePedTen | 51\% | 53\% |
| numValueBusTen | 31\% | 29\% |
| numGridCost | 2.0\% | 1.9\% |
| numRelRedTen | 10\% | 11\% |
| numAbsRedTen | 0.34 | 0.38 |
| numRelRedRef | 71\% | 62\% |
| numAbsRedRef | 0.056 | 0.057 |
| numAthreeWideRef | 0.14 | 0.16 |
| numSensePlanHO | 23.7 | 25.3 |
| numSensePlanPP | 0.54 | 0.56 |
| numAfiveHO | 5.9 | 6.6 |
| numPairedMaxP | 0.0053 | 0.0013 |
| numRobustTen | +2.56 | +3.01 |
| numRobustRef | +0.033 | +0.025 |
| numDensLowTen | 2.15 | 1.76 |
| numRelRedRefLow | 86\% | 88\% |
| numDensHighTen | 4.44 | 5.26 |
| numRelRedRefHigh | 69\% | 59\% |
| numErrPerfTen | +0.21$^\dagger$ | +0.33$^\dagger$ |
| numErrPerfRef | -0.040 | -0.042 |
| numErrMissTen | +2.12 | +2.71 |
| numErrMissRef | +0.047$^\dagger$ | +0.108 |
| numErrFalseTen | +0.69 | +0.70 |
| numErrFalseRef | -0.020$^\dagger$ | +0.006$^\dagger$ |
| numErrNoiseTen | +1.14 | +1.30 |
| numErrNoiseRef | +0.025$^\dagger$ | +0.049 |
| numErrSizeTen | +1.20 | +0.89 |
| numErrSizeRef | +0.009$^\dagger$ | -0.018$^\dagger$ |
| numErrAllTen | +4.52 | +4.92 |
| numErrAllRef | +0.045 | +0.099 |
| numErrAllUETen | +6.24 | +6.91 |
| numErrAllUERef | +0.059 | +0.101 |
| numBEPosRange | 0.08--0.16 | 0.12--0.18 |
| numBEPosRef | 0.57 | 0.64 |
| numBEUERange | 0.05--0.10 | 0.07--0.11 |
| numBEUERef | 0.24 | 0.18 |
| numBEVelMin | 0.97 | $>$1 |
| numErrRealTen | +5.78 | +6.83 |
| numErrRealRef | +0.053 | +0.099 |
| numUAPlanTen | +0.28 | +0.37 |
| numUAPlanRef | +0.003 | +0.008 |
| numUAPlanIntervene | 7.4\% | 6.6\% |

## Claims: development vs held-out

Per margin 0, 5, 10, 15, 20, 25, 30 dB, Ref, v1. '<'/'>' = mean below/above A5, '*' = Wilcoxon p < 0.05.

```
sensing-planner vs A5: dev >* >* >* >* >* >* >* >* >* | ho >* >* >* >* >* >* >* >* >* | FLIPS: none
true-loss planner vs A5: dev >* >* >  <* <* <* <* <* <* | ho >* >* >  <* <* <* <* <* <* | FLIPS: none
guarded planner (B6) vs A5: dev >* >* >* >* >* >* >* >* >* | ho >* >* >* >* >* >* >* >* >* | FLIPS: none
UA planner vs A5: dev >* >* >* >* >* >* >* >* >* | ho >* >* >* >* >* >* >* >* >* | FLIPS: none
perfect | ue 0.0 vs A5: dev >* >* >  <* <* <* <* <* <* | ho >* >* >  <* <* <* <* <* <* | FLIPS: none
R table miss vs A5: dev >* >* >* >* >* >* >* >  >* | ho >* >* >* >* >* >* >* >* >* | FLIPS: 3GPP short-range reference
R table false vs A5: dev >* >* >* >* >* >* >* <  >  | ho >* >* >* >* >* >* >* >  >* | FLIPS: 3GPP short-range reference, v1 radio (high margin)
R table noise vs A5: dev >* >* >* >* >* >* >* >  >* | ho >* >* >* >* >* >* >* >* >  | FLIPS: 3GPP short-range reference, v1 radio (high margin)
R table size vs A5: dev >* >* >* >* >* >* >* >  >  | ho >* >* >* >* >  >* >  <  >  | FLIPS: 20 dB, 30 dB, 3GPP short-range reference
R table all vs A5: dev >* >* >* >* >* >* >* >* >* | ho >* >* >* >* >* >* >* >* >* | FLIPS: none
R table all + ue 1 vs A5: dev >* >* >* >* >* >* >* >* >* | ho >* >* >* >* >* >* >* >* >* | FLIPS: none
perfect | ue 1.0 vs A5: dev >* >* >* >* >* >* >* >* >* | ho >* >* >* >* >* >* >* >* >* | FLIPS: none
R pos 0.1 | ue 0.0 vs A5: dev >* >* >  >  <  <  <  <* <* | ho >* >* >* <  <* <  <* <* >* | FLIPS: 10 dB, 15 dB, 20 dB, 30 dB, v1 radio (high margin)
R vel 1.0 | ue 0.0 vs A5: dev >* >* >  >  <  <* <  <* <* | ho >* >* >* <  <* <  <* <* <* | FLIPS: 10 dB, 15 dB, 20 dB, 25 dB, 30 dB
largest single cause (realistic): dev size miss miss miss miss miss miss miss miss | ho size miss miss miss miss miss miss miss miss | FLIPS: none
cause ranking (M miss, F false, N noise, S size): dev SFNM MNSF MSNF MNSF MNSF MNSF MNSF MNSF MNSF | ho SFNM MNSF MNSF MNFS MNFS MNFS MNFS MNFS MNFS | FLIPS: 10 dB, 15 dB, 20 dB, 25 dB, 30 dB, 3GPP short-range reference, v1 radio (high margin)
BE blocker pos: dev ['none', 'none', 'none', '0.08', '0.16', '0.12', '0.10', '0.57', '0.14'] | ho ['none', 'none', 'none', '0.12', '0.14', '0.18', '0.18', '0.64', '0.10']
BE blocker vel: dev ['none', 'none', 'none', '0.97', '>1', '>1', '>1', '>1', '>1'] | ho ['none', 'none', 'none', '>1', '>1', '>1', '>1', '>1', '>1']
BE UE pos: dev ['none', 'none', 'none', '0.08', '0.09', '0.10', '0.05', '0.24', '0.13'] | ho ['none', 'none', 'none', '0.10', '0.07', '0.10', '0.11', '0.18', '0.09']
BE pos+vel: dev ['none', 'none', 'none', '0.06', '0.12', '0.12', '0.11', '0.57', '0.09'] | ho ['none', 'none', 'none', '0.07', '0.07', '0.22', '0.26', '0.40', '0.11']
UE error binding (UE break-even < blocker-pos break-even): dev n/a n/a n/a equal UE tighter UE tighter UE tighter UE tighter UE tighter | ho n/a n/a n/a UE tighter UE tighter UE tighter UE tighter UE tighter UE tighter | FLIPS: 15 dB
any blocker accuracy beats A5 with UE 1 m (R pos s|ue1 any <0): dev no no no no no no no no no | ho no no no no no no no no no | FLIPS: none
better reactive (A3 wide vs A5): dev A3w A5 A5 A5 A5 A5 A5 A5 A5 | ho A5 A5 A3w A5 A5 A5 A5 A5 A5 | FLIPS: 0 dB, 10 dB
high density A5 > low density A5: dev yes yes yes yes yes yes yes yes yes | ho yes yes yes yes yes yes yes yes yes | FLIPS: none
```

Figure note: fig_onset panel (a) uses a fixed 0-20 dB axis for the bus/truck panel; the held-out bus event exceeds 20 dB and is clipped (frozen code, not changed).
