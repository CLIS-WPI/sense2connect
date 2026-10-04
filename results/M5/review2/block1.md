# Second external review, Block 1: clarification (no new results)

All numbers are from existing files (evaluation seeds 1001-1010, 40 jobs).

## 1. UE position error (sigma 1 m)
- Source: `xapp/predict_torch.ue_fixes` (configs/m3.yaml `rework.predict.ue_sigma_m: 1.0`).
  Gaussian, sigma 1 m per horizontal axis (x, y), height exact; drawn
  independently per 0.1 s report (white, no temporal correlation); seed
  `seed*17+3`, i.e. the same draw for the four mount/density jobs of a seed.
- Where applied: ONLY in the blockage geometry of the prediction. The
  predicted UE path is `fix(r) + tau * v_UE` (true UE velocity); the
  O-RU->UE segment used by the model-B screen and the 0.5 L + 8 m
  prefilter is built from it. The unblocked SNR used by the planner is the
  digital-twin value along the TRUE UE path (no UE error).
- Active in: the real sensing-planner (`pred_2`/`pred_4` in
  `scripts/run_m3.py`, used by `run_m5_planner.py`, B6 robust planner).
  NOT active in: genie-planner (true blocked SNR), true-LoS-loss planner
  (true loss), and the B5 perfect-track / error-injection runs
  (`review_b5_ablation.py` passes the true UE position). Hence Table II
  and the perfect-track baseline have no UE error, while the real
  sensing-planner has sigma 1 m. This is a confound between "All" (+5.46
  at 10 dB) and the real planner (+5.78); Block 2 item 4 quantifies it.
- Relative geometry: blocker k at lead tau is `p_k + tau v_k` (lane /
  sidewalk tracks of the real tracker: y pinned to the line, v_y = 0;
  perfect/injected tracks: NOT pinned), UE at `fix + tau v_UE`; per cell
  the segment O-RU->UE(tau) is screened by every blocker box (model B,
  `sim/comm/blockage_torch.screen_loss_db`), losses summed, any full block = inf.

## 2. Error-injection specification (`scripts/review_b5_ablation.py`)
- Misses: independent Bernoulli per blocker AND per report (each 0.1 s
  report drops each blocker independently). No whole-track drops, no
  temporal correlation. "miss measured" uses p = 1 - track Pd per class
  (bus/truck 0.059, pedestrian 0.535, car 0.237).
  Caveat: the real tracker loses a pedestrian for long runs; per-report
  independent drops are recovered at the next report by the receding
  planner, so this likely UNDERSTATES the effect of real misses.
- False tracks: count per report ~ Poisson(lambda); each lives for ONE
  report only (re-drawn every report), bus box on a random lane (6-12 m/s)
  or pedestrian box on a random sidewalk (1.0-1.5 m/s), x ~ U[-40, 40] m.
  lambda = 13.45 per report = unmatched CONFIRMED TRACKS per CPI of the
  final map tracker (track_fragment 4.24 + track_ghost 0.13 + track_other
  9.08; image ghost handling, budget 4, pooled over mounts; one CPI per
  0.1 s report). Unit: track instances per CPI (a false track alive for N
  CPIs counts N times).
  The 3.87 "FA/CPI" is a different quantity: unmatched CLUSTERS (detector
  + DBSCAN output, before tracking) per CPI that are ghosts or "other",
  excluding fragments (1.53) of real targets; all unmatched clusters 5.40.
  Tracks exceed clusters because confirmed tracks coast through missed
  frames, and a track of a real target whose estimate leaves the class
  gate counts as unmatched (fragment if within the target box + 1 m,
  otherwise other). Caveats: the injected rate includes fragments (4.24),
  which lie next to a real blocker; injected tracks are uniformly placed
  and non-persistent, unlike real ones.
- Noise: zero-mean Gaussian, independent per blocker, per report (no
  temporal correlation), per horizontal axis; position (x, y) and velocity
  (vx, vy) drawn independently. "noise s": sigma s m and s m/s on every
  axis (position and velocity tied to the same s). "noise measured":
  per-class sigma = tracker RMSE / sqrt(2) per axis (position: bus/truck
  1.55 m, ped 0.40 m, car 0.89 m; velocity: 2.45, 1.02, 1.99 m/s; velocity
  RMSE measured after 1 s track age). The real tracker's errors are
  temporally correlated (filter) and its lane/sidewalk tracks have y pinned,
  which the injection does not reproduce.
- Seeds: rng per (job, condition); conditions injected independently.

## 3. 95 % CIs (paired, t-based, 39 dof; Wilcoxon p two-sided) [s/UE-min]
| Condition minus A5 | 10 dB | Reference |
|---|---|---|
| Perfect tracks | +0.206 [+0.009, +0.403], p=0.69 | -0.040 [-0.074, -0.006], p=9.5e-6 |
| Misses (measured) | +0.677 [+0.405, +0.950], p=1.8e-7 | +0.018 [-0.014, +0.050], p=0.013 |
| False tracks (1x) | +1.146 [+0.967, +1.325], p=1.8e-12 | +0.053 [+0.023, +0.084], p=3.2e-5 |
| Pos./vel. (measured) | +1.611 [+1.258, +1.963], p=3.9e-8 | +0.046 [+0.004, +0.088], p=0.0078 |
| All | +5.457 [+4.649, +6.264], p=1.8e-12 | +0.071 [+0.027, +0.115], p=3.2e-5 |
| Real sensing-planner | +5.775 [+4.594, +6.957], p=1.8e-12 | +0.053 [+0.010, +0.096], p=0.021 |

Note: "Misses" at the reference: the t-CI includes 0 while Wilcoxon gives
p = 0.013 (skewed differences); not significant by the CI.
