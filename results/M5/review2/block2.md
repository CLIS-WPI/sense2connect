# Second external review, Block 2: new analyses (development seeds 1001-1010)

All runs on the development seeds (40 jobs); no tuning except the uncertainty-aware planner (tuning seeds 101-105).
Significance: two-sided Wilcoxon signed-rank p; effect size: mean paired difference to A5 [s/UE-min] with t-based 95 % CI. '(disagree)' marks entries where the two disagree.

## Calibrated tracker-error statistics (real map tracker, budget 4, development seeds)

| Class | track Pd | acquisition delay med/mean [CPI] | outage med/mean [CPI] | matched run med/mean [CPI] | free-mode share |
|---|---|---|---|---|---|
| bus/truck | 0.941 | 2 / 3.1 | 2 / 3.5 | 23 / 36.8 | 0.016 |
| pedestrian | 0.465 | 12 / 38.6 | 9 / 15.9 | 7 / 16.7 | 0.051 |
| car | 0.763 | 4 / 6.0 | 5 / 7.0 | 10 / 19.8 | 0.013 |

State error of matched tracks on the predictor's (pinned) state; x = along every lane / sidewalk. One CPI = 0.1 s.

| Class | comp. | mean | std | lag-1 phi | corr. time [s] | ACF lag 5 (measured / phi^5) |
|---|---|---|---|---|---|---|
| bus/truck | x | -0.02 | 1.84 | 0.81 | 0.47 | 0.38 / 0.35 |
| bus/truck | y | +0.04 | 1.18 | 0.67 | 0.25 | 0.11 / 0.14 |
| bus/truck | vx | +0.32 | 3.73 | 0.67 | 0.25 | 0.14 / 0.14 |
| bus/truck | vy | -0.04 | 0.66 | 0.56 | 0.17 | 0.13 / 0.05 |
| pedestrian | x | -0.07 | 0.53 | 0.85 | 0.61 | 0.60 / 0.44 |
| pedestrian | y | -0.00 | 0.18 | 0.81 | 0.47 | 0.45 / 0.34 |
| pedestrian | vx | -0.11 | 1.76 | 0.58 | 0.18 | 0.10 / 0.06 |
| pedestrian | vy | -0.04 | 0.57 | 0.46 | 0.13 | 0.13 / 0.02 |
| car | x | +0.01 | 1.25 | 0.89 | 0.90 | 0.62 / 0.57 |
| car | y | +0.00 | 0.13 | 0.74 | 0.32 | 0.26 / 0.21 |
| car | vx | +0.14 | 2.93 | 0.76 | 0.36 | 0.26 / 0.25 |
| car | vy | -0.03 | 0.53 | 0.61 | 0.20 | 0.19 / 0.08 |

False tracks (unmatched confirmed tracks): 13.45 per CPI, fragment share 0.308.

| Mount | type | count / CPI | births / CPI | lifetime median / mean [CPI] | episodes |
|---|---|---|---|---|---|
| facade | fragment | 4.78 | 0.414 | 6 / 11.5 | 4967 |
| facade | other | 10.09 | 0.824 | 10 / 12.2 | 9882 |
| lamppost | fragment | 3.51 | 0.449 | 4 / 7.8 | 5392 |
| lamppost | other | 8.53 | 0.562 | 10 / 15.2 | 6745 |

Generator check ('R table all' on the development jobs): tracked share of in-range reports bus/truck 0.948 (Pd 0.941), pedestrian 0.467 (Pd 0.465), car 0.747 (Pd 0.763); false tracks 13.12 per report.

## 2a Accuracy break-even sweeps (perfect tracks + one error; H fixed per margin)

H per margin: 0 dB 0.5, 5 dB 0.5, 10 dB 0.5, 15 dB 0.5, 20 dB 2.0, 25 dB 2.0, 30 dB 3.0, Ref 0.5, v1 0.5 s.

Break-even sigma (m or m/s) where the planner's advantage over A5 vanishes (mean difference reaches 0):

| Sweep | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB | Ref |
|---|---|---|---|---|---|---|
| UE position only (white, as the main runs) | none (not below A5 at 0) | 0.08 | 0.09 | 0.10 | 0.05 | 0.24 |
| realistic: blocker position, UE error 0 m | none (not below A5 at 0) | 0.08 | 0.16 | 0.12 | 0.10 | 0.57 |
| realistic: blocker position, UE error 1 m | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) |
| realistic: blocker velocity, UE error 0 m | none (not below A5 at 0) | 0.97 | > 1.0 | > 1.0 | > 1.0 | > 1.0 |
| realistic: blocker velocity, UE error 1 m | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) |
| realistic: blocker position + velocity, UE error 0 m | none (not below A5 at 0) | 0.06 | 0.12 | 0.12 | 0.11 | 0.57 |
| realistic: blocker position + velocity, UE error 1 m | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) |
| memoryless: blocker position, UE error 0 m | none (not below A5 at 0) | 0.05 | 0.08 | 0.05 | 0.08 | 0.17 |
| memoryless: blocker position, UE error 1 m | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) |
| memoryless: blocker velocity, UE error 0 m | none (not below A5 at 0) | 0.76 | 0.95 | 0.72 | 0.93 | > 1.0 |
| memoryless: blocker velocity, UE error 1 m | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) |
| memoryless: blocker position + velocity, UE error 0 m | none (not below A5 at 0) | 0.04 | 0.07 | 0.06 | 0.07 | 0.18 |
| memoryless: blocker position + velocity, UE error 1 m | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) | none (not below A5 at 0) |

Paired difference to A5 at 20 dB and the reference (mean [95 % CI], Wilcoxon p):

**UE position only (white, as the main runs)**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | -0.067 [-0.103, -0.030], p=0.00078 | -0.040 [-0.074, -0.006], p=9.5e-06 |
| 0.1 | +0.005 [-0.052, +0.061], p=0.21 | -0.031 [-0.065, +0.002], p=0.0097 (disagree) |
| 0.25 | +0.243 [+0.141, +0.345], p=1.7e-05 | +0.001 [-0.031, +0.034], p=0.58 |
| 0.5 | +0.426 [+0.266, +0.585], p=1.3e-07 | +0.020 [-0.013, +0.053], p=0.33 |
| 1 | +0.713 [+0.454, +0.972], p=5.4e-08 | +0.086 [+0.026, +0.145], p=0.0019 |

**realistic: blocker position, UE error 0 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | -0.067 [-0.103, -0.030], p=0.00078 | -0.040 [-0.074, -0.006], p=9.5e-06 |
| 0.1 | -0.024 [-0.065, +0.017], p=0.2 | -0.034 [-0.069, +0.001], p=0.0047 (disagree) |
| 0.25 | +0.041 [-0.013, +0.095], p=0.97 | -0.004 [-0.034, +0.026], p=0.71 |
| 0.5 | +0.179 [+0.084, +0.273], p=6e-05 | -0.006 [-0.044, +0.033], p=0.6 |
| 1 | +0.323 [+0.187, +0.458], p=4.3e-09 | +0.035 [+0.005, +0.065], p=0.038 |

**realistic: blocker position, UE error 1 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | +0.713 [+0.454, +0.972], p=5.4e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.1 | +0.684 [+0.451, +0.917], p=7e-08 | +0.094 [+0.028, +0.161], p=0.0038 |
| 0.25 | +0.687 [+0.475, +0.898], p=1.8e-12 | +0.090 [+0.024, +0.156], p=0.0027 |
| 0.5 | +0.796 [+0.572, +1.020], p=6e-11 | +0.086 [+0.025, +0.147], p=0.0018 |
| 1 | +0.840 [+0.575, +1.106], p=3.9e-08 | +0.097 [+0.018, +0.176], p=0.0033 |

**realistic: blocker velocity, UE error 0 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | -0.067 [-0.103, -0.030], p=0.00078 | -0.040 [-0.074, -0.006], p=9.5e-06 |
| 0.1 | -0.066 [-0.103, -0.028], p=0.00099 | -0.039 [-0.073, -0.005], p=5.5e-05 |
| 0.25 | -0.062 [-0.098, -0.026], p=0.0014 | -0.036 [-0.070, -0.002], p=7.6e-05 |
| 0.5 | -0.058 [-0.094, -0.022], p=0.0028 | -0.041 [-0.075, -0.006], p=2.7e-05 |
| 1 | -0.019 [-0.067, +0.030], p=0.14 | -0.037 [-0.071, -0.003], p=6.1e-05 |

**realistic: blocker velocity, UE error 1 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | +0.713 [+0.454, +0.972], p=5.4e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.1 | +0.712 [+0.451, +0.973], p=9.5e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.25 | +0.698 [+0.445, +0.951], p=5.4e-08 | +0.088 [+0.028, +0.147], p=0.0017 |
| 0.5 | +0.655 [+0.431, +0.879], p=5.2e-08 | +0.090 [+0.024, +0.156], p=0.002 |
| 1 | +0.730 [+0.474, +0.986], p=5.2e-08 | +0.065 [+0.017, +0.113], p=0.0094 |

**realistic: blocker position + velocity, UE error 0 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | -0.067 [-0.103, -0.030], p=0.00078 | -0.040 [-0.074, -0.006], p=9.5e-06 |
| 0.1 | -0.007 [-0.055, +0.041], p=0.25 | -0.029 [-0.064, +0.005], p=0.0019 (disagree) |
| 0.25 | +0.050 [-0.015, +0.115], p=0.98 | -0.028 [-0.063, +0.007], p=0.0023 (disagree) |
| 0.5 | +0.140 [+0.053, +0.227], p=0.00057 | -0.003 [-0.026, +0.021], p=0.93 |
| 1 | +0.308 [+0.159, +0.456], p=1.8e-07 | +0.015 [-0.005, +0.036], p=0.17 |

**realistic: blocker position + velocity, UE error 1 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | +0.713 [+0.454, +0.972], p=5.4e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.1 | +0.687 [+0.457, +0.917], p=1.3e-11 | +0.082 [+0.017, +0.147], p=0.0062 |
| 0.25 | +0.708 [+0.468, +0.947], p=4.5e-11 | +0.090 [+0.024, +0.156], p=0.0011 |
| 0.5 | +0.715 [+0.471, +0.958], p=3.1e-10 | +0.076 [+0.014, +0.138], p=0.0062 |
| 1 | +0.796 [+0.522, +1.070], p=3.5e-11 | +0.113 [+0.045, +0.181], p=0.00045 |

**memoryless: blocker position, UE error 0 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | -0.067 [-0.103, -0.030], p=0.00078 | -0.040 [-0.074, -0.006], p=9.5e-06 |
| 0.1 | +0.019 [-0.039, +0.076], p=0.63 | -0.025 [-0.061, +0.011], p=0.031 (disagree) |
| 0.25 | +0.208 [+0.096, +0.319], p=0.0008 | +0.030 [-0.006, +0.066], p=0.07 |
| 0.5 | +0.543 [+0.359, +0.728], p=1.8e-07 | +0.052 [+0.006, +0.098], p=0.013 |
| 1 | +0.854 [+0.563, +1.145], p=1.8e-12 | +0.062 [+0.012, +0.111], p=0.0099 |

**memoryless: blocker position, UE error 1 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | +0.713 [+0.454, +0.972], p=5.4e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.1 | +0.688 [+0.456, +0.921], p=4.5e-11 | +0.081 [+0.018, +0.143], p=0.0026 |
| 0.25 | +0.718 [+0.508, +0.928], p=1.3e-11 | +0.074 [+0.014, +0.134], p=0.0048 |
| 0.5 | +0.836 [+0.556, +1.117], p=5.3e-08 | +0.058 [+0.013, +0.103], p=0.0066 |
| 1 | +1.010 [+0.679, +1.341], p=3.6e-08 | +0.082 [+0.019, +0.145], p=0.00083 |

**memoryless: blocker velocity, UE error 0 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | -0.067 [-0.103, -0.030], p=0.00078 | -0.040 [-0.074, -0.006], p=9.5e-06 |
| 0.1 | -0.066 [-0.103, -0.029], p=0.0008 | -0.042 [-0.076, -0.007], p=3e-05 |
| 0.25 | -0.062 [-0.097, -0.027], p=0.00092 | -0.040 [-0.074, -0.005], p=0.00013 |
| 0.5 | -0.056 [-0.093, -0.019], p=0.0045 | -0.037 [-0.072, -0.002], p=0.0012 |
| 1 | +0.006 [-0.043, +0.055], p=0.55 | -0.030 [-0.063, +0.004], p=0.0027 (disagree) |

**memoryless: blocker velocity, UE error 1 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | +0.713 [+0.454, +0.972], p=5.4e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.1 | +0.715 [+0.463, +0.966], p=5.2e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.25 | +0.631 [+0.419, +0.844], p=4.7e-08 | +0.089 [+0.027, +0.151], p=0.0019 |
| 0.5 | +0.682 [+0.434, +0.931], p=8.2e-08 | +0.101 [+0.034, +0.169], p=0.0017 |
| 1 | +0.656 [+0.450, +0.862], p=7.6e-08 | +0.097 [+0.034, +0.160], p=0.00047 |

**memoryless: blocker position + velocity, UE error 0 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | -0.067 [-0.103, -0.030], p=0.00078 | -0.040 [-0.074, -0.006], p=9.5e-06 |
| 0.1 | +0.028 [-0.026, +0.082], p=0.94 | -0.022 [-0.053, +0.010], p=0.16 |
| 0.25 | +0.240 [+0.126, +0.354], p=2.2e-06 | +0.018 [-0.020, +0.056], p=0.12 |
| 0.5 | +0.517 [+0.326, +0.708], p=3.1e-10 | +0.056 [+0.010, +0.102], p=0.0039 |
| 1 | +0.811 [+0.539, +1.083], p=9.1e-12 | +0.053 [+0.010, +0.096], p=0.00083 |

**memoryless: blocker position + velocity, UE error 1 m**

| sigma | 20 dB | Ref |
|---|---|---|
| 0 | +0.713 [+0.454, +0.972], p=5.4e-08 | +0.086 [+0.026, +0.145], p=0.0019 |
| 0.1 | +0.692 [+0.461, +0.923], p=6e-11 | +0.069 [+0.007, +0.131], p=0.0029 |
| 0.25 | +0.658 [+0.458, +0.858], p=4e-08 | +0.069 [+0.020, +0.118], p=0.0034 |
| 0.5 | +0.710 [+0.457, +0.964], p=4.5e-08 | +0.063 [+0.015, +0.111], p=0.0042 |
| 1 | +0.972 [+0.669, +1.274], p=1.8e-12 | +0.058 [+0.019, +0.097], p=0.0012 |

## 2d Table II (error budget) with 'Perfect tracks' and 'All + UE error (1 m)'

**Realistic model (primary)**

| Column | 10 dB | Ref |
|---|---|---|
| Perfect tracks | +0.206 [+0.009, +0.403], p=0.69 (disagree) | -0.040 [-0.074, -0.006], p=9.5e-06 |
| Misses | +2.123 [+1.543, +2.704], p=1.3e-11 | +0.047 [-0.020, +0.114], p=0.12 |
| False tracks | +0.687 [+0.422, +0.953], p=2.1e-06 | -0.020 [-0.053, +0.014], p=0.81 |
| Pos./vel. error | +1.137 [+0.819, +1.454], p=3.9e-08 | +0.025 [-0.016, +0.065], p=0.22 |
| Size rule | +1.204 [+0.859, +1.549], p=1.6e-08 | +0.009 [-0.010, +0.027], p=0.38 |
| All | +4.519 [+3.617, +5.420], p=1.8e-12 | +0.045 [-0.006, +0.097], p=0.015 (disagree) |
| All + UE error (1 m) | +6.240 [+5.065, +7.416], p=1.8e-12 | +0.059 [+0.017, +0.102], p=0.0012 |
| Real sensing-planner (own tuned H) | +5.775 [+4.594, +6.957], p=1.8e-12 | +0.053 [+0.010, +0.096], p=0.021 |

**Memoryless model (limiting case; = first-review B5 at fixed H)**

| Column | 10 dB | Ref |
|---|---|---|
| Perfect tracks | +0.206 [+0.009, +0.403], p=0.69 (disagree) | -0.040 [-0.074, -0.006], p=9.5e-06 |
| Misses | +0.712 [+0.433, +0.991], p=4.3e-09 | -0.005 [-0.031, +0.022], p=0.97 |
| False tracks | +1.109 [+0.941, +1.276], p=1.8e-12 | +0.067 [+0.033, +0.100], p=6.2e-06 |
| Pos./vel. error | +1.470 [+1.181, +1.759], p=1.8e-12 | +0.045 [+0.010, +0.080], p=0.0027 |
| All | +5.199 [+4.417, +5.982], p=1.8e-12 | +0.085 [+0.047, +0.124], p=4.2e-06 |
| All + UE error (1 m) | +7.092 [+5.821, +8.363], p=1.8e-12 | +0.059 [+0.017, +0.100], p=0.0023 |
| Real sensing-planner (own tuned H) | +5.775 [+4.594, +6.957], p=1.8e-12 | +0.053 [+0.010, +0.096], p=0.021 |

Realistic Table II at every margin (mean difference, Wilcoxon p):

| Column | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB | Ref |
|---|---|---|---|---|---|---|
| Perfect | +0.206 (0.69) | -0.042 (0.01) | -0.067 (0.00078) | -0.026 (0.0085) | -0.014 (0.013) | -0.040 (9.5e-06) |
| Misses | +2.123 (1.3e-11) | +1.042 (1.9e-09) | +0.604 (1.2e-08) | +0.260 (3.8e-05) | +0.245 (5e-06) | +0.047 (0.12) |
| False | +0.687 (2.1e-06) | +0.204 (4.9e-06) | +0.121 (0.0012) | +0.092 (4e-05) | +0.037 (0.0088) | -0.020 (0.81) |
| Pos./vel. | +1.137 (3.9e-08) | +0.667 (1.8e-12) | +0.433 (9.1e-12) | +0.231 (2.7e-06) | +0.116 (9.1e-05) | +0.025 (0.22) |
| Size rule | +1.204 (1.6e-08) | +0.450 (6.5e-05) | +0.313 (0.0021) | +0.177 (0.00032) | +0.082 (0.0013) | +0.009 (0.38) |
| All | +4.519 (1.8e-12) | +2.016 (1.8e-12) | +1.068 (3.6e-08) | +0.678 (6.6e-08) | +0.275 (4.7e-06) | +0.045 (0.015) |
| All + UE 1 m | +6.240 (1.8e-12) | +2.863 (1.8e-12) | +1.499 (1.8e-12) | +0.833 (1.8e-12) | +0.355 (9.8e-07) | +0.059 (0.0012) |
| Real sensing-planner | +5.775 (1.8e-12) | +2.786 (1.8e-12) | +1.396 (3.9e-08) | +0.794 (3.8e-08) | +0.341 (1.9e-06) | +0.053 (0.021) |

## 2b Uncertainty-aware planner (A5 default; tuned K, alpha, theta on tuning seeds)

Zero-noise check vs the cached sensing-planner predictions: 4.5e-05 dB.

| Margin | K | alpha | theta [s] | H [s] | budget | outage | UA - A5 [95 % CI], p | HO/UE-min |
|---|---|---|---|---|---|---|---|---|
| 0 dB | 8 | 1 | 0.01 | 0.5 | 2 | 36.048 | +1.3835 [+1.2064, +1.5605], p=1.8e-12 | 1.50 |
| 5 dB | 8 | 1 | 0.3 | 1 | 4 | 10.600 | +1.3732 [+1.0396, +1.7068], p=3.6e-08 | 3.52 |
| 10 dB | 8 | 1 | 0.3 | 0.5 | 4 | 3.572 | +0.2798 [+0.1613, +0.3983], p=3.6e-08 | 5.92 |
| 15 dB | 8 | 1 | 0.3 | 0.5 | 4 | 2.036 | +0.0377 [+0.0208, +0.0545], p=1.6e-07 | 6.90 |
| 20 dB | 8 | 1 | 0.3 | 1 | 2 | 1.219 | +0.0391 [+0.0199, +0.0583], p=3.6e-07 | 6.58 |
| 25 dB | 8 | 4 | 0.1 | 3 | 4 | 0.478 | +0.0159 [+0.0066, +0.0252], p=0.00036 | 4.02 |
| 30 dB | 8 | 1 | 0.3 | 2 | 4 | 0.180 | +0.0110 [+0.0050, +0.0170], p=1.8e-05 | 1.79 |
| Ref | 8 | 1 | 0.1 | 2 | 4 | 0.082 | +0.0029 [+0.0010, +0.0048], p=0.0022 | 0.76 |
| v1 | 32 | 16 | 0.03 | 0.5 | 2 | 0.050 | +0.0108 [+0.0023, +0.0193], p=0.01 | 0.96 |

## Notes (hand-written; numbers above are generated)
- Realistic "All + UE error (1 m)" reproduces the real sensing-planner at every margin (10 dB +6.24 vs +5.78; Ref +0.059 vs +0.053), so the calibrated model closes the error budget. The memoryless model does too at 10 dB (+7.09), but it misattributes the error: misses matter much more when they come as whole-track outages (+2.12 vs +0.71 at 10 dB), false tracks less.
- New error source: the predictor's class-agnostic size rule (a car predicted as a bus box) costs +1.20 at 10 dB on its own. It was not in Table II before.
- Break-even: the UE position error is the binding constraint. With perfect tracks the advantage vanishes at a UE sigma of about 0.1 m (15-30 dB), and at 1 m no blocker accuracy restores it. Blocker position break-even (realistic, UE exact) is 0.08-0.16 m at 15-30 dB and 0.57 m at Ref. Blocker velocity matters little: the advantage survives up to 1 m/s at 20-30 dB.
- 2b: after tuning, the uncertainty-aware planner (almost) never intervenes. The tuning picks the largest theta, and at 5-30 dB the outage equals "A5 + sensing overhead" exactly (results/M5/review_b6_diag.json). Its significant deficit to A5 is the sensing overhead.
- 2c estimate (second radar at oru-1): new RCS traces for 20 tuning + 40 development jobs, about 1.5 h on GPU 1 with 4 workers (measured 285 s per job single-process, x3.1 with 4 workers), or about 2.6 h including the 40 held-out jobs; longer under the current server load. Code: the radar O-RU is hard-coded in sim/sensing/radar.py (orus[0]); editing that file changes the sensing-trace provenance hash and would refuse the existing caches. A separate radar builder avoids that; the alternative is a restamp, which needs approval. Also needed: the image-method ghost handling for the new position, the detector operating point (reuse budget 4, or re-tune on tuning seeds, about 1 h CPU), a second map tracker, and track-level fusion of the two radars (new module and tests). Then re-run calibration, the sensing-planner, the uncertainty-aware planner and the sweeps (about 2 h). Total: roughly one working day of implementation and 4-6 h of compute. It has to happen before the v1.1 freeze if it is wanted.
- fig_onset: loss curves now at 1 ms (analytic model B, same evaluation as review_a34); legend says "1 ms resolution"; the printed onset is the 1 ms value (bus/truck 694 ms, pedestrian 1 ms).
