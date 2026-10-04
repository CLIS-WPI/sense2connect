# M2 report

Status: ready for review. This milestone is not marked done. No operating point is selected.

Regenerate this table from the channel caches with `docker compose run --rm sionna python -u scripts/run_m2_review.py --workers 4`. The loader refuses a cache whose git commit, uncommitted-change hash, or config hash differs.

git 216eb537d3b157f9591e062c29ec6bd24067d81a dirty 6ac8fbeb374bb5c9031b85c72a835427ad75b4845f58a9f3dbbda1afe572b2a1 config b75bb5e6c91c85c16492d6ca67ac736a9384188a6e1593d4d5fec2c3fa449c7c

Runs are 60 s at dt = 0.1 s, one CPI per snapshot. The tracker is an EKF on range, azimuth, elevation, and radial velocity. Velocity starts from Doppler and, on the second hit, two-point differencing. Velocity RMSE is after 1 s of track age.

The constraint of at most 2 unmatched clusters per CPI is infeasible on the tuning seeds. It is not changed. Unmatched clusters are split into fragments (inside a true target's bounding box expanded by 1 m), ghosts (mirror of that box across a known wall), and other. Only ghosts and other count as false alarms. Budgets of 2, 4 and 6 false alarms per CPI are the best tuning point under each budget, labelled as such. The operating point waits on the false-handover cost.

Blind subtraction removes the slow-time mean. Twin subtraction removes the empty-scene path. No impairment was added.
Largest absolute power difference between the blind and twin maps on the first snapshot of each evaluation job, 1024 subcarriers: 1.442e-10.

Match gates are horizontal distances from the mesh origin: pedestrian 1.5 m, car 3.0 m, bus/truck 6.0 m. The 0.5× and 2× rows are sensitivity only. Mean scattering-point offset from the mesh origin, over the lamppost targets: max 1.967 m, mean 0.868 m. Gates use the mesh origin.

## Tuning trade-off, all grid points

Seeds 101–105, 1024 subcarriers. False alarms are ghost clusters plus other clusters per CPI.

| Clutter | Method | Pfa | Train | Eps [m] | Coast | Assoc. [m] | Bus/truck track Pd | Unmatched / CPI | Fragment | Ghost | Other | False alarms / CPI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blind | none | 1e-03 | 4 | 3 | 2 | 6 | 0.830 | 6.14 | 1.52 | 0.14 | 4.49 | 4.62 |
| blind | none | 1e-03 | 4 | 3 | 2 | 10 | 0.818 | 6.14 | 1.52 | 0.14 | 4.49 | 4.62 |
| blind | none | 1e-03 | 4 | 3 | 4 | 6 | 0.871 | 6.14 | 1.52 | 0.14 | 4.49 | 4.62 |
| blind | none | 1e-03 | 4 | 3 | 4 | 10 | 0.841 | 6.14 | 1.52 | 0.14 | 4.49 | 4.62 |
| blind | none | 1e-03 | 4 | 5 | 2 | 6 | 0.646 | 3.85 | 0.59 | 0.10 | 3.16 | 3.26 |
| blind | none | 1e-03 | 4 | 5 | 2 | 10 | 0.642 | 3.85 | 0.59 | 0.10 | 3.16 | 3.26 |
| blind | none | 1e-03 | 4 | 5 | 4 | 6 | 0.709 | 3.85 | 0.59 | 0.10 | 3.16 | 3.26 |
| blind | none | 1e-03 | 4 | 5 | 4 | 10 | 0.680 | 3.85 | 0.59 | 0.10 | 3.16 | 3.26 |
| blind | none | 1e-04 | 4 | 3 | 2 | 6 | 0.791 | 5.54 | 1.37 | 0.12 | 4.06 | 4.18 |
| blind | none | 1e-04 | 4 | 3 | 2 | 10 | 0.782 | 5.54 | 1.37 | 0.12 | 4.06 | 4.18 |
| blind | none | 1e-04 | 4 | 3 | 4 | 6 | 0.841 | 5.54 | 1.37 | 0.12 | 4.06 | 4.18 |
| blind | none | 1e-04 | 4 | 3 | 4 | 10 | 0.811 | 5.54 | 1.37 | 0.12 | 4.06 | 4.18 |
| blind | none | 1e-04 | 4 | 5 | 2 | 6 | 0.633 | 3.61 | 0.56 | 0.08 | 2.96 | 3.05 |
| blind | none | 1e-04 | 4 | 5 | 2 | 10 | 0.623 | 3.61 | 0.56 | 0.08 | 2.96 | 3.05 |
| blind | none | 1e-04 | 4 | 5 | 4 | 6 | 0.690 | 3.61 | 0.56 | 0.08 | 2.96 | 3.05 |
| blind | none | 1e-04 | 4 | 5 | 4 | 10 | 0.665 | 3.61 | 0.56 | 0.08 | 2.96 | 3.05 |
| blind | none | 1e-05 | 4 | 3 | 2 | 6 | 0.759 | 5.10 | 1.26 | 0.10 | 3.73 | 3.84 |
| blind | none | 1e-05 | 4 | 3 | 2 | 10 | 0.749 | 5.10 | 1.26 | 0.10 | 3.73 | 3.84 |
| blind | none | 1e-05 | 4 | 3 | 4 | 6 | 0.814 | 5.10 | 1.26 | 0.10 | 3.73 | 3.84 |
| blind | none | 1e-05 | 4 | 3 | 4 | 10 | 0.783 | 5.10 | 1.26 | 0.10 | 3.73 | 3.84 |
| blind | none | 1e-05 | 4 | 5 | 2 | 6 | 0.622 | 3.43 | 0.54 | 0.08 | 2.81 | 2.89 |
| blind | none | 1e-05 | 4 | 5 | 2 | 10 | 0.617 | 3.43 | 0.54 | 0.08 | 2.81 | 2.89 |
| blind | none | 1e-05 | 4 | 5 | 4 | 6 | 0.680 | 3.43 | 0.54 | 0.08 | 2.81 | 2.89 |
| blind | none | 1e-05 | 4 | 5 | 4 | 10 | 0.660 | 3.43 | 0.54 | 0.08 | 2.81 | 2.89 |
| blind | none | 1e-03 | 8 | 3 | 2 | 6 | 0.728 | 5.22 | 1.20 | 0.07 | 3.95 | 4.02 |
| blind | none | 1e-03 | 8 | 3 | 2 | 10 | 0.724 | 5.22 | 1.20 | 0.07 | 3.95 | 4.02 |
| blind | none | 1e-03 | 8 | 3 | 4 | 6 | 0.781 | 5.22 | 1.20 | 0.07 | 3.95 | 4.02 |
| blind | none | 1e-03 | 8 | 3 | 4 | 10 | 0.750 | 5.22 | 1.20 | 0.07 | 3.95 | 4.02 |
| blind | none | 1e-03 | 8 | 5 | 2 | 6 | 0.579 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| blind | none | 1e-03 | 8 | 5 | 2 | 10 | 0.576 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| blind | none | 1e-03 | 8 | 5 | 4 | 6 | 0.651 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| blind | none | 1e-03 | 8 | 5 | 4 | 10 | 0.623 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| blind | none | 1e-04 | 8 | 3 | 2 | 6 | 0.675 | 4.66 | 1.04 | 0.06 | 3.57 | 3.63 |
| blind | none | 1e-04 | 8 | 3 | 2 | 10 | 0.675 | 4.66 | 1.04 | 0.06 | 3.57 | 3.63 |
| blind | none | 1e-04 | 8 | 3 | 4 | 6 | 0.733 | 4.66 | 1.04 | 0.06 | 3.57 | 3.63 |
| blind | none | 1e-04 | 8 | 3 | 4 | 10 | 0.711 | 4.66 | 1.04 | 0.06 | 3.57 | 3.63 |
| blind | none | 1e-04 | 8 | 5 | 2 | 6 | 0.554 | 3.16 | 0.46 | 0.04 | 2.65 | 2.70 |
| blind | none | 1e-04 | 8 | 5 | 2 | 10 | 0.556 | 3.16 | 0.46 | 0.04 | 2.65 | 2.70 |
| blind | none | 1e-04 | 8 | 5 | 4 | 6 | 0.625 | 3.16 | 0.46 | 0.04 | 2.65 | 2.70 |
| blind | none | 1e-04 | 8 | 5 | 4 | 10 | 0.603 | 3.16 | 0.46 | 0.04 | 2.65 | 2.70 |
| blind | none | 1e-05 | 8 | 3 | 2 | 6 | 0.634 | 4.26 | 0.93 | 0.05 | 3.29 | 3.34 |
| blind | none | 1e-05 | 8 | 3 | 2 | 10 | 0.629 | 4.26 | 0.93 | 0.05 | 3.29 | 3.34 |
| blind | none | 1e-05 | 8 | 3 | 4 | 6 | 0.694 | 4.26 | 0.93 | 0.05 | 3.29 | 3.34 |
| blind | none | 1e-05 | 8 | 3 | 4 | 10 | 0.674 | 4.26 | 0.93 | 0.05 | 3.29 | 3.34 |
| blind | none | 1e-05 | 8 | 5 | 2 | 6 | 0.529 | 2.96 | 0.43 | 0.04 | 2.49 | 2.53 |
| blind | none | 1e-05 | 8 | 5 | 2 | 10 | 0.531 | 2.96 | 0.43 | 0.04 | 2.49 | 2.53 |
| blind | none | 1e-05 | 8 | 5 | 4 | 6 | 0.594 | 2.96 | 0.43 | 0.04 | 2.49 | 2.53 |
| blind | none | 1e-05 | 8 | 5 | 4 | 10 | 0.568 | 2.96 | 0.43 | 0.04 | 2.49 | 2.53 |
| blind | image | 1e-03 | 4 | 3 | 2 | 6 | 0.829 | 5.30 | 1.51 | 0.05 | 3.74 | 3.79 |
| blind | image | 1e-03 | 4 | 3 | 2 | 10 | 0.795 | 4.04 | 1.41 | 0.02 | 2.61 | 2.63 |
| blind | image | 1e-03 | 4 | 3 | 4 | 6 | 0.870 | 5.30 | 1.51 | 0.05 | 3.74 | 3.79 |
| blind | image | 1e-03 | 4 | 3 | 4 | 10 | 0.821 | 4.04 | 1.41 | 0.02 | 2.61 | 2.63 |
| blind | image | 1e-03 | 4 | 5 | 2 | 6 | 0.645 | 3.39 | 0.59 | 0.05 | 2.75 | 2.80 |
| blind | image | 1e-03 | 4 | 5 | 2 | 10 | 0.633 | 2.55 | 0.57 | 0.02 | 1.96 | 1.97 |
| blind | image | 1e-03 | 4 | 5 | 4 | 6 | 0.707 | 3.39 | 0.59 | 0.05 | 2.75 | 2.80 |
| blind | image | 1e-03 | 4 | 5 | 4 | 10 | 0.678 | 2.55 | 0.57 | 0.02 | 1.96 | 1.97 |
| blind | image | 1e-04 | 4 | 3 | 2 | 6 | 0.791 | 4.87 | 1.36 | 0.05 | 3.46 | 3.51 |
| blind | image | 1e-04 | 4 | 3 | 2 | 10 | 0.766 | 3.78 | 1.29 | 0.02 | 2.48 | 2.50 |
| blind | image | 1e-04 | 4 | 3 | 4 | 6 | 0.837 | 4.87 | 1.36 | 0.05 | 3.46 | 3.51 |
| blind | image | 1e-04 | 4 | 3 | 4 | 10 | 0.797 | 3.78 | 1.29 | 0.02 | 2.48 | 2.50 |
| blind | image | 1e-04 | 4 | 5 | 2 | 6 | 0.635 | 3.23 | 0.56 | 0.05 | 2.62 | 2.67 |
| blind | image | 1e-04 | 4 | 5 | 2 | 10 | 0.621 | 2.48 | 0.54 | 0.02 | 1.91 | 1.93 |
| blind | image | 1e-04 | 4 | 5 | 4 | 6 | 0.689 | 3.23 | 0.56 | 0.05 | 2.62 | 2.67 |
| blind | image | 1e-04 | 4 | 5 | 4 | 10 | 0.664 | 2.48 | 0.54 | 0.02 | 1.91 | 1.93 |
| blind | image | 1e-05 | 4 | 3 | 2 | 6 | 0.761 | 4.54 | 1.26 | 0.05 | 3.24 | 3.29 |
| blind | image | 1e-05 | 4 | 3 | 2 | 10 | 0.742 | 3.58 | 1.19 | 0.02 | 2.37 | 2.38 |
| blind | image | 1e-05 | 4 | 3 | 4 | 6 | 0.811 | 4.54 | 1.26 | 0.05 | 3.24 | 3.29 |
| blind | image | 1e-05 | 4 | 3 | 4 | 10 | 0.771 | 3.58 | 1.19 | 0.02 | 2.37 | 2.38 |
| blind | image | 1e-05 | 4 | 5 | 2 | 6 | 0.622 | 3.11 | 0.54 | 0.05 | 2.53 | 2.57 |
| blind | image | 1e-05 | 4 | 5 | 2 | 10 | 0.612 | 2.42 | 0.52 | 0.02 | 1.88 | 1.90 |
| blind | image | 1e-05 | 4 | 5 | 4 | 6 | 0.677 | 3.11 | 0.54 | 0.05 | 2.53 | 2.57 |
| blind | image | 1e-05 | 4 | 5 | 4 | 10 | 0.646 | 2.42 | 0.52 | 0.02 | 1.88 | 1.90 |
| blind | image | 1e-03 | 8 | 3 | 2 | 6 | 0.727 | 4.54 | 1.19 | 0.03 | 3.31 | 3.35 |
| blind | image | 1e-03 | 8 | 3 | 2 | 10 | 0.696 | 3.50 | 1.09 | 0.02 | 2.39 | 2.41 |
| blind | image | 1e-03 | 8 | 3 | 4 | 6 | 0.781 | 4.54 | 1.19 | 0.03 | 3.31 | 3.35 |
| blind | image | 1e-03 | 8 | 3 | 4 | 10 | 0.734 | 3.50 | 1.09 | 0.02 | 2.39 | 2.41 |
| blind | image | 1e-03 | 8 | 5 | 2 | 6 | 0.582 | 3.04 | 0.50 | 0.03 | 2.51 | 2.55 |
| blind | image | 1e-03 | 8 | 5 | 2 | 10 | 0.568 | 2.39 | 0.48 | 0.02 | 1.89 | 1.91 |
| blind | image | 1e-03 | 8 | 5 | 4 | 6 | 0.653 | 3.04 | 0.50 | 0.03 | 2.51 | 2.55 |
| blind | image | 1e-03 | 8 | 5 | 4 | 10 | 0.612 | 2.39 | 0.48 | 0.02 | 1.89 | 1.91 |
| blind | image | 1e-04 | 8 | 3 | 2 | 6 | 0.673 | 4.13 | 1.03 | 0.03 | 3.07 | 3.10 |
| blind | image | 1e-04 | 8 | 3 | 2 | 10 | 0.644 | 3.24 | 0.96 | 0.02 | 2.27 | 2.29 |
| blind | image | 1e-04 | 8 | 3 | 4 | 6 | 0.733 | 4.13 | 1.03 | 0.03 | 3.07 | 3.10 |
| blind | image | 1e-04 | 8 | 3 | 4 | 10 | 0.690 | 3.24 | 0.96 | 0.02 | 2.27 | 2.29 |
| blind | image | 1e-04 | 8 | 5 | 2 | 6 | 0.556 | 2.87 | 0.46 | 0.03 | 2.38 | 2.41 |
| blind | image | 1e-04 | 8 | 5 | 2 | 10 | 0.540 | 2.29 | 0.45 | 0.02 | 1.83 | 1.85 |
| blind | image | 1e-04 | 8 | 5 | 4 | 6 | 0.623 | 2.87 | 0.46 | 0.03 | 2.38 | 2.41 |
| blind | image | 1e-04 | 8 | 5 | 4 | 10 | 0.584 | 2.29 | 0.45 | 0.02 | 1.83 | 1.85 |
| blind | image | 1e-05 | 8 | 3 | 2 | 6 | 0.631 | 3.82 | 0.92 | 0.03 | 2.87 | 2.90 |
| blind | image | 1e-05 | 8 | 3 | 2 | 10 | 0.612 | 3.05 | 0.86 | 0.02 | 2.17 | 2.18 |
| blind | image | 1e-05 | 8 | 3 | 4 | 6 | 0.688 | 3.82 | 0.92 | 0.03 | 2.87 | 2.90 |
| blind | image | 1e-05 | 8 | 3 | 4 | 10 | 0.654 | 3.05 | 0.86 | 0.02 | 2.17 | 2.18 |
| blind | image | 1e-05 | 8 | 5 | 2 | 6 | 0.527 | 2.72 | 0.43 | 0.03 | 2.26 | 2.29 |
| blind | image | 1e-05 | 8 | 5 | 2 | 10 | 0.516 | 2.21 | 0.42 | 0.02 | 1.77 | 1.79 |
| blind | image | 1e-05 | 8 | 5 | 4 | 6 | 0.594 | 2.72 | 0.43 | 0.03 | 2.26 | 2.29 |
| blind | image | 1e-05 | 8 | 5 | 4 | 10 | 0.561 | 2.21 | 0.42 | 0.02 | 1.77 | 1.79 |
| twin | none | 1e-03 | 4 | 3 | 2 | 6 | 0.831 | 6.12 | 1.52 | 0.14 | 4.47 | 4.61 |
| twin | none | 1e-03 | 4 | 3 | 2 | 10 | 0.819 | 6.12 | 1.52 | 0.14 | 4.47 | 4.61 |
| twin | none | 1e-03 | 4 | 3 | 4 | 6 | 0.870 | 6.12 | 1.52 | 0.14 | 4.47 | 4.61 |
| twin | none | 1e-03 | 4 | 3 | 4 | 10 | 0.838 | 6.12 | 1.52 | 0.14 | 4.47 | 4.61 |
| twin | none | 1e-03 | 4 | 5 | 2 | 6 | 0.643 | 3.84 | 0.59 | 0.10 | 3.15 | 3.25 |
| twin | none | 1e-03 | 4 | 5 | 2 | 10 | 0.646 | 3.84 | 0.59 | 0.10 | 3.15 | 3.25 |
| twin | none | 1e-03 | 4 | 5 | 4 | 6 | 0.705 | 3.84 | 0.59 | 0.10 | 3.15 | 3.25 |
| twin | none | 1e-03 | 4 | 5 | 4 | 10 | 0.682 | 3.84 | 0.59 | 0.10 | 3.15 | 3.25 |
| twin | none | 1e-04 | 4 | 3 | 2 | 6 | 0.790 | 5.53 | 1.37 | 0.12 | 4.04 | 4.16 |
| twin | none | 1e-04 | 4 | 3 | 2 | 10 | 0.782 | 5.53 | 1.37 | 0.12 | 4.04 | 4.16 |
| twin | none | 1e-04 | 4 | 3 | 4 | 6 | 0.840 | 5.53 | 1.37 | 0.12 | 4.04 | 4.16 |
| twin | none | 1e-04 | 4 | 3 | 4 | 10 | 0.809 | 5.53 | 1.37 | 0.12 | 4.04 | 4.16 |
| twin | none | 1e-04 | 4 | 5 | 2 | 6 | 0.631 | 3.60 | 0.56 | 0.08 | 2.95 | 3.04 |
| twin | none | 1e-04 | 4 | 5 | 2 | 10 | 0.629 | 3.60 | 0.56 | 0.08 | 2.95 | 3.04 |
| twin | none | 1e-04 | 4 | 5 | 4 | 6 | 0.691 | 3.60 | 0.56 | 0.08 | 2.95 | 3.04 |
| twin | none | 1e-04 | 4 | 5 | 4 | 10 | 0.666 | 3.60 | 0.56 | 0.08 | 2.95 | 3.04 |
| twin | none | 1e-05 | 4 | 3 | 2 | 6 | 0.758 | 5.08 | 1.26 | 0.10 | 3.72 | 3.82 |
| twin | none | 1e-05 | 4 | 3 | 2 | 10 | 0.753 | 5.08 | 1.26 | 0.10 | 3.72 | 3.82 |
| twin | none | 1e-05 | 4 | 3 | 4 | 6 | 0.807 | 5.08 | 1.26 | 0.10 | 3.72 | 3.82 |
| twin | none | 1e-05 | 4 | 3 | 4 | 10 | 0.780 | 5.08 | 1.26 | 0.10 | 3.72 | 3.82 |
| twin | none | 1e-05 | 4 | 5 | 2 | 6 | 0.618 | 3.42 | 0.54 | 0.08 | 2.80 | 2.88 |
| twin | none | 1e-05 | 4 | 5 | 2 | 10 | 0.612 | 3.42 | 0.54 | 0.08 | 2.80 | 2.88 |
| twin | none | 1e-05 | 4 | 5 | 4 | 6 | 0.681 | 3.42 | 0.54 | 0.08 | 2.80 | 2.88 |
| twin | none | 1e-05 | 4 | 5 | 4 | 10 | 0.653 | 3.42 | 0.54 | 0.08 | 2.80 | 2.88 |
| twin | none | 1e-03 | 8 | 3 | 2 | 6 | 0.729 | 5.21 | 1.20 | 0.07 | 3.94 | 4.01 |
| twin | none | 1e-03 | 8 | 3 | 2 | 10 | 0.721 | 5.21 | 1.20 | 0.07 | 3.94 | 4.01 |
| twin | none | 1e-03 | 8 | 3 | 4 | 6 | 0.779 | 5.21 | 1.20 | 0.07 | 3.94 | 4.01 |
| twin | none | 1e-03 | 8 | 3 | 4 | 10 | 0.755 | 5.21 | 1.20 | 0.07 | 3.94 | 4.01 |
| twin | none | 1e-03 | 8 | 5 | 2 | 6 | 0.577 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| twin | none | 1e-03 | 8 | 5 | 2 | 10 | 0.581 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| twin | none | 1e-03 | 8 | 5 | 4 | 6 | 0.649 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| twin | none | 1e-03 | 8 | 5 | 4 | 10 | 0.625 | 3.39 | 0.49 | 0.05 | 2.85 | 2.90 |
| twin | none | 1e-04 | 8 | 3 | 2 | 6 | 0.675 | 4.66 | 1.04 | 0.06 | 3.56 | 3.62 |
| twin | none | 1e-04 | 8 | 3 | 2 | 10 | 0.672 | 4.66 | 1.04 | 0.06 | 3.56 | 3.62 |
| twin | none | 1e-04 | 8 | 3 | 4 | 6 | 0.732 | 4.66 | 1.04 | 0.06 | 3.56 | 3.62 |
| twin | none | 1e-04 | 8 | 3 | 4 | 10 | 0.702 | 4.66 | 1.04 | 0.06 | 3.56 | 3.62 |
| twin | none | 1e-04 | 8 | 5 | 2 | 6 | 0.554 | 3.15 | 0.46 | 0.04 | 2.65 | 2.69 |
| twin | none | 1e-04 | 8 | 5 | 2 | 10 | 0.554 | 3.15 | 0.46 | 0.04 | 2.65 | 2.69 |
| twin | none | 1e-04 | 8 | 5 | 4 | 6 | 0.622 | 3.15 | 0.46 | 0.04 | 2.65 | 2.69 |
| twin | none | 1e-04 | 8 | 5 | 4 | 10 | 0.598 | 3.15 | 0.46 | 0.04 | 2.65 | 2.69 |
| twin | none | 1e-05 | 8 | 3 | 2 | 6 | 0.635 | 4.26 | 0.93 | 0.05 | 3.28 | 3.33 |
| twin | none | 1e-05 | 8 | 3 | 2 | 10 | 0.632 | 4.26 | 0.93 | 0.05 | 3.28 | 3.33 |
| twin | none | 1e-05 | 8 | 3 | 4 | 6 | 0.692 | 4.26 | 0.93 | 0.05 | 3.28 | 3.33 |
| twin | none | 1e-05 | 8 | 3 | 4 | 10 | 0.671 | 4.26 | 0.93 | 0.05 | 3.28 | 3.33 |
| twin | none | 1e-05 | 8 | 5 | 2 | 6 | 0.530 | 2.95 | 0.43 | 0.04 | 2.48 | 2.52 |
| twin | none | 1e-05 | 8 | 5 | 2 | 10 | 0.531 | 2.95 | 0.43 | 0.04 | 2.48 | 2.52 |
| twin | none | 1e-05 | 8 | 5 | 4 | 6 | 0.594 | 2.95 | 0.43 | 0.04 | 2.48 | 2.52 |
| twin | none | 1e-05 | 8 | 5 | 4 | 10 | 0.576 | 2.95 | 0.43 | 0.04 | 2.48 | 2.52 |
| twin | image | 1e-03 | 4 | 3 | 2 | 6 | 0.828 | 5.29 | 1.51 | 0.05 | 3.73 | 3.78 |
| twin | image | 1e-03 | 4 | 3 | 2 | 10 | 0.801 | 4.04 | 1.41 | 0.02 | 2.61 | 2.63 |
| twin | image | 1e-03 | 4 | 3 | 4 | 6 | 0.871 | 5.29 | 1.51 | 0.05 | 3.73 | 3.78 |
| twin | image | 1e-03 | 4 | 3 | 4 | 10 | 0.825 | 4.04 | 1.41 | 0.02 | 2.61 | 2.63 |
| twin | image | 1e-03 | 4 | 5 | 2 | 6 | 0.643 | 3.38 | 0.59 | 0.05 | 2.74 | 2.79 |
| twin | image | 1e-03 | 4 | 5 | 2 | 10 | 0.630 | 2.55 | 0.58 | 0.02 | 1.95 | 1.97 |
| twin | image | 1e-03 | 4 | 5 | 4 | 6 | 0.705 | 3.38 | 0.59 | 0.05 | 2.74 | 2.79 |
| twin | image | 1e-03 | 4 | 5 | 4 | 10 | 0.675 | 2.55 | 0.58 | 0.02 | 1.95 | 1.97 |
| twin | image | 1e-04 | 4 | 3 | 2 | 6 | 0.792 | 4.85 | 1.36 | 0.05 | 3.45 | 3.50 |
| twin | image | 1e-04 | 4 | 3 | 2 | 10 | 0.768 | 3.78 | 1.28 | 0.02 | 2.48 | 2.49 |
| twin | image | 1e-04 | 4 | 3 | 4 | 6 | 0.836 | 4.85 | 1.36 | 0.05 | 3.45 | 3.50 |
| twin | image | 1e-04 | 4 | 3 | 4 | 10 | 0.800 | 3.78 | 1.28 | 0.02 | 2.48 | 2.49 |
| twin | image | 1e-04 | 4 | 5 | 2 | 6 | 0.633 | 3.22 | 0.56 | 0.05 | 2.61 | 2.66 |
| twin | image | 1e-04 | 4 | 5 | 2 | 10 | 0.622 | 2.47 | 0.54 | 0.02 | 1.91 | 1.93 |
| twin | image | 1e-04 | 4 | 5 | 4 | 6 | 0.692 | 3.22 | 0.56 | 0.05 | 2.61 | 2.66 |
| twin | image | 1e-04 | 4 | 5 | 4 | 10 | 0.661 | 2.47 | 0.54 | 0.02 | 1.91 | 1.93 |
| twin | image | 1e-05 | 4 | 3 | 2 | 6 | 0.757 | 4.53 | 1.25 | 0.05 | 3.23 | 3.28 |
| twin | image | 1e-05 | 4 | 3 | 2 | 10 | 0.739 | 3.57 | 1.19 | 0.02 | 2.37 | 2.38 |
| twin | image | 1e-05 | 4 | 3 | 4 | 6 | 0.808 | 4.53 | 1.25 | 0.05 | 3.23 | 3.28 |
| twin | image | 1e-05 | 4 | 3 | 4 | 10 | 0.770 | 3.57 | 1.19 | 0.02 | 2.37 | 2.38 |
| twin | image | 1e-05 | 4 | 5 | 2 | 6 | 0.621 | 3.10 | 0.54 | 0.04 | 2.52 | 2.56 |
| twin | image | 1e-05 | 4 | 5 | 2 | 10 | 0.605 | 2.42 | 0.52 | 0.02 | 1.88 | 1.89 |
| twin | image | 1e-05 | 4 | 5 | 4 | 6 | 0.676 | 3.10 | 0.54 | 0.04 | 2.52 | 2.56 |
| twin | image | 1e-05 | 4 | 5 | 4 | 10 | 0.647 | 2.42 | 0.52 | 0.02 | 1.88 | 1.89 |
| twin | image | 1e-03 | 8 | 3 | 2 | 6 | 0.726 | 4.53 | 1.19 | 0.03 | 3.31 | 3.34 |
| twin | image | 1e-03 | 8 | 3 | 2 | 10 | 0.693 | 3.49 | 1.09 | 0.02 | 2.38 | 2.40 |
| twin | image | 1e-03 | 8 | 3 | 4 | 6 | 0.778 | 4.53 | 1.19 | 0.03 | 3.31 | 3.34 |
| twin | image | 1e-03 | 8 | 3 | 4 | 10 | 0.732 | 3.49 | 1.09 | 0.02 | 2.38 | 2.40 |
| twin | image | 1e-03 | 8 | 5 | 2 | 6 | 0.581 | 3.04 | 0.50 | 0.03 | 2.51 | 2.55 |
| twin | image | 1e-03 | 8 | 5 | 2 | 10 | 0.570 | 2.39 | 0.48 | 0.02 | 1.89 | 1.91 |
| twin | image | 1e-03 | 8 | 5 | 4 | 6 | 0.652 | 3.04 | 0.50 | 0.03 | 2.51 | 2.55 |
| twin | image | 1e-03 | 8 | 5 | 4 | 10 | 0.606 | 2.39 | 0.48 | 0.02 | 1.89 | 1.91 |
| twin | image | 1e-04 | 8 | 3 | 2 | 6 | 0.673 | 4.12 | 1.03 | 0.03 | 3.06 | 3.09 |
| twin | image | 1e-04 | 8 | 3 | 2 | 10 | 0.648 | 3.24 | 0.96 | 0.02 | 2.26 | 2.28 |
| twin | image | 1e-04 | 8 | 3 | 4 | 6 | 0.730 | 4.12 | 1.03 | 0.03 | 3.06 | 3.09 |
| twin | image | 1e-04 | 8 | 3 | 4 | 10 | 0.688 | 3.24 | 0.96 | 0.02 | 2.26 | 2.28 |
| twin | image | 1e-04 | 8 | 5 | 2 | 6 | 0.553 | 2.87 | 0.46 | 0.03 | 2.38 | 2.41 |
| twin | image | 1e-04 | 8 | 5 | 2 | 10 | 0.541 | 2.29 | 0.45 | 0.02 | 1.83 | 1.85 |
| twin | image | 1e-04 | 8 | 5 | 4 | 6 | 0.621 | 2.87 | 0.46 | 0.03 | 2.38 | 2.41 |
| twin | image | 1e-04 | 8 | 5 | 4 | 10 | 0.579 | 2.29 | 0.45 | 0.02 | 1.83 | 1.85 |
| twin | image | 1e-05 | 8 | 3 | 2 | 6 | 0.631 | 3.82 | 0.92 | 0.03 | 2.87 | 2.90 |
| twin | image | 1e-05 | 8 | 3 | 2 | 10 | 0.611 | 3.05 | 0.86 | 0.02 | 2.16 | 2.18 |
| twin | image | 1e-05 | 8 | 3 | 4 | 6 | 0.692 | 3.82 | 0.92 | 0.03 | 2.87 | 2.90 |
| twin | image | 1e-05 | 8 | 3 | 4 | 10 | 0.660 | 3.05 | 0.86 | 0.02 | 2.16 | 2.18 |
| twin | image | 1e-05 | 8 | 5 | 2 | 6 | 0.529 | 2.72 | 0.43 | 0.03 | 2.26 | 2.29 |
| twin | image | 1e-05 | 8 | 5 | 2 | 10 | 0.515 | 2.21 | 0.42 | 0.02 | 1.77 | 1.79 |
| twin | image | 1e-05 | 8 | 5 | 4 | 6 | 0.597 | 2.72 | 0.43 | 0.03 | 2.26 | 2.29 |
| twin | image | 1e-05 | 8 | 5 | 4 | 10 | 0.557 | 2.21 | 0.42 | 0.02 | 1.77 | 1.79 |

## Labelled budgets

Each row is the tuning point with the highest bus/truck track Pd whose false alarms per CPI are at most that budget. A budget with no such point is infeasible and is not evaluated.

| Budget | Clutter | Method | Feasible | Pfa | Train | Eps | Coast | Assoc. | Tuning track Pd | Tuning FA / CPI |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | blind | none | no | — | — | — | — | — | — | — |
| 2 | blind | image | yes | 1e-03 | 4 | 5 | 4 | 10 | 0.678 | 1.97 |
| 2 | twin | none | no | — | — | — | — | — | — | — |
| 2 | twin | image | yes | 1e-03 | 4 | 5 | 4 | 10 | 0.675 | 1.97 |
| 4 | blind | none | yes | 1e-05 | 4 | 3 | 4 | 6 | 0.814 | 3.84 |
| 4 | blind | image | yes | 1e-03 | 4 | 3 | 4 | 6 | 0.870 | 3.79 |
| 4 | twin | none | yes | 1e-05 | 4 | 3 | 4 | 6 | 0.807 | 3.82 |
| 4 | twin | image | yes | 1e-03 | 4 | 3 | 4 | 6 | 0.871 | 3.78 |
| 6 | blind | none | yes | 1e-03 | 4 | 3 | 4 | 6 | 0.871 | 4.62 |
| 6 | blind | image | yes | 1e-03 | 4 | 3 | 4 | 6 | 0.870 | 3.79 |
| 6 | twin | none | yes | 1e-03 | 4 | 3 | 4 | 6 | 0.870 | 4.61 |
| 6 | twin | image | yes | 1e-03 | 4 | 3 | 4 | 6 | 0.871 | 3.78 |

## Evaluation at those budgets

Seeds 1001–1010. The budget is a label, not a selected operating point. Evaluation false alarms are measured and are not forced under the budget.

| Bandwidth | Budget | Clutter | Method | Mount | Class | GT | Track Pd | FA / CPI | Fragment | Ghost | Other | Vel. RMSE [m/s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 2 | blind | none | — | — | — | infeasible | — | — | — | — | — |
| 1024 | 2 | blind | image | lamppost | bus/truck | 30301 | 0.664 | 2.07 | 0.47 | 0.03 | 2.04 | 6.84 |
| 1024 | 2 | blind | image | lamppost | pedestrian | 119668 | 0.079 | 2.07 | 0.47 | 0.03 | 2.04 | 2.98 |
| 1024 | 2 | blind | image | lamppost | car | 80671 | 0.357 | 2.07 | 0.47 | 0.03 | 2.04 | 7.25 |
| 1024 | 2 | blind | image | facade | bus/truck | 34481 | 0.720 | 1.92 | 0.69 | 0.02 | 1.91 | 5.74 |
| 1024 | 2 | blind | image | facade | pedestrian | 134913 | 0.118 | 1.92 | 0.69 | 0.02 | 1.91 | 3.98 |
| 1024 | 2 | blind | image | facade | car | 91561 | 0.398 | 1.92 | 0.69 | 0.02 | 1.91 | 6.04 |
| 1024 | 2 | twin | none | — | — | — | infeasible | — | — | — | — | — |
| 1024 | 2 | twin | image | lamppost | bus/truck | 30301 | 0.666 | 2.07 | 0.47 | 0.03 | 2.04 | 6.89 |
| 1024 | 2 | twin | image | lamppost | pedestrian | 119668 | 0.075 | 2.07 | 0.47 | 0.03 | 2.04 | 3.10 |
| 1024 | 2 | twin | image | lamppost | car | 80671 | 0.355 | 2.07 | 0.47 | 0.03 | 2.04 | 7.22 |
| 1024 | 2 | twin | image | facade | bus/truck | 34481 | 0.719 | 1.92 | 0.70 | 0.02 | 1.90 | 5.75 |
| 1024 | 2 | twin | image | facade | pedestrian | 134913 | 0.119 | 1.92 | 0.70 | 0.02 | 1.90 | 3.98 |
| 1024 | 2 | twin | image | facade | car | 91561 | 0.395 | 1.92 | 0.70 | 0.02 | 1.90 | 6.07 |
| 1024 | 4 | blind | none | lamppost | bus/truck | 30301 | 0.804 | 3.67 | 1.07 | 0.06 | 3.60 | 5.82 |
| 1024 | 4 | blind | none | lamppost | pedestrian | 119668 | 0.212 | 3.67 | 1.07 | 0.06 | 3.60 | 1.91 |
| 1024 | 4 | blind | none | lamppost | car | 80671 | 0.559 | 3.67 | 1.07 | 0.06 | 3.60 | 5.23 |
| 1024 | 4 | blind | none | facade | bus/truck | 34481 | 0.841 | 4.35 | 1.48 | 0.18 | 4.16 | 4.47 |
| 1024 | 4 | blind | none | facade | pedestrian | 134913 | 0.335 | 4.35 | 1.48 | 0.18 | 4.16 | 2.40 |
| 1024 | 4 | blind | none | facade | car | 91561 | 0.635 | 4.35 | 1.48 | 0.18 | 4.16 | 3.90 |
| 1024 | 4 | blind | image | lamppost | bus/truck | 30301 | 0.859 | 4.04 | 1.27 | 0.05 | 3.99 | 5.60 |
| 1024 | 4 | blind | image | lamppost | pedestrian | 119668 | 0.216 | 4.04 | 1.27 | 0.05 | 3.99 | 1.92 |
| 1024 | 4 | blind | image | lamppost | car | 80671 | 0.583 | 4.04 | 1.27 | 0.05 | 3.99 | 5.44 |
| 1024 | 4 | blind | image | facade | bus/truck | 34481 | 0.897 | 3.70 | 1.78 | 0.06 | 3.64 | 4.23 |
| 1024 | 4 | blind | image | facade | pedestrian | 134913 | 0.343 | 3.70 | 1.78 | 0.06 | 3.64 | 2.30 |
| 1024 | 4 | blind | image | facade | car | 91561 | 0.677 | 3.70 | 1.78 | 0.06 | 3.64 | 3.78 |
| 1024 | 4 | twin | none | lamppost | bus/truck | 30301 | 0.806 | 3.64 | 1.07 | 0.06 | 3.58 | 5.77 |
| 1024 | 4 | twin | none | lamppost | pedestrian | 119668 | 0.206 | 3.64 | 1.07 | 0.06 | 3.58 | 1.90 |
| 1024 | 4 | twin | none | lamppost | car | 80671 | 0.558 | 3.64 | 1.07 | 0.06 | 3.58 | 5.29 |
| 1024 | 4 | twin | none | facade | bus/truck | 34481 | 0.843 | 4.34 | 1.48 | 0.18 | 4.15 | 4.42 |
| 1024 | 4 | twin | none | facade | pedestrian | 134913 | 0.330 | 4.34 | 1.48 | 0.18 | 4.15 | 2.43 |
| 1024 | 4 | twin | none | facade | car | 91561 | 0.639 | 4.34 | 1.48 | 0.18 | 4.15 | 3.90 |
| 1024 | 4 | twin | image | lamppost | bus/truck | 30301 | 0.858 | 4.02 | 1.27 | 0.06 | 3.97 | 5.53 |
| 1024 | 4 | twin | image | lamppost | pedestrian | 119668 | 0.215 | 4.02 | 1.27 | 0.06 | 3.97 | 2.06 |
| 1024 | 4 | twin | image | lamppost | car | 80671 | 0.582 | 4.02 | 1.27 | 0.06 | 3.97 | 5.38 |
| 1024 | 4 | twin | image | facade | bus/truck | 34481 | 0.896 | 3.70 | 1.79 | 0.06 | 3.64 | 4.12 |
| 1024 | 4 | twin | image | facade | pedestrian | 134913 | 0.339 | 3.70 | 1.79 | 0.06 | 3.64 | 2.37 |
| 1024 | 4 | twin | image | facade | car | 91561 | 0.675 | 3.70 | 1.79 | 0.06 | 3.64 | 3.77 |
| 1024 | 6 | blind | none | lamppost | bus/truck | 30301 | 0.861 | 4.35 | 1.29 | 0.08 | 4.27 | 5.51 |
| 1024 | 6 | blind | none | lamppost | pedestrian | 119668 | 0.239 | 4.35 | 1.29 | 0.08 | 4.27 | 1.98 |
| 1024 | 6 | blind | none | lamppost | car | 80671 | 0.585 | 4.35 | 1.29 | 0.08 | 4.27 | 5.41 |
| 1024 | 6 | blind | none | facade | bus/truck | 34481 | 0.897 | 5.19 | 1.81 | 0.24 | 4.95 | 4.10 |
| 1024 | 6 | blind | none | facade | pedestrian | 134913 | 0.374 | 5.19 | 1.81 | 0.24 | 4.95 | 2.37 |
| 1024 | 6 | blind | none | facade | car | 91561 | 0.676 | 5.19 | 1.81 | 0.24 | 4.95 | 3.77 |
| 1024 | 6 | blind | image | lamppost | bus/truck | 30301 | 0.859 | 4.04 | 1.27 | 0.05 | 3.99 | 5.60 |
| 1024 | 6 | blind | image | lamppost | pedestrian | 119668 | 0.216 | 4.04 | 1.27 | 0.05 | 3.99 | 1.92 |
| 1024 | 6 | blind | image | lamppost | car | 80671 | 0.583 | 4.04 | 1.27 | 0.05 | 3.99 | 5.44 |
| 1024 | 6 | blind | image | facade | bus/truck | 34481 | 0.897 | 3.70 | 1.78 | 0.06 | 3.64 | 4.23 |
| 1024 | 6 | blind | image | facade | pedestrian | 134913 | 0.343 | 3.70 | 1.78 | 0.06 | 3.64 | 2.30 |
| 1024 | 6 | blind | image | facade | car | 91561 | 0.677 | 3.70 | 1.78 | 0.06 | 3.64 | 3.78 |
| 1024 | 6 | twin | none | lamppost | bus/truck | 30301 | 0.864 | 4.33 | 1.28 | 0.09 | 4.25 | 5.48 |
| 1024 | 6 | twin | none | lamppost | pedestrian | 119668 | 0.237 | 4.33 | 1.28 | 0.09 | 4.25 | 2.02 |
| 1024 | 6 | twin | none | lamppost | car | 80671 | 0.584 | 4.33 | 1.28 | 0.09 | 4.25 | 5.39 |
| 1024 | 6 | twin | none | facade | bus/truck | 34481 | 0.898 | 5.17 | 1.81 | 0.24 | 4.93 | 4.13 |
| 1024 | 6 | twin | none | facade | pedestrian | 134913 | 0.369 | 5.17 | 1.81 | 0.24 | 4.93 | 2.36 |
| 1024 | 6 | twin | none | facade | car | 91561 | 0.672 | 5.17 | 1.81 | 0.24 | 4.93 | 3.80 |
| 1024 | 6 | twin | image | lamppost | bus/truck | 30301 | 0.858 | 4.02 | 1.27 | 0.06 | 3.97 | 5.53 |
| 1024 | 6 | twin | image | lamppost | pedestrian | 119668 | 0.215 | 4.02 | 1.27 | 0.06 | 3.97 | 2.06 |
| 1024 | 6 | twin | image | lamppost | car | 80671 | 0.582 | 4.02 | 1.27 | 0.06 | 3.97 | 5.38 |
| 1024 | 6 | twin | image | facade | bus/truck | 34481 | 0.896 | 3.70 | 1.79 | 0.06 | 3.64 | 4.12 |
| 1024 | 6 | twin | image | facade | pedestrian | 134913 | 0.339 | 3.70 | 1.79 | 0.06 | 3.64 | 2.37 |
| 1024 | 6 | twin | image | facade | car | 91561 | 0.675 | 3.70 | 1.79 | 0.06 | 3.64 | 3.77 |
| 2048 | 2 | blind | none | — | — | — | infeasible | — | — | — | — | — |
| 2048 | 2 | blind | image | lamppost | bus/truck | 30301 | 0.638 | 2.43 | 0.46 | 0.02 | 2.41 | 7.16 |
| 2048 | 2 | blind | image | lamppost | pedestrian | 119668 | 0.087 | 2.43 | 0.46 | 0.02 | 2.41 | 2.55 |
| 2048 | 2 | blind | image | lamppost | car | 80671 | 0.320 | 2.43 | 0.46 | 0.02 | 2.41 | 7.56 |
| 2048 | 2 | blind | image | facade | bus/truck | 34481 | 0.771 | 2.12 | 0.91 | 0.01 | 2.10 | 5.46 |
| 2048 | 2 | blind | image | facade | pedestrian | 134913 | 0.124 | 2.12 | 0.91 | 0.01 | 2.10 | 3.94 |
| 2048 | 2 | blind | image | facade | car | 91561 | 0.415 | 2.12 | 0.91 | 0.01 | 2.10 | 6.12 |
| 2048 | 2 | twin | none | — | — | — | infeasible | — | — | — | — | — |
| 2048 | 2 | twin | image | lamppost | bus/truck | 30301 | 0.636 | 2.44 | 0.46 | 0.02 | 2.43 | 7.04 |
| 2048 | 2 | twin | image | lamppost | pedestrian | 119668 | 0.086 | 2.44 | 0.46 | 0.02 | 2.43 | 2.54 |
| 2048 | 2 | twin | image | lamppost | car | 80671 | 0.318 | 2.44 | 0.46 | 0.02 | 2.43 | 7.64 |
| 2048 | 2 | twin | image | facade | bus/truck | 34481 | 0.771 | 2.11 | 0.91 | 0.01 | 2.10 | 5.54 |
| 2048 | 2 | twin | image | facade | pedestrian | 134913 | 0.121 | 2.11 | 0.91 | 0.01 | 2.10 | 3.91 |
| 2048 | 2 | twin | image | facade | car | 91561 | 0.418 | 2.11 | 0.91 | 0.01 | 2.10 | 6.12 |
| 2048 | 4 | blind | none | lamppost | bus/truck | 30301 | 0.928 | 5.91 | 1.83 | 0.12 | 5.78 | 5.02 |
| 2048 | 4 | blind | none | lamppost | pedestrian | 119668 | 0.304 | 5.91 | 1.83 | 0.12 | 5.78 | 1.85 |
| 2048 | 4 | blind | none | lamppost | car | 80671 | 0.608 | 5.91 | 1.83 | 0.12 | 5.78 | 5.76 |
| 2048 | 4 | blind | none | facade | bus/truck | 34481 | 0.948 | 8.04 | 2.31 | 0.44 | 7.60 | 3.78 |
| 2048 | 4 | blind | none | facade | pedestrian | 134913 | 0.410 | 8.04 | 2.31 | 0.44 | 7.60 | 2.54 |
| 2048 | 4 | blind | none | facade | car | 91561 | 0.703 | 8.04 | 2.31 | 0.44 | 7.60 | 3.95 |
| 2048 | 4 | blind | image | lamppost | bus/truck | 30301 | 0.937 | 5.77 | 1.93 | 0.05 | 5.72 | 5.00 |
| 2048 | 4 | blind | image | lamppost | pedestrian | 119668 | 0.278 | 5.77 | 1.93 | 0.05 | 5.72 | 1.81 |
| 2048 | 4 | blind | image | lamppost | car | 80671 | 0.614 | 5.77 | 1.93 | 0.05 | 5.72 | 5.87 |
| 2048 | 4 | blind | image | facade | bus/truck | 34481 | 0.962 | 5.47 | 2.55 | 0.05 | 5.42 | 3.56 |
| 2048 | 4 | blind | image | facade | pedestrian | 134913 | 0.386 | 5.47 | 2.55 | 0.05 | 5.42 | 2.45 |
| 2048 | 4 | blind | image | facade | car | 91561 | 0.724 | 5.47 | 2.55 | 0.05 | 5.42 | 3.66 |
| 2048 | 4 | twin | none | lamppost | bus/truck | 30301 | 0.927 | 5.89 | 1.83 | 0.11 | 5.77 | 4.98 |
| 2048 | 4 | twin | none | lamppost | pedestrian | 119668 | 0.306 | 5.89 | 1.83 | 0.11 | 5.77 | 1.83 |
| 2048 | 4 | twin | none | lamppost | car | 80671 | 0.611 | 5.89 | 1.83 | 0.11 | 5.77 | 5.79 |
| 2048 | 4 | twin | none | facade | bus/truck | 34481 | 0.948 | 8.02 | 2.31 | 0.43 | 7.58 | 3.74 |
| 2048 | 4 | twin | none | facade | pedestrian | 134913 | 0.408 | 8.02 | 2.31 | 0.43 | 7.58 | 2.54 |
| 2048 | 4 | twin | none | facade | car | 91561 | 0.699 | 8.02 | 2.31 | 0.43 | 7.58 | 3.97 |
| 2048 | 4 | twin | image | lamppost | bus/truck | 30301 | 0.936 | 5.77 | 1.93 | 0.04 | 5.73 | 5.10 |
| 2048 | 4 | twin | image | lamppost | pedestrian | 119668 | 0.279 | 5.77 | 1.93 | 0.04 | 5.73 | 1.84 |
| 2048 | 4 | twin | image | lamppost | car | 80671 | 0.616 | 5.77 | 1.93 | 0.04 | 5.73 | 5.95 |
| 2048 | 4 | twin | image | facade | bus/truck | 34481 | 0.959 | 5.44 | 2.55 | 0.05 | 5.39 | 3.45 |
| 2048 | 4 | twin | image | facade | pedestrian | 134913 | 0.390 | 5.44 | 2.55 | 0.05 | 5.39 | 2.41 |
| 2048 | 4 | twin | image | facade | car | 91561 | 0.725 | 5.44 | 2.55 | 0.05 | 5.39 | 3.69 |
| 2048 | 6 | blind | none | lamppost | bus/truck | 30301 | 0.935 | 6.64 | 1.96 | 0.14 | 6.49 | 5.02 |
| 2048 | 6 | blind | none | lamppost | pedestrian | 119668 | 0.304 | 6.64 | 1.96 | 0.14 | 6.49 | 2.03 |
| 2048 | 6 | blind | none | lamppost | car | 80671 | 0.617 | 6.64 | 1.96 | 0.14 | 6.49 | 5.91 |
| 2048 | 6 | blind | none | facade | bus/truck | 34481 | 0.960 | 9.13 | 2.59 | 0.49 | 8.64 | 3.66 |
| 2048 | 6 | blind | none | facade | pedestrian | 134913 | 0.419 | 9.13 | 2.59 | 0.49 | 8.64 | 2.59 |
| 2048 | 6 | blind | none | facade | car | 91561 | 0.722 | 9.13 | 2.59 | 0.49 | 8.64 | 3.92 |
| 2048 | 6 | blind | image | lamppost | bus/truck | 30301 | 0.937 | 5.77 | 1.93 | 0.05 | 5.72 | 5.00 |
| 2048 | 6 | blind | image | lamppost | pedestrian | 119668 | 0.278 | 5.77 | 1.93 | 0.05 | 5.72 | 1.81 |
| 2048 | 6 | blind | image | lamppost | car | 80671 | 0.614 | 5.77 | 1.93 | 0.05 | 5.72 | 5.87 |
| 2048 | 6 | blind | image | facade | bus/truck | 34481 | 0.962 | 5.47 | 2.55 | 0.05 | 5.42 | 3.56 |
| 2048 | 6 | blind | image | facade | pedestrian | 134913 | 0.386 | 5.47 | 2.55 | 0.05 | 5.42 | 2.45 |
| 2048 | 6 | blind | image | facade | car | 91561 | 0.724 | 5.47 | 2.55 | 0.05 | 5.42 | 3.66 |
| 2048 | 6 | twin | none | lamppost | bus/truck | 30301 | 0.938 | 6.62 | 1.97 | 0.13 | 6.49 | 5.14 |
| 2048 | 6 | twin | none | lamppost | pedestrian | 119668 | 0.304 | 6.62 | 1.97 | 0.13 | 6.49 | 2.08 |
| 2048 | 6 | twin | none | lamppost | car | 80671 | 0.615 | 6.62 | 1.97 | 0.13 | 6.49 | 6.01 |
| 2048 | 6 | twin | none | facade | bus/truck | 34481 | 0.960 | 9.11 | 2.60 | 0.49 | 8.62 | 3.55 |
| 2048 | 6 | twin | none | facade | pedestrian | 134913 | 0.421 | 9.11 | 2.60 | 0.49 | 8.62 | 2.61 |
| 2048 | 6 | twin | none | facade | car | 91561 | 0.723 | 9.11 | 2.60 | 0.49 | 8.62 | 3.88 |
| 2048 | 6 | twin | image | lamppost | bus/truck | 30301 | 0.936 | 5.77 | 1.93 | 0.04 | 5.73 | 5.10 |
| 2048 | 6 | twin | image | lamppost | pedestrian | 119668 | 0.279 | 5.77 | 1.93 | 0.04 | 5.73 | 1.84 |
| 2048 | 6 | twin | image | lamppost | car | 80671 | 0.616 | 5.77 | 1.93 | 0.04 | 5.73 | 5.95 |
| 2048 | 6 | twin | image | facade | bus/truck | 34481 | 0.959 | 5.44 | 2.55 | 0.05 | 5.39 | 3.45 |
| 2048 | 6 | twin | image | facade | pedestrian | 134913 | 0.390 | 5.44 | 2.55 | 0.05 | 5.39 | 2.41 |
| 2048 | 6 | twin | image | facade | car | 91561 | 0.725 | 5.44 | 2.55 | 0.05 | 5.39 | 3.69 |

## Gate sensitivity, blind clutter, image method, labelled budget

| Bandwidth | Budget | Scale | Class | Track Pd |
|---|---|---|---|---|
| 1024 | 2 | 0.5 | bus/truck | 0.407 |
| 1024 | 2 | 0.5 | pedestrian | 0.046 |
| 1024 | 2 | 0.5 | car | 0.152 |
| 1024 | 2 | 1.0 | bus/truck | 0.694 |
| 1024 | 2 | 1.0 | pedestrian | 0.099 |
| 1024 | 2 | 1.0 | car | 0.379 |
| 1024 | 2 | 2.0 | bus/truck | 0.745 |
| 1024 | 2 | 2.0 | pedestrian | 0.205 |
| 1024 | 2 | 2.0 | car | 0.551 |
| 1024 | 4 | 0.5 | bus/truck | 0.638 |
| 1024 | 4 | 0.5 | pedestrian | 0.153 |
| 1024 | 4 | 0.5 | car | 0.340 |
| 1024 | 4 | 1.0 | bus/truck | 0.879 |
| 1024 | 4 | 1.0 | pedestrian | 0.283 |
| 1024 | 4 | 1.0 | car | 0.633 |
| 1024 | 4 | 2.0 | bus/truck | 0.937 |
| 1024 | 4 | 2.0 | pedestrian | 0.437 |
| 1024 | 4 | 2.0 | car | 0.792 |
| 1024 | 6 | 0.5 | bus/truck | 0.638 |
| 1024 | 6 | 0.5 | pedestrian | 0.153 |
| 1024 | 6 | 0.5 | car | 0.340 |
| 1024 | 6 | 1.0 | bus/truck | 0.879 |
| 1024 | 6 | 1.0 | pedestrian | 0.283 |
| 1024 | 6 | 1.0 | car | 0.633 |
| 1024 | 6 | 2.0 | bus/truck | 0.937 |
| 1024 | 6 | 2.0 | pedestrian | 0.437 |
| 1024 | 6 | 2.0 | car | 0.792 |
| 2048 | 2 | 0.5 | bus/truck | 0.433 |
| 2048 | 2 | 0.5 | pedestrian | 0.055 |
| 2048 | 2 | 0.5 | car | 0.150 |
| 2048 | 2 | 1.0 | bus/truck | 0.709 |
| 2048 | 2 | 1.0 | pedestrian | 0.107 |
| 2048 | 2 | 1.0 | car | 0.371 |
| 2048 | 2 | 2.0 | bus/truck | 0.761 |
| 2048 | 2 | 2.0 | pedestrian | 0.208 |
| 2048 | 2 | 2.0 | car | 0.544 |
| 2048 | 4 | 0.5 | bus/truck | 0.776 |
| 2048 | 4 | 0.5 | pedestrian | 0.188 |
| 2048 | 4 | 0.5 | car | 0.356 |
| 2048 | 4 | 1.0 | bus/truck | 0.950 |
| 2048 | 4 | 1.0 | pedestrian | 0.335 |
| 2048 | 4 | 1.0 | car | 0.673 |
| 2048 | 4 | 2.0 | bus/truck | 0.979 |
| 2048 | 4 | 2.0 | pedestrian | 0.515 |
| 2048 | 4 | 2.0 | car | 0.848 |
| 2048 | 6 | 0.5 | bus/truck | 0.776 |
| 2048 | 6 | 0.5 | pedestrian | 0.188 |
| 2048 | 6 | 0.5 | car | 0.356 |
| 2048 | 6 | 1.0 | bus/truck | 0.950 |
| 2048 | 6 | 1.0 | pedestrian | 0.335 |
| 2048 | 6 | 1.0 | car | 0.673 |
| 2048 | 6 | 2.0 | bus/truck | 0.979 |
| 2048 | 6 | 2.0 | pedestrian | 0.515 |
| 2048 | 6 | 2.0 | car | 0.848 |

## Confirmed track before a 10 dB LoS event

1024 subcarriers, at the labelled budget. Age is at event start. A blocker that was never confirmed inside its class gate is a miss and has no age.

| Budget | Class | Mount | Method | Clutter | Events | Lead 0.1 s | 0.3 s | 0.5 s | 1.0 s | Age p50 [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | bus/truck | lamppost | image | blind | 227 | 176/227 | 166/227 | 169/227 | 157/227 | 2.60 |
| 2 | bus/truck | lamppost | image | twin | 227 | 180/227 | 175/227 | 176/227 | 158/227 | 2.50 |
| 2 | pedestrian | lamppost | image | blind | 167 | 22/167 | 26/167 | 29/167 | 26/167 | 1.80 |
| 2 | pedestrian | lamppost | image | twin | 167 | 21/167 | 24/167 | 26/167 | 26/167 | 2.20 |
| 2 | bus/truck | facade | image | blind | 212 | 180/212 | 183/212 | 176/212 | 161/212 | 3.20 |
| 2 | bus/truck | facade | image | twin | 212 | 178/212 | 184/212 | 180/212 | 167/212 | 3.20 |
| 2 | pedestrian | facade | image | blind | 188 | 20/188 | 19/188 | 23/188 | 23/188 | 2.40 |
| 2 | pedestrian | facade | image | twin | 188 | 19/188 | 19/188 | 25/188 | 25/188 | 2.90 |
| 4 | bus/truck | lamppost | none | blind | 227 | 189/227 | 188/227 | 186/227 | 162/227 | 1.00 |
| 4 | bus/truck | lamppost | none | twin | 227 | 193/227 | 190/227 | 186/227 | 163/227 | 1.20 |
| 4 | bus/truck | lamppost | image | blind | 227 | 196/227 | 190/227 | 188/227 | 172/227 | 1.20 |
| 4 | bus/truck | lamppost | image | twin | 227 | 197/227 | 189/227 | 189/227 | 172/227 | 1.10 |
| 4 | pedestrian | lamppost | none | blind | 167 | 67/167 | 64/167 | 61/167 | 53/167 | 1.70 |
| 4 | pedestrian | lamppost | none | twin | 167 | 71/167 | 67/167 | 64/167 | 50/167 | 1.60 |
| 4 | pedestrian | lamppost | image | blind | 167 | 63/167 | 64/167 | 68/167 | 56/167 | 2.30 |
| 4 | pedestrian | lamppost | image | twin | 167 | 65/167 | 66/167 | 66/167 | 57/167 | 2.20 |
| 4 | bus/truck | facade | none | blind | 212 | 188/212 | 182/212 | 173/212 | 162/212 | 1.90 |
| 4 | bus/truck | facade | none | twin | 212 | 190/212 | 181/212 | 174/212 | 164/212 | 1.80 |
| 4 | bus/truck | facade | image | blind | 212 | 193/212 | 192/212 | 188/212 | 173/212 | 1.70 |
| 4 | bus/truck | facade | image | twin | 212 | 193/212 | 194/212 | 189/212 | 170/212 | 1.80 |
| 4 | pedestrian | facade | none | blind | 188 | 86/188 | 85/188 | 96/188 | 87/188 | 2.50 |
| 4 | pedestrian | facade | none | twin | 188 | 92/188 | 88/188 | 95/188 | 87/188 | 2.20 |
| 4 | pedestrian | facade | image | blind | 188 | 79/188 | 78/188 | 86/188 | 78/188 | 3.70 |
| 4 | pedestrian | facade | image | twin | 188 | 83/188 | 74/188 | 79/188 | 76/188 | 3.20 |
| 6 | bus/truck | lamppost | none | blind | 227 | 197/227 | 192/227 | 189/227 | 173/227 | 1.10 |
| 6 | bus/truck | lamppost | none | twin | 227 | 195/227 | 191/227 | 190/227 | 171/227 | 1.10 |
| 6 | bus/truck | lamppost | image | blind | 227 | 196/227 | 190/227 | 188/227 | 172/227 | 1.20 |
| 6 | bus/truck | lamppost | image | twin | 227 | 197/227 | 189/227 | 189/227 | 172/227 | 1.10 |
| 6 | pedestrian | lamppost | none | blind | 167 | 72/167 | 75/167 | 78/167 | 62/167 | 1.90 |
| 6 | pedestrian | lamppost | none | twin | 167 | 72/167 | 72/167 | 75/167 | 63/167 | 1.90 |
| 6 | pedestrian | lamppost | image | blind | 167 | 63/167 | 64/167 | 68/167 | 56/167 | 2.30 |
| 6 | pedestrian | lamppost | image | twin | 167 | 65/167 | 66/167 | 66/167 | 57/167 | 2.20 |
| 6 | bus/truck | facade | none | blind | 212 | 192/212 | 189/212 | 188/212 | 172/212 | 1.50 |
| 6 | bus/truck | facade | none | twin | 212 | 193/212 | 192/212 | 185/212 | 169/212 | 1.60 |
| 6 | bus/truck | facade | image | blind | 212 | 193/212 | 192/212 | 188/212 | 173/212 | 1.70 |
| 6 | bus/truck | facade | image | twin | 212 | 193/212 | 194/212 | 189/212 | 170/212 | 1.80 |
| 6 | pedestrian | facade | none | blind | 188 | 86/188 | 84/188 | 96/188 | 90/188 | 3.50 |
| 6 | pedestrian | facade | none | twin | 188 | 91/188 | 91/188 | 100/188 | 91/188 | 3.20 |
| 6 | pedestrian | facade | image | blind | 188 | 79/188 | 78/188 | 86/188 | 78/188 | 3.70 |
| 6 | pedestrian | facade | image | twin | 188 | 83/188 | 74/188 | 79/188 | 76/188 | 3.20 |

# M2 follow-up

No operating point is selected. This does not mark M2 done.

Analysis git 216eb537d3b157f9591e062c29ec6bd24067d81a dirty a73ee25ae930f376d6c0c5a8eedf39b023977d01e1e8fd92ed88dc60c64daf9d config b75bb5e6c91c85c16492d6ca67ac736a9384188a6e1593d4d5fec2c3fa449c7c.
Detection caches kept the rebuild's dirty hash 6ac8fbeb374bb5c9031b85c72a835427ad75b4845f58a9f3dbbda1afe572b2a1. The commit and the config hash match, so the channels were not re-traced.

## Plain results

Twin-based clutter subtraction is a null result. Static canyon clutter is exactly zero-Doppler, and no impairment was added. On the evaluation table the blind and twin columns match.

Image-method ghost handling has no bus/truck track-Pd gain at equal parameters (mean change -0.007 over 96 tuning grid pairs). It removes 20.6% of the unmatched clusters on those pairs. The reviewed operating point was 6.14 versus 5.30 unmatched clusters per CPI, which is 14%.

## Bandwidth

2048 subcarriers are not the main sensing setting. The communication carrier in this scenario, which M3 inherits unless it is changed, is numerology 3 with 512 subcarriers (61.44 MHz). 2048 sensing subcarriers are 245.8 MHz. Use 2048 as the main setting only if that communication carrier is widened to the same bandwidth.

## Detector held fixed

Blind clutter, no ghost handling, the labelled budget of 4 false alarms per CPI on the tuning seeds. Pfa 1e-05, train 4, DBSCAN eps 3 m. The unconstrained tracker keeps association 6 m, coast 4, process_q 1.0. That process noise is the white-acceleration density in each Cartesian axis: position variance q dt^3/3, cross term q dt^2/2, velocity variance q dt, with dt = 0.1 s. Measurement sigmas are 1.5 m, 2 degrees, and 0.5 m/s radial.

The map tracker was tuned only on seeds 101–105, for bus/truck track Pd, with no false-alarm cap. Chosen association 4 m, coast 8, process_q 0.25, lane/sidewalk gate 2.5 m, sidewalk leave distance 4.0 m. Tuning bus/truck track Pd 0.895.

## Velocity, unconstrained, evaluation seeds, after 1 s

Radial and cross-range are horizontal components relative to oru-0. Along-lane is the street axis x. Cross-lane is y.

| Class | Samples | Radial RMSE | Cross-range RMSE | Along-lane RMSE | Cross-lane RMSE | ID switches | Per target |
|---|---|---|---|---|---|---|---|
| bus/truck | 41728 | 4.12 | 3.04 | 4.72 | 1.97 | 6847 | 8.432 |
| pedestrian | 51914 | 1.58 | 1.58 | 1.79 | 1.35 | 5655 | 7.195 |
| car | 79414 | 3.39 | 2.97 | 4.12 | 1.82 | 10357 | 4.007 |

## Why a pedestrian track at t−0.5 s is missing at t−0.1 s

Pedestrian 10 dB events on the evaluation seeds where a confirmed track is inside the 1.5 m gate at t−0.5 s and not at t−0.1 s: 37. The same track id has been dropped: 10. The same track id is alive but outside the gate: 25. Anything else: 2.

Coast is 4 snapshots (0.4 s) and a track is kept while its miss count is at most that. A track that was updated at t−0.5 s is therefore still alive at t−0.1 s unless it was already coasting or the coast is shorter than four frames. The usual case is the second count: the track leaves the 1.5 m pedestrian gate. Pedestrian velocity error is a few metres per second, so four frames of prediction, a turn onto a crossing, or an association onto a nearby cluster moves it by more than the gate. A new track is not confirmed until the second hit, so the lead at 0.1 s is empty even though the lead at 0.5 s was not.

## Map-constrained tracker, evaluation seeds, 1024 subcarriers

Along-lane and cross-lane are velocity RMSE [m/s] after 1 s of track age, on the street axes x and y. An ID switch is a change of the confirmed track id matched to one ground-truth identity, including after a gap. The per-target rate counts each identity once per job.

| Tracker | Class | Track Pd | Pos. RMSE [m] | Vel. RMSE [m/s] | Along-lane | Cross-lane | ID switches | Per target | FA / CPI |
|---|---|---|---|---|---|---|---|---|---|
| unconstrained | bus/truck | 0.824 | 2.91 | 5.12 | 4.72 | 1.97 | 6847 | 8.432 | 4.01 |
| unconstrained | pedestrian | 0.277 | 0.83 | 2.24 | 1.79 | 1.35 | 5655 | 7.195 | 4.01 |
| unconstrained | car | 0.599 | 1.67 | 4.50 | 4.12 | 1.82 | 10357 | 4.007 | 4.01 |
| map | bus/truck | 0.900 | 2.47 | 3.88 | 3.84 | 0.53 | 6174 | 7.575 | 4.01 |
| map | pedestrian | 0.449 | 0.55 | 1.47 | 1.41 | 0.41 | 5631 | 7.182 | 4.01 |
| map | car | 0.731 | 1.30 | 2.78 | 2.75 | 0.44 | 10346 | 4.012 | 4.01 |

## Lead time, same detector, 1024 subcarriers

| Tracker | Class | Mount | Events | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
|---|---|---|---|---|---|---|---|
| unconstrained | bus/truck | lamppost | 227 | 189/227 | 188/227 | 186/227 | 162/227 |
| unconstrained | pedestrian | lamppost | 167 | 67/167 | 64/167 | 61/167 | 53/167 |
| unconstrained | bus/truck | facade | 212 | 188/212 | 182/212 | 173/212 | 162/212 |
| unconstrained | pedestrian | facade | 188 | 86/188 | 85/188 | 96/188 | 87/188 |
| map | bus/truck | lamppost | 227 | 196/227 | 192/227 | 187/227 | 170/227 |
| map | pedestrian | lamppost | 167 | 91/167 | 92/167 | 90/167 | 80/167 |
| map | bus/truck | facade | 212 | 197/212 | 191/212 | 187/212 | 171/212 |
| map | pedestrian | facade | 188 | 127/188 | 121/188 | 117/188 | 110/188 |

# M2 final configuration

No operating point is selected. This does not mark M2 done.

Analysis git 83051cdfb1c37322c15c8ecff1021dd86c78b7ad dirty ee5e8ca3df976c84e6f6b41a6dc8e068cd8accbb8c7dc84208182a411fba1155 config f16a808bf8c487ba6e8095b3ffaf1e543200fb268c4260c58c013d3e3a9887ec.
Detection caches git 216eb537d3b157f9591e062c29ec6bd24067d81a dirty 6ac8fbeb374bb5c9031b85c72a835427ad75b4845f58a9f3dbbda1afe572b2a1 config b75bb5e6c91c85c16492d6ca67ac736a9384188a6e1593d4d5fec2c3fa449c7c. Channels were not re-traced. The config hash differs because the comm carrier was set to 1024 subcarriers; that field is not used by the sensing detections.

Comm carrier is now numerology 3, 1024 subcarriers (122.88 MHz), the same as the main sensing setting (122.88 MHz). 2048 sensing subcarriers stay a sensitivity case and are wider than this carrier.

Final configuration: blind clutter, image-method ghost handling, map-constrained tracker. Map parameters (tuning seeds 101–105, no FA cap): association 4 m, coast 8, process_q 0.25, lane/sidewalk gate 2.5 m, leave 4.0 m.

ID switches: one count per identity per job. Median lifetime is the median confirmed-track age [s] among tracks matched to that class in a job.

## Velocity sanity

No conversion bug. On an empty scene, `paths.doppler` matches `f_D = -2 v_receding / lambda` to relative error 1e-3. The range-Doppler peak converts with `v_app = f_D * lambda / 2` to within one Doppler bin (0.17 m/s) for approaching, receding, and crossing, with and without noise, for a point target and a 5-point vehicle. The EKF measurement is approaching radial `-v · r_hat`; the Jacobian matches a finite difference to 8e-11; birth and a 1 s constant-velocity track recover the true radial velocity (Cartesian RMSE 0.00 m/s approaching/receding, 0.08 m/s crossing).

The ~4 m/s bus/truck radial RMSE on the street is therefore not a Doppler sign or scale error. It is the Cartesian velocity of an associated track versus the mesh-origin ground truth after 1 s, under clutter, association, and coasting.

Empty scene. One constant-velocity target. Radar at the origin.

## EKF measurement model

Predicted approaching radial -9.5694 m/s vs -v·r_hat -9.5694 m/s: PASS
Jacobian max abs error vs finite difference 7.75e-11: PASS
H[radial, velocity] = [-0.63796178  0.717707    0.27910828]  (must be -r_hat for approaching measurement)
Birth velocity [ 6.10492846 -6.86804452 -2.6709062 ] vs -v_app r_hat [ 6.10492846 -6.86804452 -2.6709062 ]: err 0.000e+00 PASS

## Doppler sign and scale (empty scene, ConstantRCS)

Sionna: f_D = -2 v_receding / lambda, v_receding > 0 leaving the radar. Positive Doppler is approaching. Conversion: v_app = f_D * lambda / 2.

| Geometry | Noise | True v_app | Path v_app | Map v_app | Path f_D | Formula f_D | Range |
|---|---|---|---|---|---|---|---|
| approaching | False | 12.000 | 12.000 | 12.045 | 2241.6 | 2241.6 | 30.5 |
| approaching | True | 12.000 | 12.000 | 12.045 | 2241.6 | 2241.6 | 30.5 |
| receding | False | -12.000 | -12.000 | -12.045 | -2241.6 | -2241.6 | 30.5 |
| receding | True | -12.000 | -12.000 | -12.045 | -2241.6 | -2241.6 | 30.5 |
| crossing | False | -0.000 | 0.000 | 0.167 | 0.0 | -0.0 | 24.4 |
| crossing | True | -0.000 | 0.000 | 0.167 | 0.0 | -0.0 | 24.4 |

## Extended target (5 scattering points, vehicle-multi-sp)

| Geometry | Noise | Center v_app | Map v_app | Path v_app (max |a|) |
|---|---|---|---|---|
| approaching | False | 12.000 | 12.045 | 12.000 |
| approaching | True | 12.000 | 12.045 | 12.000 |
| receding | False | -12.000 | -12.045 | -12.000 |
| receding | True | -12.000 | -12.045 | -12.000 |
| crossing | False | -0.000 | 0.167 | 0.000 |
| crossing | True | -0.000 | 0.167 | 0.000 |

## EKF on noiseless point detections

| Geometry | True v_app | Track v_app after 1 s | Cartesian vel RMSE |
|---|---|---|---|
| approaching | 12.000 | 12.000 | 0.000 |
| receding | -12.000 | -12.000 | 0.000 |
| crossing | -8.263 | -8.284 | 0.075 |

## Budget 2 FA/CPI, evaluation seeds

Detector (tuning pick): Pfa 1e-03, train 4, eps 5 m, ghost association 10 m. Tuning bus/truck track Pd 0.678, FA/CPI 1.97.

| Class | Track Pd | FA/CPI | Radial RMSE | Along-lane | Cross-lane | ID switches | Per target | Median life [s] |
|---|---|---|---|---|---|---|---|---|
| bus/truck | 0.824 | 2.00 | 3.41 | 3.82 | 0.30 | 4812 | 5.926 | 3.00 |
| pedestrian | 0.279 | 2.00 | 1.52 | 1.58 | 0.33 | 4226 | 5.561 | 2.10 |
| car | 0.607 | 2.00 | 2.37 | 2.77 | 0.25 | 6296 | 2.447 | 3.00 |

| Class | Mount | Events | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
|---|---|---|---|---|---|---|
| bus/truck | lamppost | 227 | 191/227 | 185/227 | 181/227 | 164/227 |
| pedestrian | lamppost | 167 | 60/167 | 59/167 | 57/167 | 54/167 |
| bus/truck | facade | 212 | 196/212 | 192/212 | 186/212 | 168/212 |
| pedestrian | facade | 188 | 73/188 | 73/188 | 78/188 | 71/188 |

## Budget 4 FA/CPI, evaluation seeds

Detector (tuning pick): Pfa 1e-03, train 4, eps 3 m, ghost association 6 m. Tuning bus/truck track Pd 0.870, FA/CPI 3.79.

| Class | Track Pd | FA/CPI | Radial RMSE | Along-lane | Cross-lane | ID switches | Per target | Median life [s] |
|---|---|---|---|---|---|---|---|---|
| bus/truck | 0.941 | 3.87 | 3.16 | 3.45 | 0.38 | 6075 | 7.454 | 3.90 |
| pedestrian | 0.465 | 3.87 | 1.29 | 1.39 | 0.37 | 5542 | 7.051 | 3.20 |
| car | 0.763 | 3.87 | 2.47 | 2.79 | 0.34 | 11181 | 4.324 | 3.70 |

| Class | Mount | Events | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
|---|---|---|---|---|---|---|
| bus/truck | lamppost | 227 | 200/227 | 191/227 | 191/227 | 168/227 |
| pedestrian | lamppost | 167 | 96/167 | 94/167 | 89/167 | 73/167 |
| bus/truck | facade | 212 | 200/212 | 196/212 | 190/212 | 175/212 |
| pedestrian | facade | 188 | 121/188 | 117/188 | 119/188 | 104/188 |

## Budget 6 FA/CPI, evaluation seeds

Detector (tuning pick): Pfa 1e-03, train 4, eps 3 m, ghost association 6 m. Tuning bus/truck track Pd 0.870, FA/CPI 3.79.

| Class | Track Pd | FA/CPI | Radial RMSE | Along-lane | Cross-lane | ID switches | Per target | Median life [s] |
|---|---|---|---|---|---|---|---|---|
| bus/truck | 0.941 | 3.87 | 3.16 | 3.45 | 0.38 | 6075 | 7.454 | 3.90 |
| pedestrian | 0.465 | 3.87 | 1.29 | 1.39 | 0.37 | 5542 | 7.051 | 3.20 |
| car | 0.763 | 3.87 | 2.47 | 2.79 | 0.34 | 11181 | 4.324 | 3.70 |

| Class | Mount | Events | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
|---|---|---|---|---|---|---|
| bus/truck | lamppost | 227 | 200/227 | 191/227 | 191/227 | 168/227 |
| pedestrian | lamppost | 167 | 96/167 | 94/167 | 89/167 | 73/167 |
| bus/truck | facade | 212 | 200/212 | 196/212 | 190/212 | 175/212 |
| pedestrian | facade | 188 | 121/188 | 117/188 | 119/188 | 104/188 |

## H1 mechanisms

H1 was originally twin clutter subtraction plus ghost handling. After the first M2 results it was revised to three mechanisms. The ghost and map numbers below were evaluated after that revision.

(a) Twin-based static-clutter subtraction versus blind: null. Static canyon clutter is exactly zero-Doppler. Power maps differ by 1.442e-10. No impairment was added.

(b) Twin-aware ghost handling (image method on known buildings): at identical parameters, no bus/truck track-Pd gain (mean change -0.007 over 96 tuning pairs) and 14–21% fewer unmatched clusters (reviewed point 6.14 vs 5.30 is 14%; pair mean 20.6%). At a fixed false-alarm budget the image method is the one that reaches 2 FA/CPI, and at budget 4 it raises bus/truck track Pd relative to the no-ghost pick.

(c) Map-constrained tracking versus unconstrained EKF: gains. On the earlier budget-4 blind/none detector, bus/truck track Pd 0.824 -> 0.900, pedestrian 0.277 -> 0.449, cross-lane velocity RMSE ~2 -> ~0.5 m/s. The tables above are the same map tracker with image-method ghost handling at each labelled budget.

