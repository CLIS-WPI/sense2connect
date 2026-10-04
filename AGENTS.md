# AGENTS.md — instructions for coding agents

## Project
sense2connect: ISAC (integrated sensing and communication) for proactive
inter-cell handover against mmWave blockage in Open RAN. Venue: WCNC 2027
(not blind; see ROADMAP.md).

Pipeline (current):
1. Sionna RT street canyon at 28 GHz, two O-RUs = two cells (oru-0 north
   side lamppost 5 m or facade 8 m; oru-1 lamppost 5 m on the UE sidewalk),
   8x8 arrays at the O-RUs, single-antenna pedestrian UEs.
2. Moving bodies are `TR38901SensingTarget`s (cars, buses, trucks as
   `vehicle-multi-sp`, pedestrians as `human`).
3. Sensing: monostatic radar at oru-0 via `RCSSolver` (TX one element,
   RX 8x8), background via `PathSolver`, ideal full duplex (assumption).
   Processing: OFDM range-Doppler-angle, CA-CFAR, twin-aware ghost
   handling (image method on known buildings), DBSCAN, Doppler-aided
   tracking.
4. Comm: paths traced WITHOUT sensing targets, then 3GPP TR 38.901
   blockage model B applied to every segment of every path.
5. xApp (near-RT RIC): blockage prediction from tracks, proactive
   inter-cell handover; E2 latency measured separately with OAI
   (rfsimulator) + FlexRIC (M4).

## Hard rules
- Work follows ROADMAP.md: one milestone at a time, report, then STOP
  for human review. Never start the next milestone without approval.
- Research integrity rules in ROADMAP.md are non-negotiable.
- All Python runs INSIDE the Docker container (`docker compose run --rm sionna ...`).
  Never pip-install into the host Python.
- Versions are pinned (sionna 2.2.0, sionna-rt 2.2.0, mitsuba 3.9.1,
  drjit 1.5.0, torch 2.9.1). Do not upgrade or add dependencies without asking;
  if one is needed, add it to `docker/requirements.txt` and explain why.
- Do not use sudo, change drivers, Docker daemon config, or anything outside
  this repository without explicit approval.
- Git: never commit or push unless explicitly asked; show the diff summary
  first. `results/` is git-ignored and is never committed.
- Do not edit `paper/` unless asked; numbers for the paper come from
  scripts (`paper/numbers.tex`, see ROADMAP M5), never typed by hand.
- Names: the venue is not blind, so author names, affiliations and
  acknowledgements are allowed in `paper/` (written by the humans). Code,
  comments, configs, notebooks, scripts, figures and result files still
  carry no personal data (no names, e-mail addresses, home paths, user
  names, credentials or grant numbers).
- Reproducibility: every experiment takes a YAML config from `configs/`,
  uses explicit seeds and `deterministic=True` on both solvers, and writes
  outputs to `results/<experiment>/`.
- Ray trace once, process many times: sensing/comm variants, bandwidths
  and tuning sweeps run from the channel cache in `results/cache/`.
  Re-trace only when the scene, trajectories or solver settings change.
- After any change to `docker/` or dependencies, rerun
  `scripts/sanity_check.py --require-gpu` and report the output.
- Build gotcha: the image needs `libatomic1` in the apt-get line of
  `docker/Dockerfile.sionna` (otherwise `import mitsuba` fails with
  `libatomic.so.1`).
- This server has 2x H100 NVL shared with other users. Experiments are
  locked to GPU 1: keep `NVIDIA_VISIBLE_DEVICES=1` in the `sionna` service
  env of `docker-compose.yml`.
- M4 (OAI + FlexRIC) runs in its own branch/worktree and agent session,
  CPU only; it must not touch `sim/`, `results/cache/` or GPU 1 jobs.

## Verified facts about Sionna 2.2 (tested, do not re-derive)
- Sionna 2.x PHY uses PyTorch (not TensorFlow). Use `out_type="torch"` or
  `"numpy"` in `paths.cir()`.
- RT variant on GPU must be `cuda_ad_mono_polarized`; `llvm_*` means OptiX/GPU
  is not visible (check `NVIDIA_DRIVER_CAPABILITIES=all`). The variant is
  fixed at first `import sionna.rt`; check it via `mi.variant()`.
- `RCSSolver` reproduces the monostatic radar equation (rel. error ~1e-8)
  and Doppler `f_D = -2 v_r / lambda` (`paths.doppler`, positive = approaching).
- Sensing targets are perfect absorbers for `PathSolver`: blockage is binary,
  even with diffraction enabled. Hence comm paths are traced without targets
  and model B is applied on top.
- Target Doppler covers rigid translation only (no rotation, no
  micro-Doppler). Do not build features on micro-Doppler.
- `RCSSolver` does not support diffuse reflection or diffraction.
- Moving a target via `position` does NOT set its velocity; set `velocity`
  explicitly every snapshot.
- Passing length/width/height to `TR38901SensingTarget` scales the cuboid
  and the scattering points together.
- With `deterministic=True` on `RCSSolver` and `PathSolver`, a paired run in
  one process is bitwise identical; across processes comm-path floats may
  differ by up to ~1.5e-5 (compare with tolerance).
- `Paths.objects` integers and names of nameless meshes are NOT stable
  across scene loads; compare paths across processes with the load-stable
  `path_key` (floor / building / bounding-box label), not raw ids.
- Static clutter is exactly zero-Doppler in simulation, so blind
  (zero-Doppler) and twin-based clutter removal give identical results.
  This is a known null result; do not add impairments to change it.
- Performance: during tracing the GPU is nearly idle (~2% SM); time goes to
  Dr.Jit kernel compilation (path counts change every snapshot), scene
  edits and the solvers. Several processes on one GPU were slower than one
  (measured). Cached processing runs at ~0.01 s/snapshot.
- Reference API usage: the official RCS tutorial
  (https://nvlabs.github.io/sionna/rt/tutorials/RCS.html).

## Code style
- Python 3.12, type hints, docstrings with units (m, s, Hz, dB).
- Small modules under `sim/`; no logic in notebooks beyond plotting.
- GPU code (torch) keeps a NumPy reference implementation with an
  equality test.
- Keep GPU memory in mind: batch scene edits (`scene.add([...])`),
  set `samples_per_sp` / `buffer_size_per_sp` from config.