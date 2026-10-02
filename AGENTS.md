# AGENTS.md — instructions for coding agents

## Project
sense2connect: digital-twin-aware ISAC (integrated sensing and communication)
for proactive mmWave blockage management in Open RAN. Target venue:
IEEE WoWMoM 2027, regular paper, deadline 1 Dec 2026 (AoE), double-blind.

Pipeline (planned):
1. Sionna RT scene with moving TR 38.901 sensing targets (vehicles, humans).
2. Sensing channel via `RCSSolver`, background via `PathSolver`, combined with
   `Paths.concat()`.
3. Radar processing with `sionna.phy.isac`: twin-based background subtraction,
   CFAR, ghost (multi-bounce) handling, tracking.
4. Comm channel per UE via `PathSolver`; blockage prediction from tracks.
5. xApp logic (near-RT RIC) acting on predictions; E2 latency measured
   separately with OAI (rfsimulator) + FlexRIC.

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
- Do not commit or push unless asked. Show the diff summary first.
  (The repo is not git-initialized yet; `git` commands fail.)
- Double-blind: never write author names, lab, university or the GitHub
  organisation name into code, comments, notebooks, configs or figures.
- Reproducibility: every experiment takes a YAML config from `configs/`,
  uses explicit seeds (`RCSSolver(deterministic=True)` where exactness
  matters), and writes outputs to `results/<experiment>/` (git-ignored).
- After any change to `docker/` or dependencies, rerun
  `scripts/sanity_check.py --require-gpu` and report the output.
- Build gotcha (verified on a fresh build): the first sanity run fails at
  `import mitsuba` with `ImportError: libatomic.so.1` because the image
  lacks `libatomic1`. Fix: add `libatomic1` to the apt-get line in
  `docker/Dockerfile.sionna` and rebuild.
- This server has 2× H100 NVL shared with other users. Experiments are
  locked to GPU 1: keep `NVIDIA_VISIBLE_DEVICES=1` in the `sionna` service
  env of `docker-compose.yml` (it overrides the Dockerfile's `all`).

## Verified facts about Sionna 2.2 (tested, do not re-derive)
- Sionna 2.x PHY uses PyTorch (not TensorFlow). Use `out_type="torch"` or
  `"numpy"` in `paths.cir()`.
- RT variant on GPU must be `cuda_ad_mono_polarized`; `llvm_*` means OptiX/GPU
  is not visible (check `NVIDIA_DRIVER_CAPABILITIES=all`). The variant is
  fixed at first `import sionna.rt`; check it via `mi.variant()`.
- `RCSSolver` reproduces the monostatic radar equation (rel. error ~1e-8)
  and Doppler `f_D = -2 v_r / lambda` (`paths.doppler`, positive = approaching).
- Sensing targets are perfect absorbers for `PathSolver`: blockage is binary,
  even with diffraction enabled. A realistic blockage loss model
  (3GPP TR 38.901 blockage model B, knife-edge) must be applied on top,
  using the target geometry.
- Target Doppler covers rigid translation only (no rotation, no
  micro-Doppler). Do not build features on micro-Doppler.
- `RCSSolver` does not support diffuse reflection or diffraction.
- Moving a target via `position` does NOT set its velocity; set `velocity`
  explicitly every snapshot.
- Reference API usage: the official RCS tutorial
  (https://nvlabs.github.io/sionna/rt/tutorials/RCS.html).

## Code style
- Python 3.12, type hints, docstrings with units (m, s, Hz, dB).
- Small modules under `sim/`; no logic in notebooks beyond plotting.
- Keep GPU memory in mind: batch scene edits (`scene.add([...])`),
  set `samples_per_sp` / `buffer_size_per_sp` from config.
