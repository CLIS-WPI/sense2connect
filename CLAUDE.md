# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The hard rules, pinned versions, and verified Sionna 2.2 facts are in
AGENTS.md above. Milestones, research-integrity rules, hypotheses and
findings are in `ROADMAP.md`; read the current milestone (first one not
`DONE`) before starting work.

## Commands
Everything runs in the container (GPU 1 only, repo mounted at `/workspace`):

```bash
docker compose build
docker compose run --rm sionna python scripts/sanity_check.py --require-gpu   # env + physics checks
docker compose run --rm sionna python scripts/test_m3.py                      # one test file
docker compose run --rm sionna python scripts/test_m3.py PhyTest.test_steering_is_unit_norm  # one test
docker compose run --rm sionna python scripts/test_perf.py                    # GPU vs NumPy equality, cache vs direct solve
```

- Tests are plain `unittest` scripts (`scripts/test_*.py`, no pytest);
  `test_perf.py` is a set of functions called from `main()`.
- `test_m1.py`/`test_m2.py`/`test_perf.py` trace scenes and need the GPU;
  `test_m3.py` is pure logic.
- Scripts insert the repo root into `sys.path` themselves; run them as
  `python scripts/<name>.py` from `/workspace`.

## Architecture
Data flows in stages, each cached under `results/` so later stages never
re-trace:

1. **Scenario** (`sim/scenes/`): `config.load_yaml` + `traffic.prepare_scenario`
   build the street canyon, the two O-RUs and the fleet from a YAML config
   (`configs/m2_scenario.yaml` is the current radio/scene config;
   `m1_scenario.yaml` is kept for repeating M1). `loop.py` steps the scene
   and solves comm paths on a target-free copy, then applies model B
   (`sim/comm/blockage.py`, torch twin in `blockage_torch.py`).
2. **Sensing trace** (`sim/sensing/trace.py`): one job = `(seed, mount,
   density)`. Runs `RCSSolver`/`PathSolver`, pads paths to `max_paths`
   (stable shapes avoid Dr.Jit recompiles), and writes the channel cache
   via `sim/sensing/cache.py` to `results/cache/<mount>/<density>/seed_<seed>`.
   `scripts/run_batch.py` / `run_m2_review.py` drive many jobs.
3. **Radar processing from cache** (`sim/sensing/`): waveform → `radar.py`
   (range-Doppler-angle, GPU path in `process_torch.py`) → `cfar.py` →
   `ghost.py` (image method) → `cluster.py` (DBSCAN) → `track.py` (EKF) or
   `track_map.py` (map-constrained) → `metrics.py`. Scripts `run_m2*.py`
   sweep these from the cache.
4. **Comm traces for M3**: `scripts/cache_comm.py` writes model-B comm
   traces (`comm_trace.json`) per job, no RCS.
5. **xApp** (`xapp/`): `tracks.py` replays the map tracker from cached
   detections; `predict.py` extrapolates tracks and predicts blockage
   windows per cell; `policy.py` decides handover/return with a hold timer;
   `baselines.py` has A3, RSRP-trend and oracle; `simulate.py` runs the
   10 ms comm timeline with E2 delay and handover interruption;
   `sim/comm/phy.py` gives SNR/throughput with sensing overhead.
   `scripts/run_m3.py` tunes on tuning seeds and evaluates on evaluation
   seeds (`configs/m3.yaml`, `configs/seeds.yaml`).

Provenance (`sim/sensing/provenance.py`): every cache entry stores git
commit, dirty-tree hash and config hash; loaders raise `CacheRefused` on a
mismatch. Editing tracked source therefore invalidates caches for loaders
that check it; `scripts/restamp_cache.py` restamps finished traces only
when the ray traces themselves are unaffected.

Each milestone ends with `results/<milestone>/report.md`, then stop for
human review.

## Claude Code specifics
- Read `ROADMAP.md` (current milestone) at the start of every session.
- Start every milestone task with a written plan; wait for approval
  before running anything longer than a few minutes.
- Long runs go to the background with logs in `results/<milestone>/logs/`;
  poll them instead of blocking the session.
- GPU: only GPU 1. Never use GPU 0; the server is shared.
- Keep an explicit todo list for the current goal and update it.

## Provenance rules
- Never run `scripts/restamp_cache.py` without explicit human approval.
  Each use is logged in `results/provenance_log.md` with the reason, the
  files changed, and why the cached stage is unaffected.
- Provenance should be stage-scoped: a cache's code hash covers only the
  source files that produce it (e.g. sensing traces: `sim/scenes/`,
  `sim/sensing/trace.py`, `sim/sensing/cache.py`, scene config), so edits
  in `xapp/` or analysis scripts do not invalidate ray-trace caches.
  Propose this change in the M3 plan before implementing it.