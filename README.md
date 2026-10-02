# sense2connect

Digital-twin-aware ISAC for proactive blockage management in Open RAN,
simulated with Sionna 2.2 (RT `RCSSolver` + PHY `sionna.phy.isac`).

> Double-blind: keep this repository private until the review is over.
> For the submission, link an anonymized mirror (e.g. anonymous.4open.science)
> and keep names, affiliations and metadata out of code, comments and notebooks.

## Requirements
- NVIDIA GPU (developed on H100), recent NVIDIA driver
- Docker + NVIDIA Container Toolkit

## Quick start
```bash
docker compose build
docker compose run --rm sionna python scripts/sanity_check.py --require-gpu
```
Expected: all checks `PASS`, variant `cuda_ad_mono_polarized`, and
`results/sanity_range_doppler.png` with the peak on the red cross.

Jupyter Lab (bound to localhost; tunnel with `ssh -L 8888:localhost:8888 <server>`):
```bash
JUPYTER_TOKEN=<secret> docker compose up sionna
```

## Troubleshooting
- Variant is `llvm_*` instead of `cuda_*`, or OptiX errors: the container
  needs `NVIDIA_DRIVER_CAPABILITIES=all` (set in Dockerfile and compose).
- PyTorch cannot use the GPU: the host driver may be too old for CUDA 12.8;
  rebuild with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126`.

## Layout
| Path | Content |
|---|---|
| `docker/` | Dockerfile and pinned requirements (Sionna 2.2.0) |
| `scripts/sanity_check.py` | Environment and physics checks, benchmark |
| `sim/scenes` | Scenes and target trajectories |
| `sim/sensing` | Background subtraction, CFAR, tracking |
| `sim/comm` | Comm channel, blockage, beams |
| `xapp/` | Blockage-prediction xApp |
| `oran_latency/` | E2 loop latency (OAI rfsim + FlexRIC) |
| `configs/` | Experiment configs |
| `results/` | Outputs (git-ignored) |

## Known modelling limits (Sionna 2.2)
- Sensing targets are perfect absorbers for `PathSolver`: blockage is binary.
  A blockage loss model (e.g. 3GPP TR 38.901 model B) is applied on top.
- Target Doppler covers rigid translation only (no rotation / micro-Doppler).
