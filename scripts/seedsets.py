"""Seed sets for the evaluation pipeline.

configs/seeds.yaml holds the tuning seeds (101-105) and the development
seeds (1001-1010, called "evaluation" there). configs/seeds_heldout.yaml
holds the held-out test seeds (2001-2010). The environment variable
S2C_EVAL_SET selects which set the pipeline evaluates on:
- "dev" (default): "evaluation" = development seeds;
- "heldout": "evaluation" = held-out seeds; the development seeds stay
  available as "development". Tuning seeds never change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def eval_set() -> str:
    value = os.environ.get("S2C_EVAL_SET", "dev")
    if value not in ("dev", "heldout"):
        raise SystemExit(f"S2C_EVAL_SET must be 'dev' or 'heldout', got {value!r}")
    return value


def load_seeds() -> dict[str, Any]:
    from sim.scenes.config import load_yaml

    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    held = [int(s) for s in load_yaml(ROOT / "configs" / "seeds_heldout.yaml")["heldout"]]
    if set(held) & (set(int(s) for s in seeds["tuning"]) | set(int(s) for s in seeds["evaluation"])):
        raise SystemExit("held-out seeds overlap the tuning or development seeds")
    out = dict(seeds)
    out["development"] = list(seeds["evaluation"])
    out["heldout"] = held
    if eval_set() == "heldout":
        out["evaluation"] = held
    return out
