"""Seed sets for paper 2: tuning 101-105 and development 1001-1010 (configs/seeds.yaml),
held-out 2001-2010 (configs/seeds_heldout.yaml; pre-fix record) and held-out2 3001-3010
(configs/p2.yaml). S2C_EVAL_SET in {dev, heldout, heldout2} selects the evaluation set;
the tag is the same name. Reads the YAML files directly (scripts/seedsets.py, a paper-1
file, only knows dev and heldout and is not modified).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = ("dev", "heldout", "heldout2")


def eval_tag() -> str:
    v = os.environ.get("S2C_EVAL_SET", "dev")
    if v not in SETS:
        raise SystemExit(f"S2C_EVAL_SET must be one of {SETS}, got {v!r}")
    return v


def load() -> dict:
    from sim.scenes.config import load_yaml

    s = load_yaml(ROOT / "configs" / "seeds.yaml")
    h = load_yaml(ROOT / "configs" / "seeds_heldout.yaml")["heldout"]
    h2 = load_yaml(ROOT / "configs" / "p2.yaml")["heldout2"]
    out = {"tuning": [int(x) for x in s["tuning"]], "development": [int(x) for x in s["evaluation"]], "heldout": [int(x) for x in h],
           "heldout2": [int(x) for x in h2]}
    allsets = [set(out[k]) for k in out]
    for i in range(len(allsets)):
        for j in range(i + 1, len(allsets)):
            if allsets[i] & allsets[j]:
                raise SystemExit("seed sets overlap")
    out["evaluation"] = {"dev": out["development"], "heldout": out["heldout"], "heldout2": out["heldout2"]}[eval_tag()]
    return out
