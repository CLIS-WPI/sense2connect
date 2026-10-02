"""Load the M1 YAML scenario and the seed list."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_scenario(path: Path | None = None) -> dict[str, Any]:
    """Return the scenario mapping and check that its seed is listed."""
    scenario_path = path if path is not None else ROOT / "configs" / "m1_scenario.yaml"
    scenario = load_yaml(scenario_path)
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    allowed = set(seeds["tuning"]) | set(seeds["evaluation"])
    seed = int(scenario["seed"])
    if seed not in allowed:
        raise ValueError(f"seed {seed} is not in configs/seeds.yaml")
    return scenario


def subcarrier_spacing_hz(numerology: int) -> float:
    """3GPP subcarrier spacing [Hz] for numerology ``mu``."""
    if numerology < 0:
        raise ValueError("numerology must be >= 0")
    return 15_000.0 * (2 ** numerology)
