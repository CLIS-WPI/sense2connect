"""Comm-only model-B traces for M3. No RCS. Writes comm_trace.json per job."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml
from sim.scenes.loop import run_scenario
from sim.scenes.traffic import prepare_scenario

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")


def main() -> None:
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    from seedsets import load_seeds  # S2C_EVAL_SET selects development or held-out seeds
    seeds = load_seeds()
    jobs = [
        (mount, density, int(seed))
        for seed in list(seeds["tuning"]) + list(seeds["evaluation"])
        for mount in MOUNTS
        for density in DENSITIES
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.map(_one, [(raw, mount, density, seed) for mount, density, seed in jobs])
    print("comm traces", len(jobs), flush=True)


def _one(item) -> None:
    raw, mount, density, seed = item
    directory = ROOT / "results" / "cache" / mount / density / f"seed_{seed}"
    path = directory / "comm_trace.json"
    if path.exists():
        print(f"skip {mount} {density} {seed}", flush=True)
        return
    print(f"comm {mount} {density} {seed}", flush=True)
    scenario = prepare_scenario(raw, seed=seed, mount=mount, density=density, duration_s=60.0, dt_s=0.1)
    scenario["blockage"]["min_gap_s"] = 0.5
    result = run_scenario(
        scenario,
        directory / "_comm_tmp",
        render=False,
        comm_only=True,
        write_outputs=False,
        keep_snapshot_rows=False,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "mount": mount,
                "density": density,
                "dt_s": 0.1,
                "n_snapshots": int(result["n_snapshots"]),
                "n_ue": len(scenario["ues"]),
                "n_oru": len(scenario["orus"]),
                "ue_trace": result["ue_trace"],
                "ue_events": result["ue_events"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {mount} {density} {seed}", flush=True)


if __name__ == "__main__":
    main()
