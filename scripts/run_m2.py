"""Tune the monostatic radar on the tuning seeds and score the evaluation seeds.

Clutter removal (blind, twin) and ghost handling (none, twin) share one
ray-tracing pass. CFAR is never run without thermal noise. The match gate
is fixed; CFAR and the tracker are what the tuning seeds choose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.scenes.config import load_yaml  # noqa: E402
from sim.scenes.traffic import prepare_scenario  # noqa: E402
from sim.sensing.cluster import cluster_detections  # noqa: E402
from sim.sensing.ghost import reject_ghosts  # noqa: E402
from sim.sensing.metrics import CLASSES, empty_score, summarize, update_score  # noqa: E402
from sim.sensing.radar import scan_scenario  # noqa: E402
from sim.sensing.track import Tracker  # noqa: E402
from sim.sensing.waveform import waveform_from_config  # noqa: E402

MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")
CLUTTER = ("blind", "twin")
CFAR_GRID = [
    {"guard": 2, "train": train, "pfa": pfa}
    for train in (4, 8)
    for pfa in (1e-3, 1e-4, 1e-5)
]
EPS_M = (3.0, 5.0)
COAST = (2, 4)
ASSOCIATION_M = (6.0, 10.0)
FA_CAP = 2.0


def _add(total: dict[str, Any], part: dict[str, Any]) -> None:
    total["frames"] += part["frames"]
    total["false_clusters"] += part["false_clusters"]
    total["confirmed_updates"] += part["confirmed_updates"]
    total["unmatched_confirmed"] += part["unmatched_confirmed"]
    for name in CLASSES:
        for key, value in part["classes"][name].items():
            total["classes"][name][key] += value


def _replay(
    frames: list[dict[str, Any]],
    *,
    n_subcarriers: int,
    clutter: str,
    cfar: dict[str, float],
    eps_m: float,
    coast: int,
    association_m: float,
    ghost: bool,
    dt_s: float,
    gate_m: float,
    walls: list[float],
    min_samples: int,
    confirm_hits: int,
) -> dict[str, Any]:
    score = empty_score()
    tracker = Tracker(
        dt_s=dt_s,
        association_gate_m=association_m,
        coast_frames=coast,
        confirm_hits=confirm_hits,
    )
    key = (int(n_subcarriers), clutter, int(cfar["train"]), float(cfar["pfa"]))
    for frame in frames:
        detections = list(frame["detections"][key])
        if ghost:
            detections = reject_ghosts(detections, walls, gate_m)
        clusters = cluster_detections(detections, eps_m, min_samples)
        alive = tracker.step(clusters)
        tracks = [
            {
                "x_m": float(track.state[0]),
                "y_m": float(track.state[1]),
                "vx_mps": float(track.state[2]),
                "vy_mps": float(track.state[3]),
                "confirmed": bool(track.confirmed),
            }
            for track in alive
        ]
        update_score(score, frame["ground_truth"], clusters, tracks, gate_m)
    return score


def _candidates() -> list[dict[str, Any]]:
    rows = []
    for clutter in CLUTTER:
        for ghost in (False, True):
            for cfar in CFAR_GRID:
                for eps_m in EPS_M:
                    for coast in COAST:
                        for association_m in ASSOCIATION_M:
                            rows.append(
                                {
                                    "clutter": clutter,
                                    "ghost": ghost,
                                    "cfar": cfar,
                                    "eps_m": eps_m,
                                    "coast": coast,
                                    "association_m": association_m,
                                }
                            )
    return rows


def _rank(summary: dict[str, Any]) -> tuple:
    pd = summary["classes"]["bus/truck"]["track_pd"]
    pd_value = -1.0 if pd is None else float(pd)
    fa = float(summary["false_clusters_per_cpi"])
    return (fa <= FA_CAP, pd_value, -fa)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _scan(raw: dict, seeds: list[int], waves, cfar_grid: list[dict[str, float]]) -> list[dict[str, Any]]:
    wave_list = list(waves) if isinstance(waves, list) else [waves]
    rows = []
    for mount in MOUNTS:
        for density in DENSITIES:
            for seed in seeds:
                scenario = prepare_scenario(
                    raw,
                    seed=int(seed),
                    mount=mount,
                    density=density,
                    duration_s=float(raw["sensing_radar"]["duration_s"]),
                    dt_s=wave_list[0].cpi_s,
                )
                counts = ",".join(str(item.n_subcarriers) for item in wave_list)
                print(
                    f"scan mount={mount} density={density} seed={seed} "
                    f"frames={scenario['n_snapshots']} subcarriers={counts}",
                    flush=True,
                )
                frames = scan_scenario(scenario, wave_list, cfar_grid)
                rows.append(
                    {
                        "mount": mount,
                        "density": density,
                        "seed": int(seed),
                        "dt_s": float(scenario["dt_s"]),
                        "frames": frames,
                    }
                )
    return rows


def _tune(cases: list[dict[str, Any]], spec: dict[str, Any]) -> dict[tuple[str, bool], dict[str, Any]]:
    chosen: dict[tuple[str, bool], dict[str, Any]] = {}
    gate_m = float(spec["match_gate_m"])
    walls = [float(value) for value in spec["wall_y_m"]]
    min_samples = int(spec["cluster"]["min_samples"])
    confirm_hits = int(spec["tracker"]["confirm_hits"])
    grouped: dict[tuple[str, bool], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for candidate in _candidates():
        total = empty_score()
        for case in cases:
            part = _replay(
                case["frames"],
                n_subcarriers=1024,
                clutter=candidate["clutter"],
                cfar=candidate["cfar"],
                eps_m=float(candidate["eps_m"]),
                coast=int(candidate["coast"]),
                association_m=float(candidate["association_m"]),
                ghost=bool(candidate["ghost"]),
                dt_s=float(case["dt_s"]),
                gate_m=gate_m,
                walls=walls,
                min_samples=min_samples,
                confirm_hits=confirm_hits,
            )
            _add(total, part)
        summary = summarize(total)
        key = (str(candidate["clutter"]), bool(candidate["ghost"]))
        grouped.setdefault(key, []).append((candidate, summary))
    for key, rows in grouped.items():
        best = max(rows, key=lambda item: _rank(item[1]))
        chosen[key] = {"params": best[0], "tuning": best[1]}
        print(f"chosen {key[0]} ghost={key[1]} {_rank(best[1])}", flush=True)
    return chosen


def _evaluate(cases: list[dict[str, Any]], chosen: dict, spec: dict[str, Any], n_subcarriers: int) -> dict[str, Any]:
    gate_m = float(spec["match_gate_m"])
    walls = [float(value) for value in spec["wall_y_m"]]
    min_samples = int(spec["cluster"]["min_samples"])
    confirm_hits = int(spec["tracker"]["confirm_hits"])
    tables = []
    for (clutter, ghost), item in chosen.items():
        params = item["params"]
        for mount in MOUNTS:
            total = empty_score()
            for case in cases:
                if case["mount"] != mount:
                    continue
                part = _replay(
                    case["frames"],
                    n_subcarriers=n_subcarriers,
                    clutter=clutter,
                    cfar=params["cfar"],
                    eps_m=float(params["eps_m"]),
                    coast=int(params["coast"]),
                    association_m=float(params["association_m"]),
                    ghost=ghost,
                    dt_s=float(case["dt_s"]),
                    gate_m=gate_m,
                    walls=walls,
                    min_samples=min_samples,
                    confirm_hits=confirm_hits,
                )
                _add(total, part)
            tables.append({"clutter": clutter, "ghost": ghost, "mount": mount, "summary": summarize(total)})
    return {"rows": tables}


def _report(wave_note: str, chosen: dict, evaluations: dict[str, dict], spec: dict) -> str:
    lines = [
        "# M2 report",
        "",
        "Status: ready for review. This milestone is not marked done. Nothing was committed or pushed.",
        "",
        "Monostatic radar at `oru-0` on `configs/m2_scenario.yaml`. "
        "The second O-RU is the other cell and is not a second radar. "
        "Assumption: ideal full duplex, so the colocated transmitter and receiver include no self-interference. "
        "Paths shorter than 0.15 m are dropped.",
        "",
        wave_note,
        "",
        f"Each run is {spec['duration_s']} s and each frame is one CPI. "
        f"A detection or track matches a ground-truth identity inside {spec['match_gate_m']} m. "
        "That gate is the 8×8 broadside beamwidth at a few tens of metres (about 14° at 30 m is about 7 m) "
        "and was not tuned. A wrap starts a new identity, so the old lap is a different target. "
        "Vehicles use the 3GPP five-scatterer model; DBSCAN runs before the tracker. "
        "CFAR always sees thermal noise (30 dBm, 7 dB noise figure, the waveform bandwidth).",
        "",
        "Tuning maximises bus/truck track probability of detection on seeds 101–105, "
        f"subject to at most {FA_CAP:.0f} unmatched clusters per CPI. "
        "Each clutter × ghost variant sees the same grid. "
        "The evaluation table uses seeds 1001–1010 and those frozen parameters. "
        "2048 subcarriers were not retuned.",
        "",
        "## Chosen parameters",
        "",
        "| Clutter | Ghost | Pfa | Train | DBSCAN eps [m] | Coast | Association [m] | Tuning bus/truck track Pd | FA / CPI |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (clutter, ghost), item in chosen.items():
        params = item["params"]
        summary = item["tuning"]
        lines.append(
            f"| {clutter} | {'twin' if ghost else 'none'} | {params['cfar']['pfa']:.0e} | {params['cfar']['train']} | "
            f"{params['eps_m']:.0f} | {params['coast']} | {params['association_m']:.0f} | "
            f"{_fmt(summary['classes']['bus/truck']['track_pd'])} | {_fmt(summary['false_clusters_per_cpi'], 2)} |"
        )
    for label, table in evaluations.items():
        lines.extend(["", f"## Evaluation, {label}", ""])
        lines.append(
            "| Clutter | Ghost | Mount | Class | GT samples | Det. Pd | Track Pd | Pos. RMSE [m] | Vel. RMSE [m/s] | OSPA [m] |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in table["rows"]:
            for name in CLASSES:
                stats = row["summary"]["classes"][name]
                lines.append(
                    f"| {row['clutter']} | {'twin' if row['ghost'] else 'none'} | {row['mount']} | {name} | "
                    f"{stats['gt_samples']} | {_fmt(stats['detection_pd'])} | {_fmt(stats['track_pd'])} | "
                    f"{_fmt(stats['position_rmse_m'], 2)} | {_fmt(stats['velocity_rmse_mps'], 2)} | {_fmt(stats['ospa_m'], 2)} |"
                )
        lines.extend(["", "| Clutter | Ghost | Mount | False clusters / CPI | Ghost-track rate |", "|---|---|---|---|---|"])
        for row in table["rows"]:
            lines.append(
                f"| {row['clutter']} | {'twin' if row['ghost'] else 'none'} | {row['mount']} | "
                f"{_fmt(row['summary']['false_clusters_per_cpi'], 2)} | {_fmt(row['summary']['ghost_track_rate'])} |"
            )
    lines.extend(
        [
            "",
            "False clusters and the ghost-track rate are not split by class: an unmatched track has no blocker label. "
            "Class OSPA uses that class's ground truth and the tracks matched to it; unmatched ground truth pays the 8 m cutoff.",
            "",
            "Priority is bus/truck track probability of detection. Cars and pedestrians are reported beside it.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    seeds = load_yaml(ROOT / "configs" / "seeds.yaml")
    spec = raw["sensing_radar"]
    output = ROOT / "results" / "M2"
    output.mkdir(parents=True, exist_ok=True)
    wave = waveform_from_config(raw)
    tuning_cases = _scan(raw, [int(seed) for seed in seeds["tuning"]], wave, CFAR_GRID)
    chosen = _tune(tuning_cases, spec)
    del tuning_cases
    frozen = []
    seen = set()
    for item in chosen.values():
        key = (int(item["params"]["cfar"]["train"]), float(item["params"]["cfar"]["pfa"]))
        if key not in seen:
            seen.add(key)
            frozen.append(item["params"]["cfar"])
    wide = waveform_from_config(raw, n_subcarriers=int(spec["waveform"]["n_subcarriers_alt"]))
    cases = _scan(raw, [int(seed) for seed in seeds["evaluation"]], [wave, wide], frozen)
    evaluations = {
        "1024 subcarriers": _evaluate(cases, chosen, spec, wave.n_subcarriers),
        "2048 subcarriers": _evaluate(cases, chosen, spec, wide.n_subcarriers),
    }
    del cases
    note = (
        f"Waveform: numerology {wave.numerology}, PRF {wave.prf_hz:.0f} Hz, "
        f"CPI {wave.cpi_slots} slots ({wave.cpi_s * 1e3:.0f} ms), "
        f"v_max {wave.v_max_mps:.1f} m/s, v_res {wave.v_res_mps:.2f} m/s. "
        f"Default bandwidth {wave.bandwidth_hz / 1e6:.1f} MHz ({wave.n_subcarriers} subcarriers); "
        f"the second bandwidth is {waveform_from_config(raw, 2048).bandwidth_hz / 1e6:.1f} MHz. "
        f"Sensing overhead {wave.overhead:.4f} = {wave.sensing_symbols_per_slot}/{wave.symbols_per_slot} symbols. "
        "Transmit array: one isotropic element. Receive array: 8×8, 0.5 wavelength. "
        "The communication O-RU array in this config stays 8×8."
    )
    text = _report(note, chosen, evaluations, spec)
    (output / "report.md").write_text(text, encoding="utf-8")
    payload = {
        "assumption": "monostatic ideal full duplex, no self-interference",
        "chosen": {
            f"{clutter}-ghost-{ghost}": {
                "params": item["params"],
                "tuning": item["tuning"],
            }
            for (clutter, ghost), item in chosen.items()
        },
        "evaluation": evaluations,
    }
    (output / "metrics.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
