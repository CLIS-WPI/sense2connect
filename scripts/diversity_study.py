"""M1.5 diversity study on the tuning seeds.

Part 1 uses configs/m1_scenario.yaml (one O-RU). Part 2 uses
configs/m2_scenario.yaml (two O-RUs). Both are 60 s at dt = 0.1 s.
Geometry is not changed after the runs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.diversity import los_loss_db, oracle_link, other_link, power_ratio_db, serving_link  # noqa: E402
from sim.comm.events import apply_hysteresis  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from sim.scenes.loop import _ue_events, run_scenario  # noqa: E402
from sim.scenes.traffic import prepare_scenario  # noqa: E402

DURATION_S = 60.0
DT_S = 0.1
MIN_GAP_S = 0.5
MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")
KIND_GROUP = {"car": "car", "bus": "bus/truck", "truck": "bus/truck", "pedestrian": "pedestrian"}
CLASSES = ("bus/truck", "pedestrian", "car")


def _kind(event: dict) -> str:
    kind = event.get("blocker_kind")
    if kind not in KIND_GROUP:
        return "unknown"
    return KIND_GROUP[kind]


def _quantile(values: list[float]) -> dict[float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return {}
    array = np.sort(np.asarray(finite, dtype=np.float64))
    return {percent: float(np.quantile(array, percent)) for percent in (0.1, 0.5, 0.9)}


def _fmt_cdf(values: list[float]) -> str:
    table = _quantile(values)
    if not table:
        return "—"
    return " / ".join(f"{table[percent]:.2f}" for percent in (0.1, 0.5, 0.9))


def _run(raw: dict, seed: int, mount: str, density: str, output: Path) -> dict:
    scenario = prepare_scenario(
        raw,
        seed=int(seed),
        mount=mount,
        density=density,
        duration_s=DURATION_S,
        dt_s=DT_S,
    )
    scenario["blockage"]["min_gap_s"] = 0.0
    print(
        f"run mount={mount} density={density} seed={seed} orus={len(scenario['orus'])} "
        f"snapshots={scenario['n_snapshots']}",
        flush=True,
    )
    result = run_scenario(
        scenario,
        output / "_tmp",
        render=False,
        comm_only=True,
        write_outputs=False,
        keep_snapshot_rows=False,
    )
    return {
        "seed": int(seed),
        "mount": mount,
        "density": density,
        "height_m": float(scenario["oru_height_m"]),
        "n_ue": len(scenario["ues"]),
        "n_snapshots": int(result["n_snapshots"]),
        "dt_s": float(result["dt_s"]),
        "n_oru": len(scenario["orus"]),
        "raw_events": result["ue_events_raw"],
        "trace": result["ue_trace"],
    }


def _by_link(trace: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(row["ue"], row["oru"], int(row["snapshot"])): row for row in trace}


def _by_ue_snap(trace: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in trace:
        grouped.setdefault((row["ue"], int(row["snapshot"])), []).append(row)
    return grouped


def _active_snaps(event: dict, links: dict[tuple[str, str, int], dict]) -> list[dict]:
    """Snapshots of a 10 dB event that are still at or above 10 dB."""
    chosen = []
    for snap in range(int(event["start_snapshot"]), int(event["end_snapshot"]) + 1):
        row = links.get((event["ue"], event["oru"], snap))
        if row is None or los_loss_db(row) < 10.0:
            continue
        chosen.append(row)
    return chosen


def best_alt_db(event: dict, links: dict[tuple[str, str, int], dict]) -> float | None:
    """Worst ``10 log10(max non-LoS post-model-B power / unblocked LoS)`` [dB]."""
    values = []
    for row in _active_snaps(event, links):
        values.append(power_ratio_db(row.get("best_available_alt_power"), row.get("los_power")))
    if not values:
        return None
    return float(min(values))


def _los10(events: list[dict]) -> list[dict]:
    return [
        event
        for event in events
        if event["metric"] == "los" and abs(float(event["threshold_db"]) - 10.0) < 1e-9
    ]


def part1(rows: list[dict]) -> dict[str, dict]:
    """Best post-model-B non-LoS path on the single O-RU, per mount."""
    pooled: dict[str, list[float]] = {mount: [] for mount in MOUNTS}
    removed = {mount: 0 for mount in MOUNTS}
    missing = {mount: 0 for mount in MOUNTS}
    counts = {mount: 0 for mount in MOUNTS}
    for row in rows:
        merged = apply_hysteresis(row["raw_events"], MIN_GAP_S, float(row["dt_s"]))
        links = _by_link(row["trace"])
        for event in _los10(merged):
            counts[row["mount"]] += 1
            value = best_alt_db(event, links)
            if value is None:
                missing[row["mount"]] += 1
            elif not math.isfinite(value):
                removed[row["mount"]] += 1
            else:
                pooled[row["mount"]].append(value)
    return {
        mount: {
            "n_events": counts[mount],
            "missing": missing[mount],
            "removed": removed[mount],
            "samples_db": pooled[mount],
        }
        for mount in MOUNTS
    }


def _serving_events(row: dict) -> list[dict]:
    """10 dB LoS events whose O-RU is the strongest unblocked link at the start."""
    merged = apply_hysteresis(row["raw_events"], MIN_GAP_S, float(row["dt_s"]))
    grouped = _by_ue_snap(row["trace"])
    chosen = []
    for event in _los10(merged):
        links = grouped.get((event["ue"], int(event["start_snapshot"])), [])
        if not links:
            continue
        if str(serving_link(links)["oru"]) == str(event["oru"]):
            chosen.append(event)
    return chosen


def _other_record(event: dict, row: dict) -> dict | None:
    links = _by_link(row["trace"])
    grouped = _by_ue_snap(row["trace"])
    active = _active_snaps(event, links)
    if not active:
        return None
    unblocked = True
    ratios = []
    for sample in active:
        snap_links = grouped.get((event["ue"], int(sample["snapshot"])), [])
        other = other_link(snap_links, str(event["oru"]))
        if other is None or los_loss_db(other) >= 3.0:
            unblocked = False
        if other is None:
            ratios.append(-math.inf)
        else:
            ratios.append(power_ratio_db(other.get("blocked_power"), sample.get("los_power")))
    return {
        "kind": _kind(event),
        "other_unblocked": unblocked,
        "ratio_db": float(min(ratios)) if ratios else None,
    }


def part2_events(rows: list[dict]) -> dict:
    """Other-O-RU remedy during serving-link 10 dB LoS events."""
    by_mount: dict[str, list[dict]] = {mount: [] for mount in MOUNTS}
    for row in rows:
        if row["n_oru"] < 2:
            continue
        for event in _serving_events(row):
            record = _other_record(event, row)
            if record is not None:
                by_mount[row["mount"]].append(record)
    summary = {}
    for mount, records in by_mount.items():
        classes = {}
        for kind in CLASSES:
            subset = [item for item in records if item["kind"] == kind]
            ratios = [float(item["ratio_db"]) for item in subset if item["ratio_db"] is not None]
            finite = [value for value in ratios if math.isfinite(value)]
            classes[kind] = {
                "n_events": len(subset),
                "unblocked": sum(1 for item in subset if item["other_unblocked"]),
                "ratio_removed": len(ratios) - len(finite),
                "ratios_db": finite,
            }
        ratios = [float(item["ratio_db"]) for item in records if item["ratio_db"] is not None]
        finite = [value for value in ratios if math.isfinite(value)]
        summary[mount] = {
            "n_events": len(records),
            "unblocked": sum(1 for item in records if item["other_unblocked"]),
            "ratio_removed": len(ratios) - len(finite),
            "ratios_db": finite,
            "classes": classes,
        }
    return summary


def _both_blocked(row: dict) -> dict[str, int]:
    grouped = _by_ue_snap(row["trace"])
    oru_names = sorted({item["oru"] for item in row["trace"]})
    ues = sorted({ue for ue, _snap in grouped})
    both = 0
    serving_outage = 0
    serving_no_remedy = 0
    total = 0
    for ue in ues:
        for snap in range(int(row["n_snapshots"])):
            links = grouped.get((ue, snap), [])
            total += 1
            by_oru = {item["oru"]: item for item in links}
            losses = [los_loss_db(by_oru[name]) if name in by_oru else math.inf for name in oru_names]
            if oru_names and all(loss >= 10.0 for loss in losses):
                both += 1
            if not links:
                serving_outage += 1
                serving_no_remedy += 1
                continue
            serving = serving_link(links)
            if los_loss_db(serving) < 10.0:
                continue
            serving_outage += 1
            other = other_link(links, str(serving["oru"]))
            if other is None or los_loss_db(other) >= 10.0:
                serving_no_remedy += 1
    return {
        "total": total,
        "both": both,
        "serving_outage": serving_outage,
        "serving_no_remedy": serving_no_remedy,
    }


def part2_time(rows: list[dict]) -> dict[str, dict[str, int]]:
    pooled = {mount: {"total": 0, "both": 0, "serving_outage": 0, "serving_no_remedy": 0} for mount in MOUNTS}
    for row in rows:
        if row["n_oru"] < 2:
            continue
        counts = _both_blocked(row)
        bucket = pooled[row["mount"]]
        for key, value in counts.items():
            bucket[key] += int(value)
    return pooled


def _selected_rate(row: dict, mode: str) -> dict:
    grouped = _by_ue_snap(row["trace"])
    ues = sorted({ue for ue, _snap in grouped})
    synthetic = []
    times = [index * float(row["dt_s"]) for index in range(int(row["n_snapshots"]))]
    for ue in ues:
        for snap in range(int(row["n_snapshots"])):
            links = grouped.get((ue, snap), [])
            if not links:
                loss = math.inf
                blocker_id = None
                kind = None
            else:
                chosen = serving_link(links) if mode == "fixed" else oracle_link(links)
                loss = los_loss_db(chosen)
                blocker_id = chosen.get("los_blocker_id")
                kind = chosen.get("los_blocker_kind")
            synthetic.append(
                {
                    "ue": ue,
                    "oru": "selected",
                    "snapshot": snap,
                    "los_loss_db": loss,
                    "los_blocker_id": blocker_id,
                    "los_blocker_kind": kind,
                    "strongest_loss_db": loss,
                    "strongest_blocker_id": blocker_id,
                    "strongest_blocker_kind": kind,
                    "power_loss_db": loss,
                    "power_blocker_id": blocker_id,
                    "power_blocker_kind": kind,
                }
            )
    raw = _ue_events(synthetic, times, [3.0, 10.0, 20.0])
    merged = apply_hysteresis(raw, MIN_GAP_S, float(row["dt_s"]))
    events = _los10(merged)
    minutes = row["n_ue"] * DURATION_S / 60.0
    durations = [
        (int(event["end_snapshot"]) - int(event["start_snapshot"]) + 1) * float(row["dt_s"]) for event in events
    ]
    return {
        "events_per_ue_minute": len(events) / minutes,
        "n_events": len(events),
        "durations_s": durations,
    }


def part2_rates(rows: list[dict]) -> list[dict]:
    table = []
    for mount in MOUNTS:
        for density in DENSITIES:
            chosen = [row for row in rows if row["mount"] == mount and row["density"] == density and row["n_oru"] >= 2]
            fixed = [_selected_rate(row, "fixed") for row in chosen]
            oracle = [_selected_rate(row, "oracle") for row in chosen]
            table.append(
                {
                    "mount": mount,
                    "density": density,
                    "fixed_rate": float(np.mean([item["events_per_ue_minute"] for item in fixed])) if fixed else 0.0,
                    "oracle_rate": float(np.mean([item["events_per_ue_minute"] for item in oracle])) if oracle else 0.0,
                    "fixed_p50": _quantile([value for item in fixed for value in item["durations_s"]]).get(0.5),
                    "oracle_p50": _quantile([value for item in oracle for value in item["durations_s"]]).get(0.5),
                }
            )
    return table


def _write_cdf(path: Path, series: dict[str, list[float]], xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    for label, values_in in series.items():
        values = np.sort(np.asarray(values_in, dtype=np.float64))
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        probs = (np.arange(values.size) + 1) / values.size
        axis.step(values, probs, where="post", label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("CDF")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, linewidth=0.4, alpha=0.6)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _share(count: int, total: int) -> str:
    if total <= 0:
        return "—"
    return f"{count}/{total} ({100.0 * count / total:.1f}%)"


def _report(part1_summary: dict, events: dict, time_share: dict, rates: list[dict], seeds: list[int]) -> str:
    lines = [
        "# M1.5 diversity",
        "",
        "Tuning seeds only "
        f"({', '.join(str(seed) for seed in seeds)}), 60 s, dt = 0.1 s. "
        f"UE events use the accepted hysteresis: gap merge at {MIN_GAP_S} s, then 10 dB nested inside 3 dB. "
        "A 10 dB sample is a snapshot inside the event whose LoS loss is still at least 10 dB. "
        "Geometry was not changed after these runs.",
        "",
        "## Best available alternative on one O-RU",
        "",
        "Config: `configs/m1_scenario.yaml`. "
        "For each 10 dB LoS event, the value is the worst "
        "`10 log10(P / P_LoS)` over the event's 10 dB snapshots. "
        "`P` is the maximum post-model-B power over all non-LoS paths on that O-RU. "
        "`P_LoS` is the unblocked LoS power. "
        "This searches every reflected path, not only the strongest unblocked one.",
        "",
        "| Mount | Events | No alternative | Alternative removed | p10 / p50 / p90 [dB] |",
        "|---|---|---|---|---|",
    ]
    for mount, height in (("lamppost", 5), ("facade", 8)):
        item = part1_summary[mount]
        lines.append(
            f"| {mount} {height} m | {item['n_events']} | {item['missing']} | {item['removed']} | "
            f"{_fmt_cdf(item['samples_db'])} |"
        )
    lines.extend(
        [
            "",
            "CDF: `best_alt_cdf.png`.",
            "",
            "## Two O-RUs",
            "",
            "Config: `configs/m2_scenario.yaml`. "
            "The serving O-RU at a snapshot is the one with the largest unblocked link power. "
            "A 10 dB LoS event is kept when its O-RU is serving at the event start. "
            "The other link is unblocked when its LoS loss stays below 3 dB on every 10 dB snapshot of that event. "
            "The power ratio is the worst `10 log10(P_other / P_LoS)` over those snapshots, "
            "where `P_other` is the other O-RU's total power after model B and `P_LoS` is the serving unblocked LoS power.",
            "",
            "| Mount | Serving events | Other link unblocked | Other-link power p10 / p50 / p90 [dB] | Removed |",
            "|---|---|---|---|---|",
        ]
    )
    for mount, height in (("lamppost", 5), ("facade", 8)):
        item = events[mount]
        lines.append(
            f"| {mount} {height} m | {item['n_events']} | {_share(item['unblocked'], item['n_events'])} | "
            f"{_fmt_cdf(item['ratios_db'])} | {item['ratio_removed']} |"
        )
    lines.extend(["", "Split by the serving event's blocker:", ""])
    lines.append("| Mount | Blocker | Events | Other link unblocked | Power p10 / p50 / p90 [dB] |")
    lines.append("|---|---|---|---|---|")
    for mount, height in (("lamppost", 5), ("facade", 8)):
        for kind in ("bus/truck", "pedestrian"):
            item = events[mount]["classes"][kind]
            lines.append(
                f"| {mount} {height} m | {kind} | {item['n_events']} | "
                f"{_share(item['unblocked'], item['n_events'])} | {_fmt_cdf(item['ratios_db'])} |"
            )
    lines.extend(
        [
            "",
            "Both links blocked means both LoS losses are at least 10 dB (a missing LoS counts as blocked). "
            "The serving-outage column is the share of those serving-link 10 dB snapshots where the other link is also at least 10 dB.",
            "",
            "| Mount | Time both links ≥ 10 dB | Of serving-link 10 dB time, other also ≥ 10 dB |",
            "|---|---|---|",
        ]
    )
    for mount, height in (("lamppost", 5), ("facade", 8)):
        item = time_share[mount]
        lines.append(
            f"| {mount} {height} m | {_share(item['both'], item['total'])} | "
            f"{_share(item['serving_no_remedy'], item['serving_outage'])} |"
        )
    lines.extend(
        [
            "",
            "Serving-link LoS event rates, events per UE-minute, mean over the five seeds. "
            "Fixed serving follows the strongest unblocked link and does not react to blockage. "
            "Oracle picks the O-RU with the smaller LoS loss at that snapshot. "
            "The oracle rate is an upper bound on what switching the O-RU could gain, with no sensing delay.",
            "",
            "| Mount | Density | Fixed 10 dB rate | Oracle 10 dB rate | Fixed duration p50 [s] | Oracle duration p50 [s] |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in rates:
        height = 5 if item["mount"] == "lamppost" else 8
        fixed_p50 = "—" if item["fixed_p50"] is None else f"{item['fixed_p50']:.2f}"
        oracle_p50 = "—" if item["oracle_p50"] is None else f"{item['oracle_p50']:.2f}"
        lines.append(
            f"| {item['mount']} {height} m | {item['density']} | {item['fixed_rate']:.2f} | "
            f"{item['oracle_rate']:.2f} | {fixed_p50} | {oracle_p50} |"
        )
    lines.extend(["", "Other-O-RU power CDFs: `other_oru_cdf.png`.", ""])
    return "\n".join(lines) + "\n"


def main() -> None:
    seeds = [int(seed) for seed in load_yaml(ROOT / "configs" / "seeds.yaml")["tuning"]]
    output = ROOT / "results" / "M1.5"
    output.mkdir(parents=True, exist_ok=True)
    m1 = load_yaml(ROOT / "configs" / "m1_scenario.yaml")
    m2 = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    single = [
        _run(m1, seed, mount, density, output)
        for mount in MOUNTS
        for density in DENSITIES
        for seed in seeds
    ]
    dual = [
        _run(m2, seed, mount, density, output)
        for mount in MOUNTS
        for density in DENSITIES
        for seed in seeds
    ]
    best = part1(single)
    events = part2_events(dual)
    time_share = part2_time(dual)
    rates = part2_rates(dual)
    _write_cdf(
        output / "best_alt_cdf.png",
        {f"{mount} {5 if mount == 'lamppost' else 8} m": best[mount]["samples_db"] for mount in MOUNTS},
        "Best non-LoS power after model B, relative to unblocked LoS [dB]",
    )
    other_series = {}
    for mount in MOUNTS:
        height = 5 if mount == "lamppost" else 8
        other_series[f"{mount} {height} m"] = events[mount]["ratios_db"]
        for kind in ("bus/truck", "pedestrian"):
            other_series[f"{mount} {height} m, {kind}"] = events[mount]["classes"][kind]["ratios_db"]
    _write_cdf(
        output / "other_oru_cdf.png",
        other_series,
        "Other O-RU power after model B, relative to serving unblocked LoS [dB]",
    )
    text = _report(best, events, time_share, rates, seeds)
    (output / "report.md").write_text(text, encoding="utf-8")
    payload = {
        "seeds": seeds,
        "min_gap_s": MIN_GAP_S,
        "best_available_alt": {
            mount: {key: value for key, value in item.items() if key != "samples_db"}
            | {"p10_p50_p90_db": [_quantile(item["samples_db"]).get(p) for p in (0.1, 0.5, 0.9)]}
            for mount, item in best.items()
        },
        "other_oru": {
            mount: {
                "n_events": item["n_events"],
                "unblocked": item["unblocked"],
                "p10_p50_p90_db": [_quantile(item["ratios_db"]).get(p) for p in (0.1, 0.5, 0.9)],
                "classes": {
                    kind: {
                        "n_events": spec["n_events"],
                        "unblocked": spec["unblocked"],
                        "p10_p50_p90_db": [_quantile(spec["ratios_db"]).get(p) for p in (0.1, 0.5, 0.9)],
                    }
                    for kind, spec in item["classes"].items()
                },
            }
            for mount, item in events.items()
        },
        "time_both_blocked": time_share,
        "rates": rates,
    }
    (output / "diversity.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
