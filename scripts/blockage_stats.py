"""UE-level and per-path blockage statistics.

Writes results/M1/blockage_stats.md. Geometry is the deployment in
configs/m1_scenario.yaml. Runs are not dropped when blockage is rare.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.comm.events import annotate_los_events, apply_hysteresis, blocker_oru_distances, merge_ue_events  # noqa: E402
from sim.scenes.config import load_yaml  # noqa: E402
from sim.scenes.loop import run_scenario  # noqa: E402
from sim.scenes.traffic import prepare_scenario  # noqa: E402

# Smallest gap that is tried. The study picks the first feasible value.
GAP_CANDIDATES_S = (0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)

DURATION_S = 60.0
MOUNTS = ("lamppost", "facade")
DENSITIES = ("low", "high")
METRICS = ("los", "strongest", "power")
METRIC_LABEL = {
    "los": "LoS loss",
    "strongest": "strongest-path loss",
    "power": "total power loss",
}
KIND_GROUP = {
    "car": "car",
    "bus": "bus/truck",
    "truck": "bus/truck",
    "pedestrian": "pedestrian",
}
PATH_CLASSES = ("los", "ground", "wall", "double")


def _duration_s(event: dict, dt_s: float) -> float:
    return float(event["end_s"]) - float(event["start_s"]) + dt_s


def _censored(event: dict, dt_s: float, n_snapshots: int) -> bool:
    last_s = (n_snapshots - 1) * dt_s
    return abs(float(event["start_s"])) < 1e-9 or abs(float(event["end_s"]) - last_s) < 1e-9


def _quantile_table(values: list[float]) -> dict[float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return {}
    array = np.sort(np.asarray(finite, dtype=np.float64))
    return {percent: float(np.quantile(array, percent)) for percent in (0.1, 0.5, 0.9)}


def _kind(event: dict) -> str:
    kind = event.get("blocker_kind")
    if kind not in KIND_GROUP:
        return "unknown"
    return KIND_GROUP[kind]


def _empty_kinds() -> dict[str, float]:
    return {"car": 0.0, "bus/truck": 0.0, "pedestrian": 0.0, "unknown": 0.0}


def _summarize_ue(events: list[dict], dt_s: float, n_snapshots: int, n_ue: int, duration_s: float) -> dict:
    ue_minutes = n_ue * duration_s / 60.0
    by_metric = {}
    for metric in METRICS:
        by_threshold = {}
        for threshold in (3.0, 10.0, 20.0):
            chosen = [
                event
                for event in events
                if event["metric"] == metric and abs(float(event["threshold_db"]) - threshold) < 1e-9
            ]
            durations = [_duration_s(event, dt_s) for event in chosen]
            losses = [
                float(event["max_loss_db"])
                for event in chosen
                if event["max_loss_db"] is not None and np.isfinite(event["max_loss_db"])
            ]
            kinds = _empty_kinds()
            for event in chosen:
                kinds[_kind(event)] += 1.0
            by_threshold[f"{threshold:.0f}"] = {
                "n_events": len(chosen),
                "events_per_ue_minute": len(chosen) / ue_minutes,
                "censored": sum(1 for event in chosen if _censored(event, dt_s, n_snapshots)),
                "durations_s": durations,
                "losses_db": losses,
                "kinds": kinds,
                "n_infinite_loss": sum(
                    1
                    for event in chosen
                    if event["max_loss_db"] is None or not np.isfinite(event["max_loss_db"])
                ),
            }
        by_metric[metric] = by_threshold
    return by_metric


def _summarize_paths(events: list[dict]) -> dict:
    counts = {name: 0 for name in PATH_CLASSES}
    weights = {name: 0.0 for name in PATH_CLASSES}
    other = 0
    for event in events:
        path_class = str(event.get("path_class", "other"))
        weight = float(event.get("power_weight", 0.0))
        if path_class not in counts:
            other += 1
            continue
        counts[path_class] += 1
        weights[path_class] += weight
    return {"counts": counts, "weights": weights, "other": other, "n_events": len(events)}


def _add_kinds(total: dict[str, float], extra: dict[str, float]) -> None:
    for key, value in extra.items():
        total[key] = total.get(key, 0.0) + float(value)


def _run_case(raw: dict, seed: int, mount: str, density: str, dt_s: float) -> dict:
    scenario = prepare_scenario(
        raw,
        seed=int(seed),
        mount=mount,
        density=density,
        duration_s=DURATION_S,
        dt_s=dt_s,
    )
    # Raw events, so the gap can be chosen after every tuning seed has run.
    scenario["blockage"]["min_gap_s"] = 0.0
    print(
        f"run mount={mount} height={scenario['oru_height_m']} density={density} "
        f"seed={seed} dt={dt_s} snapshots={scenario['n_snapshots']}",
        flush=True,
    )
    result = run_scenario(
        scenario,
        ROOT / "results" / "M1" / "_stats_tmp",
        render=False,
        comm_only=True,
        write_outputs=False,
        keep_snapshot_rows=False,
    )
    n_ue = len(scenario["ues"])
    n_snapshots = int(result["n_snapshots"])
    used_dt = float(result["dt_s"])
    return {
        "seed": int(seed),
        "height_m": float(scenario["oru_height_m"]),
        "dt_s": used_dt,
        "n_snapshots": n_snapshots,
        "n_ue": n_ue,
        "raw_events": result["ue_events_raw"],
        "trace": result["ue_trace"],
        "scenario": scenario,
        "paths": _summarize_paths(result["events"]),
        "los_kinds": _los_kinds(result["events"]),
        "stability": result["path_id_stability"],
    }


def _count_events(events: list[dict], metric: str, threshold: float) -> int:
    return sum(
        1
        for event in events
        if event["metric"] == metric and abs(float(event["threshold_db"]) - threshold) < 1e-9
    )


def _violating_pairs(rows: list[dict], min_gap_s: float, *, nest: bool) -> int:
    """(run, metric) pairs where the 10 dB count still exceeds the 3 dB count."""
    violations = 0
    for row in rows:
        if nest:
            merged = apply_hysteresis(row["raw_events"], min_gap_s, float(row["dt_s"]))
        else:
            merged = merge_ue_events(row["raw_events"], min_gap_s, float(row["dt_s"]))
        for metric in METRICS:
            if _count_events(merged, metric, 10.0) > _count_events(merged, metric, 3.0):
                violations += 1
    return violations


def _choose_gap(rows: list[dict]) -> tuple[float, list[tuple[float, int]], list[tuple[float, int]]]:
    """Pick the gap from the gap-only sweep, then check the nested rule.

    Gap-only merge never cleared every inversion: a dip can stay between 3 dB
    and 10 dB for longer than the gap that would also glue separate outages
    together. The chosen gap is the candidate with the fewest gap-only
    inversions. Nesting higher thresholds inside the 3 dB event is what makes
    the 10 dB count stay at or below the 3 dB count.
    """
    gap_only = [(gap, _violating_pairs(rows, gap, nest=False)) for gap in GAP_CANDIDATES_S]
    min_gap_s = min(gap_only, key=lambda item: (item[1], item[0]))[0]
    nested = [(gap, _violating_pairs(rows, gap, nest=True)) for gap in GAP_CANDIDATES_S]
    return min_gap_s, gap_only, nested


def _finalize_row(row: dict, min_gap_s: float) -> None:
    """Merge, annotate headroom and warning, and store the UE summary."""
    dt_s = float(row["dt_s"])
    merged = apply_hysteresis(row["raw_events"], min_gap_s, dt_s)
    blocker_ids = sorted({str(event["blocker_id"]) for event in merged if event.get("blocker_id")})
    distances = (
        blocker_oru_distances(row["scenario"], blocker_ids, int(row["n_snapshots"]), dt_s) if blocker_ids else {}
    )
    range_m = float(row["scenario"]["blockage"].get("sensing_range_m", 40.0))
    annotate_los_events(merged, row["trace"], distances, dt_s, range_m)
    row["ue_events"] = merged
    row["ue"] = _summarize_ue(merged, dt_s, int(row["n_snapshots"]), int(row["n_ue"]), DURATION_S)
    row["headroom_db"] = [
        float(event["headroom_db"])
        for event in merged
        if event["metric"] == "los"
        and abs(float(event["threshold_db"]) - 10.0) < 1e-9
        and event.get("headroom_db") is not None
    ]
    row["headroom_available_db"] = [
        float(event["headroom_available_db"])
        for event in merged
        if event["metric"] == "los"
        and abs(float(event["threshold_db"]) - 10.0) < 1e-9
        and event.get("headroom_available_db") is not None
    ]
    row["headroom_missing"] = sum(
        1
        for event in merged
        if event["metric"] == "los"
        and abs(float(event["threshold_db"]) - 10.0) < 1e-9
        and event.get("headroom_db") is None
    )
    row["alt_same_blocker"] = sum(
        1
        for event in merged
        if event["metric"] == "los"
        and abs(float(event["threshold_db"]) - 10.0) < 1e-9
        and event.get("alt_same_blocker")
    )
    row["n_los10"] = sum(
        1
        for event in merged
        if event["metric"] == "los" and abs(float(event["threshold_db"]) - 10.0) < 1e-9
    )
    row["warning"] = {name: [] for name in ("bus/truck", "pedestrian", "car")}
    row["warning_censored"] = {name: 0 for name in row["warning"]}
    row["warning_outside"] = {name: 0 for name in row["warning"]}
    for event in merged:
        if event["metric"] != "los" or abs(float(event["threshold_db"]) - 10.0) > 1e-9:
            continue
        kind = _kind(event)
        if kind not in row["warning"]:
            continue
        if not event.get("in_range"):
            row["warning_outside"][kind] += 1
        elif event.get("censored"):
            row["warning_censored"][kind] += 1
        elif event.get("warning_s") is not None:
            row["warning"][kind].append(float(event["warning_s"]))


def _strip_raw(row: dict) -> None:
    row.pop("raw_events", None)
    row.pop("trace", None)
    row.pop("scenario", None)
    row.pop("ue_events", None)


def _los_kinds(events: list[dict]) -> dict[str, float]:
    kinds = _empty_kinds()
    for event in events:
        if event.get("path_class") != "los":
            continue
        kinds[_kind(event)] += 1.0
    return kinds


def _mean_rate(seed_rows: list[dict], metric: str, threshold: str) -> float:
    rates = [row["ue"][metric][threshold]["events_per_ue_minute"] for row in seed_rows]
    return float(np.mean(rates))


def _pool_ue(seed_rows: list[dict], metric: str, threshold: str) -> dict:
    durations: list[float] = []
    losses: list[float] = []
    kinds = _empty_kinds()
    n_events = 0
    n_censored = 0
    n_infinite = 0
    rates = []
    for row in seed_rows:
        bucket = row["ue"][metric][threshold]
        durations.extend(bucket["durations_s"])
        losses.extend(bucket["losses_db"])
        _add_kinds(kinds, bucket["kinds"])
        n_events += bucket["n_events"]
        n_censored += bucket["censored"]
        n_infinite += bucket["n_infinite_loss"]
        rates.append(bucket["events_per_ue_minute"])
    return {
        "rates": rates,
        "mean_rate": float(np.mean(rates)) if rates else 0.0,
        "durations_s": durations,
        "losses_db": losses,
        "kinds": kinds,
        "n_events": n_events,
        "censored": n_censored,
        "n_infinite": n_infinite,
    }


def _write_cdf_plot(path: Path, by_case: dict[str, dict[str, list[float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for label, samples in by_case.items():
        for axis, metric in zip(axes, METRICS):
            values = np.sort(np.asarray(samples[metric], dtype=np.float64))
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            probs = (np.arange(values.size) + 1) / values.size
            axis.step(values, probs, where="post", label=label)
    for axis, metric in zip(axes, METRICS):
        axis.set_xlabel(f"{METRIC_LABEL[metric]} duration [s]")
        axis.set_ylabel("CDF")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, linewidth=0.4, alpha=0.6)
        axis.set_title("10 dB")
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_two_cdf(
    path: Path,
    left: dict[str, list[float]],
    left_xlabel: str,
    right: dict[str, list[float]],
    right_xlabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for axis, samples, xlabel in (
        (axes[0], left, left_xlabel),
        (axes[1], right, right_xlabel),
    ):
        for label, values_in in samples.items():
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


def _fmt_cdf(values: list[float], digits: int) -> str:
    table = _quantile_table(values)
    if not table:
        return "—"
    return " / ".join(f"{table[percent]:.{digits}f}" for percent in (0.1, 0.5, 0.9))


def _fmt_kinds(kinds: dict[str, float], total: float) -> str:
    if total <= 0.0:
        return "no events"
    parts = []
    for name in ("car", "bus/truck", "pedestrian"):
        count = kinds.get(name, 0.0)
        parts.append(f"{name} {100.0 * count / total:.1f}%")
    return ", ".join(parts)


def _section_case(lines: list[str], case: dict) -> None:
    label = f"{case['mount']} {case['height_m']:.0f} m, {case['density']}"
    lines.append(f"## {label}")
    lines.append("")
    lines.append(
        "UE events per UE-minute, mean over seeds. "
        "Duration CDF is p10 / p50 / p90 in seconds. "
        "Blocker shares use the blocker that dominates the peak snapshot of each event."
    )
    lines.append("")
    lines.append("| Metric | 3 dB rate | 10 dB rate | 20 dB rate | 10 dB duration [s] | 10 dB blocker share |")
    lines.append("|---|---|---|---|---|---|")
    for metric in METRICS:
        pooled = {threshold: _pool_ue(case["seeds"], metric, threshold) for threshold in ("3", "10", "20")}
        lines.append(
            "| {label} | {r3:.3f} | {r10:.3f} | {r20:.3f} | {cdf} | {kinds} |".format(
                label=METRIC_LABEL[metric],
                r3=pooled["3"]["mean_rate"],
                r10=pooled["10"]["mean_rate"],
                r20=pooled["20"]["mean_rate"],
                cdf=_fmt_cdf(pooled["10"]["durations_s"], 3),
                kinds=_fmt_kinds(pooled["10"]["kinds"], pooled["10"]["n_events"]),
            )
        )
    lines.append("")
    for metric in METRICS:
        pooled = _pool_ue(case["seeds"], metric, "10")
        per_seed = ", ".join(
            f"{row['seed']} → {row['ue'][metric]['10']['events_per_ue_minute']:.3f}" for row in case["seeds"]
        )
        lines.append(
            f"{METRIC_LABEL[metric]} at 10 dB: total events {pooled['n_events']}, "
            f"censored {pooled['censored']}, non-finite peaks {pooled['n_infinite']}. "
            f"Per seed: {per_seed}."
        )
        lines.append("")
        lines.append(
            f"Duration CDF (3 / 10 / 20 dB): "
            + ", ".join(
                f"{threshold} dB {_fmt_cdf(_pool_ue(case['seeds'], metric, threshold)['durations_s'], 3)}"
                for threshold in ("3", "10", "20")
            )
            + "."
        )
        lines.append("")


def _pool_paths(seed_rows: list[dict]) -> tuple[dict[str, int], dict[str, float], int]:
    counts = {name: 0 for name in PATH_CLASSES}
    weights = {name: 0.0 for name in PATH_CLASSES}
    n_events = 0
    for row in seed_rows:
        n_events += int(row["paths"]["n_events"])
        for name in PATH_CLASSES:
            counts[name] += int(row["paths"]["counts"][name])
            weights[name] += float(row["paths"]["weights"][name])
    return counts, weights, n_events


def _pool_los(seed_rows: list[dict]) -> dict[str, float]:
    kinds = _empty_kinds()
    for row in seed_rows:
        _add_kinds(kinds, row["los_kinds"])
    return kinds


def main() -> None:
    raw = load_yaml(ROOT / "configs" / "m1_scenario.yaml")
    seeds = [int(seed) for seed in load_yaml(ROOT / "configs" / "seeds.yaml")["tuning"]]
    output = ROOT / "results" / "M1"
    output.mkdir(parents=True, exist_ok=True)
    dt_s = float(raw["dt_s"])
    cases = []
    for mount in MOUNTS:
        for density in DENSITIES:
            seed_rows = [
                _run_case(raw, int(seed), mount, density, dt_s) for seed in seeds
            ]
            cases.append(
                {
                    "mount": mount,
                    "height_m": float(seed_rows[0]["height_m"]),
                    "density": density,
                    "seeds": seed_rows,
                }
            )

    fine_seed = seeds[0]
    fine = _run_case(raw, fine_seed, "facade", "high", 0.02)
    tuning_rows = [row for case in cases for row in case["seeds"]] + [fine]
    min_gap_s, gap_table, nested_table = _choose_gap(tuning_rows)
    for row in tuning_rows:
        _finalize_row(row, min_gap_s)
    coarse = next(
        row
        for case in cases
        if case["mount"] == "facade" and case["density"] == "high"
        for row in case["seeds"]
        if row["seed"] == fine_seed
    )

    lines = [
        "# Blockage statistics",
        "",
        "Deployment from `configs/m1_scenario.yaml`. O-RU height is the mount height: "
        "lamppost 5 m, facade 8 m. Vehicles and pedestrians are drawn from each tuning seed. "
        "Pedestrians walk independently on both sidewalks, in both directions, and some cross "
        "the street at a random station. Positions were not moved after looking at blockage.",
        "",
        f"Each case is {DURATION_S:.0f} s at dt = {dt_s} s. Tuning seeds: {', '.join(str(seed) for seed in seeds)}.",
        "",
        "Comm paths are solved with no sensing targets. Model B then scales every segment. "
        "Three UE metrics use that unblocked counterfactual:",
        "",
        "- LoS loss [dB]: model-B loss of the LoS path, summed over blockers.",
        "- Strongest-path loss [dB]: model-B loss of the path with the largest unblocked power.",
        "- Total power loss [dB]: `10 log10(P_unblocked / P_blocked)`, with powers summed over paths.",
        "",
        "A UE event is a contiguous run of snapshots where that metric is at least the threshold. "
        "The primary threshold is 10 dB. 3 dB and 20 dB are reported beside it. "
        "Duration is (end − start) + dt. An event that touches the first or last snapshot is censored. "
        "The blocker share is the type with the largest contribution at the peak snapshot "
        "(largest screen loss for LoS and the strongest path; largest single-blocker power removal "
        "for the total).",
        "",
        "Per-path events are still one blocker on one path while that blocker's own loss is at least "
        f"{raw['blockage']['event_loss_db']} dB. They are broken down by path class. "
        "The power weight of a per-path event is that path's share of the UE's unblocked power, "
        "averaged over the event.",
        "",
        "A path id is the interaction-type sequence plus the `Paths.objects` integer sequence. "
        "Delay is not included. Those integers are assigned per scene load, so they are not compared "
        "across processes. Inside one run, persistence is the fraction of path ids present at a "
        "snapshot that are still present at the next snapshot.",
        "",
        "`PathSolver(deterministic=True)`.",
        "",
        "UE events of one metric separated by an inactive gap shorter than "
        f"`min_gap_s` = {min_gap_s:.1f} s are merged. "
        "That gap is the candidate with the fewest gap-only inversions on the tuning runs "
        "(the five seeds at dt = 0.1 s and the dt = 0.02 s run). "
        "No candidate through 5 s removed every inversion, and longer gaps created new ones "
        "by joining separate 3 dB outages while a 10 dB dip was still open. "
        "After the gap merge, 10 dB fragments inside one 3 dB event are collapsed into one event, "
        "and 20 dB fragments are collapsed inside that 10 dB event. "
        "That nesting is what keeps the 10 dB count from exceeding the 3 dB count.",
        "",
        "Gap-only inversions, (run, metric) pairs with more 10 dB events than 3 dB events:",
        "",
    ]
    for gap, violations in gap_table:
        mark = " ← chosen" if abs(gap - min_gap_s) < 1e-12 else ""
        lines.append(f"- {gap:.1f} s: {violations}{mark}")
    lines.append("")
    lines.append("After nesting, the same count of inversions:")
    lines.append("")
    for gap, violations in nested_table:
        mark = " ← chosen" if abs(gap - min_gap_s) < 1e-12 else ""
        lines.append(f"- {gap:.1f} s: {violations}{mark}")
    lines.append("")
    plot_cases: dict[str, dict[str, list[float]]] = {}
    for case in cases:
        _section_case(lines, case)
        label = f"{case['mount']} {case['height_m']:.0f} m, {case['density']}"
        plot_cases[label] = {
            metric: _pool_ue(case["seeds"], metric, "10")["durations_s"] for metric in METRICS
        }
        counts, weights, n_events = _pool_paths(case["seeds"])
        weight_total = sum(weights.values())
        lines.append("Per-path events by class (count, then power-weighted share):")
        lines.append("")
        for name in PATH_CLASSES:
            count_share = 0.0 if n_events == 0 else 100.0 * counts[name] / n_events
            weight_share = 0.0 if weight_total <= 0.0 else 100.0 * weights[name] / weight_total
            lines.append(
                f"- {name}: {counts[name]} ({count_share:.1f}% of events), "
                f"power weight {weight_share:.1f}%"
            )
        lines.append("")
        stability = [float(row["stability"]["mean_persistence"] or 0.0) for row in case["seeds"]]
        distinct = [int(row["stability"]["n_distinct_ids"]) for row in case["seeds"]]
        lines.append(
            "Path-id persistence, mean over seeds: "
            f"{float(np.mean(stability)):.3f} "
            f"(per seed {', '.join(f'{value:.3f}' for value in stability)}). "
            f"Distinct ids per seed: {', '.join(str(value) for value in distinct)}."
        )
        lines.append("")

    lines.append("## Which blocker types block LoS")
    lines.append("")
    lines.append(
        "Counted from per-path events whose class is LoS (one blocker, loss at least "
        f"{raw['blockage']['event_loss_db']} dB), pooled over density and the tuning seeds."
    )
    lines.append("")
    for height, mount in ((5.0, "lamppost"), (8.0, "facade")):
        rows = [
            row
            for case in cases
            if case["mount"] == mount
            for row in case["seeds"]
        ]
        kinds = _pool_los(rows)
        total = sum(kinds.values())
        lines.append(
            f"- {mount} {height:.0f} m: {total:.0f} LoS path-events. {_fmt_kinds(kinds, total)}."
        )
    lines.append("")
    lines.append(
        "The same question on the UE LoS metric at 10 dB, pooled the same way:"
    )
    lines.append("")
    for height, mount in ((5.0, "lamppost"), (8.0, "facade")):
        rows = [row for case in cases if case["mount"] == mount for row in case["seeds"]]
        pooled = _pool_ue(rows, "los", "10")
        # _pool_ue on mixed densities is fine: mean of per-seed rates, kinds summed.
        lines.append(
            f"- {mount} {height:.0f} m: {pooled['n_events']} UE LoS events at 10 dB. "
            f"{_fmt_kinds(pooled['kinds'], pooled['n_events'])}."
        )
    lines.append("")

    lines.append("## Headroom during 10 dB LoS events")
    lines.append("")
    lines.append(
        "Unblocked headroom is the worst `10 log10(P_alt / P_LoS)` during the event. "
        "Both powers are the unblocked counterfactual on that O-RU. "
        "Available headroom uses the same alternative after model B on that snapshot, "
        "over the unblocked LoS power: the power actually left on the backup. "
        "`P_alt` is the strongest path that is not LoS. "
        "A missing value means that snapshot had no alternative path. "
        "A non-finite available value means model B removed that alternative. "
        "The same-blocker share is the fraction of 10 dB LoS events in which the event's "
        "blocker also puts at least the per-path loss threshold on that alternative. "
        "CDFs are over events, pooled across density and the tuning seeds."
    )
    lines.append("")
    lines.append(
        "| Mount | Unblocked events | None | Unblocked p10 / p50 / p90 [dB] | "
        "Available (finite) | Available removed | Available p10 / p50 / p90 [dB] | Same blocker |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    headroom_plot: dict[str, list[float]] = {}
    for height, mount in ((5.0, "lamppost"), (8.0, "facade")):
        samples: list[float] = []
        available: list[float] = []
        missing = 0
        same = 0
        n_los10 = 0
        for case in cases:
            if case["mount"] != mount:
                continue
            for row in case["seeds"]:
                samples.extend(row["headroom_db"])
                available.extend(row["headroom_available_db"])
                missing += int(row["headroom_missing"])
                same += int(row["alt_same_blocker"])
                n_los10 += int(row["n_los10"])
        finite_available = [value for value in available if np.isfinite(value)]
        removed = len(available) - len(finite_available)
        headroom_plot[f"{mount} {height:.0f} m unblocked"] = samples
        headroom_plot[f"{mount} {height:.0f} m available"] = finite_available
        share = 0.0 if n_los10 == 0 else 100.0 * same / n_los10
        lines.append(
            f"| {mount} {height:.0f} m | {len(samples)} | {missing} | {_fmt_cdf(samples, 2)} | "
            f"{len(finite_available)} | {removed} | {_fmt_cdf(finite_available, 2)} | "
            f"{same}/{n_los10} ({share:.1f}%) |"
        )
    lines.append("")

    lines.append("## Warning time")
    lines.append("")
    lines.append(
        "For each 10 dB LoS event, the reported warning is an upper bound on how long "
        "the dominant blocker has already been inside "
        f"{raw['blockage']['sensing_range_m']:.0f} m of the O-RU when the event starts. "
        "It is the time since that identity entered the range, not a predictor's lead time. "
        "Distance is the 3D range. A wrap starts a new identity, so time from the previous lap "
        "is not included. The value is censored when that identity is already inside the range "
        "on its first sample (the scenario starts, or the wrap places the new id inside the range). "
        "An event whose blocker is outside the range at the start has no warning. "
        "The CDF uses the uncensored upper bounds, pooled over mounts, densities, and tuning seeds."
    )
    lines.append("")
    lines.append("| Blocker | Uncensored events | Censored | Outside range | Upper bound p10 / p50 / p90 [s] |")
    lines.append("|---|---|---|---|---|")
    warning_plot: dict[str, list[float]] = {}
    for kind in ("bus/truck", "pedestrian", "car"):
        samples = []
        censored = 0
        outside = 0
        for case in cases:
            for row in case["seeds"]:
                samples.extend(row["warning"][kind])
                censored += int(row["warning_censored"][kind])
                outside += int(row["warning_outside"][kind])
        warning_plot[kind] = samples
        lines.append(
            f"| {kind} | {len(samples)} | {censored} | {outside} | {_fmt_cdf(samples, 3)} |"
        )
    lines.append("")

    lines.append("## Time step")
    lines.append("")
    lines.append(
        f"Facade, high density, seed {fine_seed}, {DURATION_S:.0f} s. "
        f"dt = {coarse['dt_s']} s ({coarse['n_snapshots']} snapshots) against "
        f"dt = {fine['dt_s']} s ({fine['n_snapshots']} snapshots). "
        "Same trajectories; only the sample interval changes."
    )
    lines.append("")
    lines.append("| Metric | 10 dB events/UE-minute at 0.1 s | at 0.02 s | duration p50 at 0.1 s | at 0.02 s |")
    lines.append("|---|---|---|---|---|")
    for metric in METRICS:
        left = coarse["ue"][metric]["10"]
        right = fine["ue"][metric]["10"]
        left_p50 = _quantile_table(left["durations_s"]).get(0.5)
        right_p50 = _quantile_table(right["durations_s"]).get(0.5)
        lines.append(
            "| {label} | {a:.3f} | {b:.3f} | {c} | {d} |".format(
                label=METRIC_LABEL[metric],
                a=left["events_per_ue_minute"],
                b=right["events_per_ue_minute"],
                c="—" if left_p50 is None else f"{left_p50:.3f}",
                d="—" if right_p50 is None else f"{right_p50:.3f}",
            )
        )
    lines.append("")
    lines.append(
        "Path-id persistence in this pair: "
        f"{coarse['stability']['mean_persistence']} at dt = {coarse['dt_s']} s, "
        f"{fine['stability']['mean_persistence']} at dt = {fine['dt_s']} s."
    )
    lines.append("")

    plot_path = output / "blockage_cdf.png"
    _write_cdf_plot(plot_path, plot_cases)
    extra_plot = output / "headroom_warning_cdf.png"
    _write_two_cdf(
        extra_plot,
        headroom_plot,
        "Headroom [dB]",
        warning_plot,
        "Upper bound on time in range [s]",
    )
    lines.append(
        f"Empirical duration CDFs of the 10 dB UE events, pooled over the tuning seeds: `{plot_path.name}`."
    )
    lines.append(f"Headroom and warning CDFs: `{extra_plot.name}`.")
    lines.append("")
    lines.append(
        "Empty cells are cases with no event. A rare result is reported as measured; "
        "the geometry was not changed to produce more events."
    )
    lines.append("")
    text = "\n".join(lines)
    (output / "blockage_stats.md").write_text(text, encoding="utf-8")
    for row in tuning_rows:
        _strip_raw(row)
    payload = {
        "min_gap_s": min_gap_s,
        "gap_table": [{"gap_s": gap, "inversions": count} for gap, count in gap_table],
        "cases": cases,
        "dt_comparison": {"coarse": coarse, "fine": fine},
    }
    (output / "blockage_stats.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
