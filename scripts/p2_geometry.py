"""Paper 2 (review 5): scenario geometry and traffic of the paper-1 configuration used by paper 2.

Everything is read from configs (configs/m2_scenario.yaml, configs/m3.yaml, configs/p2.yaml), the
scenario builder (sim.scenes.traffic.prepare_scenario: O-RU positions incl. the wrapped O-RU 1),
the walkable map of the frozen estimator (sim.positioning.estimator_v2.walk_map: sidewalk bands
between curb and facade) and the bounding boxes of the scene meshes (sionna.rt.load_scene; needs
the container). Coordinates [m]: x along the street, y across (south facade negative), z up.
Traffic densities are expectations: every vehicle draws its lane uniformly from the two lanes and
its speed uniformly from its class range; actors circulate on the periodic street of length
`period_m`. Density per lane [1/km] = (count / 2 lanes) / period; flow per lane [1/min] = density x
mean class speed x 60 s. Sidewalk walkers draw their sidewalk uniformly (two sidewalks).
UE track check: min/max of the cached UE positions (results/cache/*/*/seed_*/comm_geometry.npz,
held-out set 3001-3010). Writes paper2/tables/geometry.json (input for the scenario figure).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def r(v: float, n: int = 3) -> float:
    return float(round(float(v), n))


def main() -> None:
    from sim.positioning.estimator_v2 import walk_map
    from sim.scenes.config import load_yaml
    from sim.scenes.loop import _scene_by_name
    from sim.scenes.traffic import prepare_scenario
    from sionna.rt import load_scene

    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    m3 = load_yaml(ROOT / "configs" / "m3.yaml")
    p2cfg = load_yaml(ROOT / "configs" / "p2.yaml")
    scene = load_scene(_scene_by_name(str(raw["scene"])))
    boxes = {}
    for name, obj in scene.objects.items():
        bb = obj.mi_mesh.bbox()
        boxes[name] = {"min": [r(v) for v in bb.min], "max": [r(v) for v in bb.max]}
    blds = {k: v for k, v in boxes.items() if k != "floor"}
    south = [v for v in blds.values() if v["max"][1] < 0]
    north = [v for v in blds.values() if v["min"][1] > 0]
    y_s = max(v["max"][1] for v in south)
    y_n_near = min(v["min"][1] for v in north)
    y_n_far = max(v["min"][1] for v in north)
    x_lo = min(v["min"][0] for v in south)
    x_hi = max(v["max"][0] for v in south)
    period = float(raw["lanes"][0]["length_m"])
    lanes = [{"name": ln["name"], "y_m": float(ln["origin_m"][1]), "direction_x": float(ln["direction"][0]), "x_range_m": sorted([float(ln["origin_m"][0]), float(ln["origin_m"][0]) + float(ln["direction"][0]) * float(ln["length_m"])])}
             for ln in raw["lanes"]]
    orus = {}
    wm = None
    for mount in ("lamppost", "facade"):
        sc = prepare_scenario(raw, seed=3001, mount=mount, density="low", duration_s=60.0, dt_s=0.1)
        orus[mount] = [{"name": o["name"], "mount": o["mount"], "position_m": [r(v) for v in o["position_m"]]} for o in sc["orus"]]
        wm = walk_map(raw, sc, p2cfg)
    sidewalks = []
    bands = {"south": wm["sidewalks_y"][0], "north": wm["sidewalks_y"][1]}
    for sw in raw["sidewalks"]:
        sidewalks.append({"name": sw["name"], "centre_y_m": float(sw["origin_m"][1]), "band_y_m": [r(v) for v in bands[sw["name"]]],
                          "x_range_m": [float(sw["origin_m"][0]), float(sw["origin_m"][0]) + float(sw["length_m"])]})
    ues = []
    for ue in raw["ues"]:
        sw = next(s for s in raw["sidewalks"] if s["name"] == ue["sidewalk"])
        ues.append({"name": ue["name"], "sidewalk": ue["sidewalk"], "y_m": float(sw["origin_m"][1]), "height_m": float(ue["height_m"]),
                    "speed_mps": float(ue["speed_mps"]), "s0_m": float(ue["s0_m"]), "x_range_m": [float(sw["origin_m"][0]), float(sw["origin_m"][0]) + float(sw["length_m"])]})
    # check against the cached UE positions of the held-out jobs
    xs, ys, zs = [], [], []
    for s in p2cfg["heldout2"]:
        for mount in ("lamppost", "facade"):
            for dens in ("low", "high"):
                g = np.load(ROOT / "results" / "cache" / mount / dens / f"seed_{s}" / "comm_geometry.npz")
                ue = g["ue_position_m"]
                xs.append(ue[..., 0].ravel()); ys.append(ue[..., 1].ravel()); zs.append(ue[..., 2].ravel())
    xs, ys, zs = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    kinds = raw["blocker_kinds"]
    traffic = {}
    for dens, cnt in raw["traffic"].items():
        veh = {}
        for kind, key in (("car", "n_cars"), ("bus", "n_buses"), ("truck", "n_trucks")):
            n = int(cnt[key])
            v_mean = float(np.mean(kinds[kind]["speed_mps"]))
            dens_km = n / len(lanes) / (period / 1000.0)
            veh[kind] = {"count_per_period": n, "per_lane_per_km": r(dens_km, 2), "per_lane_per_min": r(dens_km / 1000.0 * v_mean * 60.0, 2),
                         "speed_mps": [float(v) for v in kinds[kind]["speed_mps"]]}
        tot = sum(v["count_per_period"] for v in veh.values())
        veh["all"] = {"count_per_period": tot, "per_lane_per_km": r(tot / len(lanes) / (period / 1000.0), 2),
                      "per_lane_per_min": r(sum(v["per_lane_per_min"] for v in veh.values()), 2)}
        n_sw = int(cnt["n_sidewalk_pedestrians"])
        traffic[dens] = {"vehicles": veh,
                         "pedestrians": {"sidewalk_walkers": n_sw, "per_sidewalk_per_km": r(n_sw / len(sidewalks) / (period / 1000.0), 2),
                                         "crossing_walkers": int(cnt["n_crossing_pedestrians"]), "speed_mps": [float(v) for v in kinds["pedestrian"]["speed_mps"]]}}
    veh_speeds = [v for k in ("car", "bus", "truck") for v in kinds[k]["speed_mps"]]
    out = {
        "definition": __doc__,
        "scene": str(raw["scene"]),
        "street": {"building_row_x_m": [x_lo, x_hi], "length_m": r(x_hi - x_lo, 1), "period_m": period,
                   "facade_y_m": {"south": y_s, "north": y_n_near, "north_setback": y_n_far},
                   "width_m": r(y_n_near - y_s, 3), "width_setback_m": r(y_n_far - y_s, 3), "curb_abs_y_m": r(wm["sidewalks_y"][1][0]),
                   "lane_width_m": float(p2cfg.get("lane_width_m", 3.5)), "mesh_bboxes": boxes},
        "lanes": lanes,
        "sidewalks": sidewalks,
        "crossing_half_width_m": float(wm["crossing_half_width"]),
        "orus": orus,
        "oru_offset_x_m": {m: r(abs(o[0]["position_m"][0] - o[1]["position_m"][0])) for m, o in orus.items()},
        "oru_array": {"rows": int(raw["array"]["oru"]["num_rows"]), "cols": int(raw["array"]["oru"]["num_cols"]), "spacing_wl": float(raw["array"]["oru"]["horizontal_spacing"]),
                      "plane": "y-z", "boresight": "+x"},
        "ues": ues,
        "ue_cache_check": {"x_min": r(xs.min()), "x_max": r(xs.max()), "y": sorted(set(np.round(ys, 3).tolist())), "z": sorted(set(np.round(zs, 3).tolist()))},
        "run": {"duration_s": 60.0, "epoch_s": float(raw["dt_s"]), "comm_step_s": float(m3["dt_comm_s"])},
        "traffic": traffic,
        "vehicle_speed_mps": [min(veh_speeds), max(veh_speeds)],
        "blocker_sizes_m": {k: {"length": float(v["length_m"]), "width": float(v["width_m"]), "height": float(v["height_m"])} for k, v in kinds.items()},
    }
    (ROOT / "paper2" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper2" / "tables" / "geometry.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("definition",)}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
