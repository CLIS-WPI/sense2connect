"""Paper-1 diagnostic for paper 2 (development seeds; paper-1 code imported read-only, nothing changed).

The paper-1 radar at oru-0 uses the same 8 x 8 planar array in the y-z plane
(boresight +x): an echo from (x, y) and its mirror image (2 x_r - x, y) across
the array plane x = x_r give the same array response. For the final paper-1
map tracker (detector budget 4, image-method ghost handling, development seeds
1001-1010, 40 runs) this counts, per CPI, the unmatched confirmed tracks (false
tracks) and the unmatched clusters (detector + DBSCAN output after ghost
rejection) that lie BEHIND the panel (x < x_r) at the mirror position of a true
target: the mirror image (2 x_r - x, y) lies inside a true target's horizontal
box expanded by 1 m (the paper-1 fragment definition). Matching to the ground
truth with the paper-1 class gates. Writes results/P2/diag_radar_mirror_dev.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _job(item):
    import torch

    torch.set_num_threads(1)
    from review2_calibrate import replay_rows
    from sim.scenes.config import load_yaml
    from sim.sensing.cluster import cluster_detections
    from sim.sensing.ghost import reject_ghosts
    from sim.sensing.metrics import CLASS_GATES_M, _in_box, _match_classes, _xy
    from xapp.tracks import load_detections

    job, = item
    seed, mount, density = job
    raw = load_yaml(ROOT / "configs" / "m2_scenario.yaml")
    cfg = load_yaml(ROOT / "configs" / "m3.yaml")
    sizes = raw["blocker_kinds"]
    spec = raw["sensing_radar"]
    dd = cfg["sensing"]["budgets"][4]
    det = load_detections(ROOT / "results" / "cache" / mount / density / f"seed_{seed}")
    xr = float(det["radar_position_m"][0])
    tracks = replay_rows(job)
    gates = {c: float(v) for c, v in CLASS_GATES_M.items()}
    key = f"blind:{int(dd['train'])}:{float(dd['pfa'])}"
    walls = [float(v) for v in spec["wall_y_m"]]
    cnt = {k: 0 for k in ("tracks_false", "tracks_behind", "tracks_mirror", "clusters_false", "clusters_behind", "clusters_mirror", "frames")}
    for frame, rows in zip(det["frames"], tracks):
        gt = frame["ground_truth"]
        dets = reject_ghosts(list(frame["detections"][key]), walls, float(dd["ghost_association_m"]))
        clusters = cluster_detections(dets, float(dd["eps_m"]), int(spec["cluster"]["min_samples"]))
        cnt["frames"] += 1
        for name, items in (("tracks", rows), ("clusters", clusters)):
            pairs = _match_classes(gt, _xy(gt), _xy(items), gates)
            used = {j for _, j in pairs}
            for j, it in enumerate(items):
                if j in used:
                    continue
                x, y = float(it["x_m"]), float(it["y_m"])
                cnt[f"{name}_false"] += 1
                if x < xr:
                    cnt[f"{name}_behind"] += 1
                    mir = np.array([2 * xr - x, y])
                    if any(_in_box(mir, g, sizes) for g in gt):
                        cnt[f"{name}_mirror"] += 1
    return mount, cnt


def main() -> None:
    from seedsets import load_seeds

    seeds = load_seeds()
    jobs = [(int(s), m, d) for s in seeds["development"] for m in ("lamppost", "facade") for d in ("low", "high")]
    with mp.get_context("spawn").Pool(4) as pool:
        parts = pool.map(_job, [(j,) for j in jobs], chunksize=1)
    out = {"definition": __doc__, "per_mount": {}}
    for mount in ("all", "lamppost", "facade"):
        tot = {}
        for m, c in parts:
            if mount in ("all", m):
                for k, v in c.items():
                    tot[k] = tot.get(k, 0) + v
        out["per_mount"][mount] = {**tot,
                                   "tracks_mirror_share": tot["tracks_mirror"] / max(tot["tracks_false"], 1),
                                   "clusters_mirror_share": tot["clusters_mirror"] / max(tot["clusters_false"], 1),
                                   "tracks_behind_share": tot["tracks_behind"] / max(tot["tracks_false"], 1),
                                   "clusters_behind_share": tot["clusters_behind"] / max(tot["clusters_false"], 1),
                                   "false_tracks_per_cpi": tot["tracks_false"] / tot["frames"], "unmatched_clusters_per_cpi": tot["clusters_false"] / tot["frames"]}
        v = out["per_mount"][mount]
        print(f"{mount}: false tracks {v['false_tracks_per_cpi']:.2f}/CPI, behind panel {100 * v['tracks_behind_share']:.1f} %, at a target's mirror {100 * v['tracks_mirror_share']:.1f} %; "
              f"unmatched clusters {v['unmatched_clusters_per_cpi']:.2f}/CPI, behind {100 * v['clusters_behind_share']:.1f} %, mirror {100 * v['clusters_mirror_share']:.1f} %")
    (ROOT / "results" / "P2" / "diag_radar_mirror_dev.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
