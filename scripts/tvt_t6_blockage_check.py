"""TVT T6: blockage-model check - 3GPP model B (knife-edge screens, the papers' model) vs Sionna RT 2.2
first-order diffraction around a box blocker.

Sionna RT 2.2 supports first-order diffraction (PathSolver(diffraction=True, edge_diffraction=True); verified:
with diffraction=True alone no path is found around an isolated box, with edge_diffraction=True the
diffracted path over the box edge appears). Geometry of the paper-1 canyon cross-section without buildings or ground (isolates the
blocker): O-RU at (0, 6.5, 5) m (lamppost height), UE at (dx, -7, 1.5) m; a blocker box moves along
x through the LoS: bus 12 x 2.5 x 3.2 m in the eastbound lane (y = -2.5) or pedestrian 0.5 x 0.5 x
1.75 m on the sidewalk next to the UE (y = -6). Blocker material: ITU "metal" (vehicle body) or a
perfect absorber (closest to model B's absorbing screen). Received power = sum over Sionna paths of
|a|^2 (single antennas, isotropic), loss = power without blocker / power with blocker [dB], with and
without diffraction; model B loss from the frozen sim.comm.blockage_torch.screen_loss_db for the same
box and segment. Writes results/TVT/T6/blockage_check.json. Run: python scripts/tvt_t6_blockage_check.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "TVT" / "T6"

CASES = {"bus": {"size": (12.0, 2.5, 3.2), "y": -2.5}, "pedestrian": {"size": (0.5, 0.5, 1.75), "y": -6.0}}
XML = """<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="metal"><string name="type" value="metal"/><float name="thickness" value="0.01"/></bsdf>
  <shape type="cube" id="blocker">
    <transform name="to_world"><scale x="{sx}" y="{sy}" z="{sz}"/><translate x="0" y="0" z="0"/></transform>
    <ref id="metal" name="bsdf"/>
  </shape>
</scene>
"""


def main() -> None:
    import drjit as dr  # noqa: F401
    import mitsuba as mi
    import torch

    from sim.comm.blockage_torch import screen_loss_db
    from sionna.rt import AbsorberRadioMaterial, PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    wl = 299792458.0 / 28e9
    tx_p = np.array([0.0, 6.5, 5.0])
    res = {"definition": __doc__, "cases": {}}
    xs = np.linspace(-12.0, 12.0, 49)
    for name, case in CASES.items():
        L, W, H = case["size"]
        path = OUT / f"_blk_{name}.xml"
        path.write_text(XML.format(sx=L / 2, sy=W / 2, sz=H / 2))
        for mat in ("metal", "absorber"):
            for dx in (0.0, 10.0):
                rx_p = np.array([dx, -7.0, 1.5])
                scene = load_scene(str(path), merge_shapes=False)
                scene.frequency = 28e9
                scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
                scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
                scene.add(Transmitter("tx", position=tx_p.tolist()))
                scene.add(Receiver("rx", position=rx_p.tolist()))
                blk = scene.get("blocker")
                if mat == "absorber":
                    blk.radio_material = AbsorberRadioMaterial("absorber")
                solver = PathSolver()
                rows = []
                # free-space reference (blocker far away)
                blk.position = mi.Point3f(1000.0, 1000.0, H / 2)
                ref = {}
                for diff in (False, True):
                    p = solver(scene, max_depth=1, diffraction=diff, edge_diffraction=diff, specular_reflection=False, refraction=False)
                    a = np.asarray(p.cir(out_type="numpy")[0]).reshape(-1)
                    ref[diff] = float(np.sum(np.abs(a) ** 2))
                for x in xs:
                    # the blocker crosses the LoS: place its centre on the line at the lane / sidewalk y
                    s = (case["y"] - tx_p[1]) / (rx_p[1] - tx_p[1])
                    xl = tx_p[0] + s * (rx_p[0] - tx_p[0])
                    c = np.array([xl + x, case["y"], H / 2])
                    blk.position = mi.Point3f(*c.tolist())
                    out = {"offset_m": float(x)}
                    for diff in (False, True):
                        p = solver(scene, max_depth=1, diffraction=diff, edge_diffraction=diff, specular_reflection=False, refraction=False)
                        a = np.asarray(p.cir(out_type="numpy")[0]).reshape(-1)
                        pw = float(np.sum(np.abs(a) ** 2))
                        out["sionna_diffraction" if diff else "sionna_no_diffraction"] = 10 * math.log10(max(ref[diff], 1e-300) / max(pw, 1e-300))
                    mb = screen_loss_db(torch.as_tensor(tx_p), torch.as_tensor(rx_p), torch.as_tensor(c), torch.as_tensor(L), torch.as_tensor(W),
                                        torch.as_tensor(H), wl)
                    out["model_b"] = float(mb) if math.isfinite(float(mb)) else 300.0
                    rows.append(out)
                key = f"{name}|{mat}|dx{dx:g}"
                arr = {k: np.array([r[k] for r in rows]) for k in ("sionna_diffraction", "sionna_no_diffraction", "model_b")}
                blocked = arr["model_b"] >= 10.0
                res["cases"][key] = {"rows": rows, "n_blocked_modelB_ge10": int(blocked.sum()),
                                     "median_abs_diff_blocked_db": float(np.median(np.abs(np.minimum(arr["sionna_diffraction"], 60) - np.minimum(arr["model_b"], 60))[blocked])) if blocked.any() else None,
                                     "agreement_blocked_ge10db": float(np.mean((arr["sionna_diffraction"] >= 10.0) == blocked)),
                                     "max_loss_diffraction_db": float(np.max(np.minimum(arr["sionna_diffraction"], 300))), "max_loss_model_b_db": float(np.max(arr["model_b"]))}
                print(key, {k: v for k, v in res["cases"][key].items() if k != "rows"}, flush=True)
        path.unlink()
    (OUT / "blockage_check.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
