"""TVT freeze: copy every tuned parameter of T1-T6 from results/ into committed files under configs/tvt_frozen/.

Human decision before the freeze: T7 must not re-tune; it reads these files. Every value is copied
verbatim from the result file that produced it (no number typed by hand); each output records its
source path and the source's sha256. Duplicated parameters (a scheme tuned in more than one run) must
agree, otherwise the script stops. The learned-baseline weights are not copied; they are referenced by
path and sha256 (scripts/tvt_t5_handover.py --learned-manifest refuses other weights).

Outputs (configs/tvt_frozen/):
  tracker.json                 T4 tracker parameters incl. the constant visibility prior (scripts/tvt_t4_track.py --params)
  visibility_calibration.json  T3 isotonic recalibration and coverage prior (scripts/tvt_t4_track.py --calibration)
  handover_<model>.json        per-margin parameters of every handover scheme for signaling model ideal / failure_aware
                               (scripts/tvt_t5_handover.py --signaling <model> --fixed); primary objective wrap-masked
  handover_intersection_<model>.json   the same, re-tuned on the intersection tuning seeds (T6 second deployment)
  p2_estimator.json            estimator-A parameters of the planner_p2 inputs (back-to-back panels: re-tuned on the tuning
                               seeds, results/TVT/est_A/est_tuned_A.json; isotropic: paper 2's)
  learned.json                 path + sha256 + training summary of the learned predictor (--learned-manifest)
  manifest.json                sources, sha256, git commit of the freeze
Run: python scripts/tvt_freeze_params.py   (container), then on the HOST: bash scripts/tvt_freeze_stamp.sh stamp
(git is not available in the container: the commit is filled from the host and checked with
bash scripts/tvt_freeze_stamp.sh check; scripts/test_tvt.py requires a full commit hash).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
OUT = ROOT / "configs" / "tvt_frozen"

# scheme parameters per signaling model (signaling round before the freeze: every scheme re-tuned in one run per model)
MODELS = ("ideal", "failure_aware")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def scheme_params(files: list[Path]) -> tuple[dict, dict]:
    out: dict = {}
    origin: dict = {}
    for f in files:
        d = json.loads(f.read_text())
        for lab, cell in d["schemes"].items():
            for sch, v in cell.items():
                if " vs " in sch or "params" not in v:
                    continue
                prev = out.setdefault(lab, {}).get(sch)
                if prev is not None:
                    if prev["params"] != v["params"]:
                        raise SystemExit(f"{lab} {sch}: {rel(f)} {v['params']} != {origin[(lab, sch)]} {prev['params']}")
                    continue
                out[lab][sch] = {"params": v["params"]}
                origin[(lab, sch)] = rel(f)
    return out, {f"{lab} | {sch}": src for (lab, sch), src in origin.items()}


def _git_head() -> str:
    """HEAD if git works here (host), else "" - stamped from the host by scripts/tvt_freeze_stamp.sh stamp."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from sim.tvt.panels import enabled as _pn

    manifest = {"definition": __doc__, "array": "back_to_back (configs/tvt.yaml panels)" if _pn() else "single_iso", "git_commit": _git_head(),
                "files": {}}

    def write(name: str, obj: dict, sources: list[Path]) -> None:
        obj = {**obj, "_sources": {rel(s): sha(s) for s in sources}}
        (OUT / name).write_text(json.dumps(obj, indent=1) + "\n")
        manifest["files"][name] = {"sources": obj["_sources"], "sha256": sha(OUT / name)}
        print(f"wrote configs/tvt_frozen/{name} from {', '.join(rel(s) for s in sources)}")

    f = RES / "TVT" / "T4" / "tuned.json"
    t4 = json.loads(f.read_text())
    write("tracker.json", {"params": t4["params"], "tuning": {"grid": t4["grid"], "best": t4["best"]}}, [f])

    f = RES / "TVT" / "T3" / "visibility.json"
    t3 = json.loads(f.read_text())
    write("visibility_calibration.json", {"calibration": t3["calibration"], "coverage_prior": t3["coverage_prior"], "config": t3["config"]}, [f])

    for old in ("handover.json", "handover_intersection.json"):  # first-round files, superseded by the per-model files
        if (OUT / old).exists():
            (OUT / old).unlink()
    for m in MODELS:
        f = RES / "TVT" / "T5" / f"handover_{m}.json"
        params, origin = scheme_params([f])
        d = json.loads(f.read_text())
        write(f"handover_{m}.json", {"model": d["model"], "signaling": d["signaling"], "signaling_config": d["signaling_config"], "evaluation": d["evaluation"],
                                     "margin_ref_db": d["margin_ref_db"], "grids": d["grids"], "schemes": params, "origin": origin}, [f])
        f = RES / "TVT" / "T5" / f"handover_intersection_tuned_{m}.json"
        params, origin = scheme_params([f])
        d = json.loads(f.read_text())
        write(f"handover_intersection_{m}.json", {"model": d["model"], "signaling": d["signaling"], "margin_ref_db": d["margin_ref_db"], "schemes": params,
                                                 "origin": origin}, [f])

    from sim.tvt.panels import enabled

    if enabled():  # back-to-back panels: estimator A re-tuned on the panel measurements (scripts/tvt_est_tune.py)
        f = RES / "TVT" / "est_A" / "est_tuned_A.json"
        d = json.loads(f.read_text())
        write("p2_estimator.json", {"array": "back_to_back", "tuned": d["tuned"], "base_v1": d["base"]}, [f])
    else:
        f = RES / "P2" / "est_tuned.json"
        write("p2_estimator.json", {"tuned": json.loads(f.read_text())["tuned"]}, [f])

    w = RES / "TVT" / "T5" / "learned.pt"
    lj = RES / "TVT" / "T5" / "learned.json"
    if w.exists() and lj.exists():
        info = json.loads(lj.read_text())
        obj = {"path": rel(w), "sha256": sha(w), **{k: info[k] for k in ("train_seeds", "val_seeds", "ue_sigma_m", "taus_s", "val_auc_per_tau")}}
        write("learned.json", obj, [lj])
    else:
        print("learned predictor not trained yet: learned.json NOT written")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
