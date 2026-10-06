"""Register the TVT custom scenes with sionna.rt.scene so that the unchanged paper-1 loaders
(sim/scenes/loop._scene_by_name looks up ``sionna.rt.scene.<name>``) can load them by name.
Call ``register()`` in every process (and pool worker) before a scene is loaded."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = {"tvt_intersection": ROOT / "configs" / "scenes" / "tvt_intersection" / "tvt_intersection.xml"}


def register() -> None:
    import sionna.rt as rt

    for name, path in SCENES.items():
        setattr(rt.scene, name, str(path))
