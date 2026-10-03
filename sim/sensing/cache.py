"""Ray-trace a sensing job once and reload it for every radar variant.

One cache entry is one ``(seed, mount, density)`` at the scenario time
step. Paths keep the coefficient, delay, Doppler, angles, and object ids.
Ground truth is stored beside them. Later bandwidths and detector settings
read this directory and do not call a solver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_FIELDS = (
    "a",
    "tau",
    "doppler",
    "theta_r",
    "phi_r",
    "theta_t",
    "phi_t",
    "object_names",
    "interactions",
    "valid",
)


def cache_dir(root: Path, mount: str, density: str, seed: int) -> Path:
    """Directory for one traced job."""
    return Path(root) / mount / density / f"seed_{int(seed)}"


def labels_from_scene(scene: object, actor_positions_m: dict[str, np.ndarray]) -> dict[int, str]:
    """Map a Mitsuba object id to a load-stable name.

    Building and floor names are already stable. A nameless scatterer is
    labeled with the nearest actor, because Mitsuba's ``no-name-N`` index
    changes on every scene load.
    """
    labels: dict[int, str] = {}
    centers: dict[int, np.ndarray] = {}
    named: dict[str, np.ndarray] = {}
    for name, obj in scene.objects.items():  # type: ignore[attr-defined]
        box = obj.mi_mesh.bbox()
        center = np.array(
            [
                0.5 * (box.min.x + box.max.x),
                0.5 * (box.min.y + box.max.y),
                0.5 * (box.min.z + box.max.z),
            ],
            dtype=np.float64,
        )
        object_id = int(obj.object_id)
        text = str(name)
        centers[object_id] = center
        if text == "floor" or text.startswith("building") or text in actor_positions_m:
            labels[object_id] = text
            if text in actor_positions_m:
                named[text] = center
        else:
            labels[object_id] = text
    for object_id, text in list(labels.items()):
        if text == "floor" or text.startswith("building") or text in actor_positions_m:
            continue
        center = centers[object_id]
        if not actor_positions_m:
            labels[object_id] = "scatterer"
            continue
        nearest = min(
            actor_positions_m,
            key=lambda actor: float(np.linalg.norm(actor_positions_m[actor] - center)),
        )
        labels[object_id] = nearest
    return labels


def pack_paths(paths: object, labels: dict[int, str]) -> dict[str, np.ndarray]:
    """Copy the solver output that a later CPI can rebuild.

    ``a`` and ``tau`` are the coefficient and delay at the snapshot, not
    the 256-symbol CPI. Doppler rebuilds the slow-time phase.
    """
    coefficients, delays = paths.cir(  # type: ignore[attr-defined]
        sampling_frequency=1.0,
        num_time_steps=1,
        normalize_delays=False,
        out_type="numpy",
    )
    packed = {
        "a": np.asarray(coefficients),
        "tau": np.asarray(delays),
    }
    for name in _FIELDS:
        if name in packed or name == "object_names":
            continue
        packed[name] = _to_numpy(getattr(paths, name))
    packed["object_names"] = _object_names(_to_numpy(paths.objects), labels)
    inactive = np.asarray(packed["interactions"]) == 0
    packed["object_names"][inactive] = ""
    return _sort_paths(packed)


def save_static(directory: Path, packed: dict[str, np.ndarray]) -> None:
    """Write the empty-scene clutter paths."""
    directory.mkdir(parents=True, exist_ok=True)
    _save(directory / "static.npz", packed)


def save_frame(
    directory: Path,
    index: int,
    rcs: dict[str, np.ndarray],
    background: dict[str, np.ndarray],
    ground_truth: list[dict[str, Any]],
    comm: dict[str, np.ndarray] | None = None,
) -> None:
    """Write one snapshot. Path counts may differ across snapshots."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    groups = [("rcs", rcs), ("bg", background)]
    if comm is not None:
        groups.append(("comm", comm))
    for prefix, packed in groups:
        for name, value in packed.items():
            payload[f"{prefix}_{name}"] = np.asarray(value)
    payload["gt_x"] = np.asarray([row["x_m"] for row in ground_truth], dtype=np.float64)
    payload["gt_y"] = np.asarray([row["y_m"] for row in ground_truth], dtype=np.float64)
    payload["gt_z"] = np.asarray([row["z_m"] for row in ground_truth], dtype=np.float64)
    payload["gt_vx"] = np.asarray([row["vx_mps"] for row in ground_truth], dtype=np.float64)
    payload["gt_vy"] = np.asarray([row["vy_mps"] for row in ground_truth], dtype=np.float64)
    payload["gt_id"] = np.asarray([row["id"] for row in ground_truth])
    payload["gt_kind"] = np.asarray([row["kind"] for row in ground_truth])
    _save(directory / f"frame_{index:04d}.npz", payload)


def save_meta(directory: Path, meta: dict[str, Any]) -> None:
    """Write the job description next to the frames, including provenance."""
    from sim.sensing.provenance import attach

    directory.mkdir(parents=True, exist_ok=True)
    stamped = attach(meta, kind="trace")
    (directory / "meta.json").write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_static(directory: Path) -> dict[str, np.ndarray]:
    """Read the empty-scene clutter paths."""
    _require_trace(directory)
    return _load(directory / "static.npz")


def load_frame(directory: Path, index: int) -> dict[str, np.ndarray]:
    """Read one snapshot."""
    _require_trace(directory)
    return _load(directory / f"frame_{index:04d}.npz")


def load_meta(directory: Path) -> dict[str, Any]:
    """Read the job description. Refuses a different commit, dirty tree, or config."""
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    _require_trace(directory, meta)
    return meta


def split_frame(frame: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    """Return the path fields stored under ``prefix``."""
    return {name: frame[f"{prefix}_{name}"] for name in _FIELDS}


_checked: set[str] = set()


def _require_trace(directory: Path, meta: dict[str, Any] | None = None) -> None:
    from sim.sensing.provenance import require

    key = str(directory.resolve())
    if key in _checked:
        return
    if meta is None:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    require(meta, kind="trace")
    _checked.add(key)


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _sort_paths(packed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Order path slots by delay and geometry so two solves compare equal."""
    coefficients = np.array(packed["a"], copy=True)
    delays = np.array(packed["tau"], copy=True)
    if coefficients.ndim == 5:
        coefficients = coefficients[..., None]
    if delays.ndim == 3:
        delays = delays[..., None]
    n_rx = int(coefficients.shape[0])
    n_tx = int(coefficients.shape[2])
    n_paths = int(coefficients.shape[-2])
    order_by_link: list[np.ndarray] = []
    for rx_index in range(n_rx):
        for tx_index in range(n_tx):
            keys: list[bytes] = []
            for path_index in range(n_paths):
                pieces = [
                    np.ascontiguousarray(delays[rx_index, tx_index, path_index]),
                    np.ascontiguousarray(packed["doppler"][rx_index, tx_index, path_index]),
                    np.ascontiguousarray(packed["theta_r"][rx_index, tx_index, path_index]),
                    np.ascontiguousarray(packed["phi_r"][rx_index, tx_index, path_index]),
                    np.ascontiguousarray(packed["theta_t"][rx_index, tx_index, path_index]),
                    np.ascontiguousarray(packed["phi_t"][rx_index, tx_index, path_index]),
                    np.ascontiguousarray(packed["object_names"][:, rx_index, tx_index, path_index]),
                    np.ascontiguousarray(packed["valid"][rx_index, tx_index, path_index]),
                ]
                keys.append(b"".join(np.asarray(piece).tobytes() for piece in pieces))
            order_by_link.append(np.array(sorted(range(n_paths), key=lambda index: keys[index]), dtype=np.int64))
    link = 0
    for rx_index in range(n_rx):
        for tx_index in range(n_tx):
            order = order_by_link[link]
            link += 1
            coefficients[rx_index, :, tx_index, :, :, :] = np.take(
                coefficients[rx_index, :, tx_index, :, :, :], order, axis=-2
            )
            delays[rx_index, tx_index, :] = delays[rx_index, tx_index, order]
            for name in ("doppler", "theta_r", "phi_r", "theta_t", "phi_t", "valid"):
                values = np.array(packed[name], copy=True)
                values[rx_index, tx_index] = values[rx_index, tx_index, order]
                packed[name] = values
            for name in ("object_names", "interactions"):
                values = np.array(packed[name], copy=True)
                values[:, rx_index, tx_index] = np.take(values[:, rx_index, tx_index], order, axis=-1)
                packed[name] = values
    packed["a"] = coefficients
    packed["tau"] = delays
    return packed


def _object_names(objects: np.ndarray, labels: dict[int, str]) -> np.ndarray:
    """Load-stable shape names. Integer Mitsuba ids change across scene loads."""
    names = np.empty(objects.shape, dtype=object)
    for index in np.ndindex(objects.shape):
        names[index] = labels.get(int(objects[index]), "")
    return names


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, tuple) and len(value) == 2:
        return np.asarray(value[0]) + 1j * np.asarray(value[1])
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)  # type: ignore[union-attr]
    return np.asarray(array)
