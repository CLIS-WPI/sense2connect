"""Git commit, dirty tree, and config identity for every cache entry.

A loader refuses an entry when the commit, the hash of uncommitted
changes, or the config hash does not match this process.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = (ROOT / "configs" / "m2_scenario.yaml", ROOT / "configs" / "seeds.yaml")


class CacheRefused(RuntimeError):
    """The cache entry was built from different code or a different config."""


def version() -> dict[str, str]:
    """Commit, dirty-tree hash, and config hash for this process."""
    return dict(_version())


def attach(meta: dict[str, Any], *, kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``meta`` with a provenance record and its cache key."""
    identity = {
        "kind": kind,
        "seed": meta.get("seed"),
        "mount": meta.get("mount"),
        "density": meta.get("density"),
        "dt_s": meta.get("dt_s"),
        "n_frames": meta.get("n_frames"),
    }
    if extra:
        identity.update(extra)
    version = _version()
    cache_key = hashlib.sha256(
        (json.dumps(identity, sort_keys=True, default=str) + version["config_hash"]).encode("utf-8")
    ).hexdigest()
    stamped = dict(meta)
    stamped["provenance"] = {**version, "cache_key": cache_key}
    return stamped


def require(meta: dict[str, Any], *, kind: str, extra: dict[str, Any] | None = None) -> None:
    """Raise when this entry is not from the current code and config."""
    got = meta.get("provenance")
    if not isinstance(got, dict):
        raise CacheRefused(f"{kind} cache has no provenance")
    expected = attach(meta, kind=kind, extra=extra)["provenance"]
    for field in ("git_commit", "dirty_hash", "config_hash", "cache_key"):
        if got.get(field) != expected[field]:
            raise CacheRefused(f"{kind} cache {field} does not match the current tree")


def scoped_version(sources: list[Path]) -> dict[str, str]:
    """Content hash of the producing sources and the configs.

    For caches that are expensive to rebuild and depend only on a few
    modules (the comm geometry trace). Edits elsewhere in the tree do not
    invalidate them; edits to ``sources`` or the configs do.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in sources):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {"source_hash": digest.hexdigest(), "config_hash": _version()["config_hash"]}


def require_scoped(meta: dict[str, Any], *, kind: str, sources: list[Path]) -> None:
    """Raise when a scoped cache entry is not from the current sources and configs."""
    got = meta.get("provenance")
    if not isinstance(got, dict) or got.get("kind") != kind:
        raise CacheRefused(f"{kind} cache has no scoped provenance")
    expected = scoped_version(sources)
    for field in ("source_hash", "config_hash"):
        if got.get(field) != expected[field]:
            raise CacheRefused(f"{kind} cache {field} does not match the current sources")


# --- Stage-scoped provenance of the radar detection caches (detections_<n>.json).
# The detection stage depends on the sensing code and the sensing part of the
# scenario config, not on the rest of the tree (xApp, scripts, paper). This file
# only writes and checks stamps, so it is not part of the stage hash; that also
# lets the stage functions below live here without changing the hash.
# Caches written before this stamp existed carry only the whole-tree record of
# the M2 run; they are checked against the stage hash of the sources in the M2
# commit, which holds the sensing code that produced them.
DETECTION_KIND = "detections_stage"
DETECTION_LEGACY_COMMIT = "a35536b"  # "M2: sensing pipeline, map tracker, budgets"
_SENSING_CONFIG_KEYS = ("carrier_hz", "numerology", "n_subcarriers", "sensing_radar")


def _detection_source_paths(names: list[str]) -> list[str]:
    return sorted(n for n in names if n.startswith("sim/sensing/") and n.endswith(".py") and not n.endswith("/provenance.py"))


def _sensing_config_bytes(text: str) -> bytes:
    import yaml

    raw = yaml.safe_load(text)
    return json.dumps({k: raw.get(k) for k in _SENSING_CONFIG_KEYS}, sort_keys=True, default=str).encode("utf-8")


def detection_stage_hash(commit: str | None = None) -> dict[str, str]:
    """Hash of the sensing sources and sensing config, in the working tree or at ``commit``."""
    if commit is None:
        names = [str(p.relative_to(ROOT)) for p in (ROOT / "sim" / "sensing").glob("*.py")]
        read = lambda n: (ROOT / n).read_bytes()  # noqa: E731
        config = (ROOT / "configs" / "m2_scenario.yaml").read_text(encoding="utf-8")
    else:
        names = str(_git_uncached("ls-tree", "-r", "--name-only", commit, "sim/sensing")).split()
        read = lambda n: _git_uncached("show", f"{commit}:{n}", binary=True)  # noqa: E731
        config = str(_git_uncached("show", f"{commit}:configs/m2_scenario.yaml"))
    src = hashlib.sha256()
    for name in _detection_source_paths(names):
        src.update(name.encode("utf-8"))
        src.update(read(name))
    return {"source_hash": src.hexdigest(), "sensing_config_hash": hashlib.sha256(_sensing_config_bytes(config)).hexdigest()}


def require_detection_stage(payload: dict[str, Any]) -> str:
    """Raise unless a detection cache matches the current sensing stage. Returns how it validated."""
    current = detection_stage_hash()
    stamp = payload.get("stage_provenance")
    if isinstance(stamp, dict):
        reference, how = stamp, "stage stamp"
    else:
        if not isinstance(payload.get("provenance"), dict):
            raise CacheRefused("detection cache has no provenance")
        reference, how = detection_stage_hash(DETECTION_LEGACY_COMMIT), f"legacy cache, sensing stage of {DETECTION_LEGACY_COMMIT}"
    for field in ("source_hash", "sensing_config_hash"):
        if reference.get(field) != current[field]:
            raise CacheRefused(f"detection cache {field} differs from the current sensing stage ({how}); rebuild detections")
    return how


def _git_uncached(*args: str, binary: bool = False) -> str | bytes:
    command = ["git", "-c", "safe.directory=*", *args]
    return subprocess.check_output(command, cwd=ROOT, text=not binary)


@lru_cache(maxsize=1)
def _git(*args: str) -> str | bytes:
    command = ["git", "-c", "safe.directory=*", *args]
    text = args[0] != "diff"
    return subprocess.check_output(command, cwd=ROOT, text=text)


def _version() -> dict[str, str]:
    commit = str(_git("rev-parse", "HEAD")).strip()
    diff = _git("diff", "HEAD")
    untracked = str(
        _git("ls-files", "--others", "--exclude-standard")
    )
    digest = hashlib.sha256(diff)
    for line in untracked.splitlines():
        path = ROOT / line
        if path.is_file():
            digest.update(line.encode("utf-8"))
            digest.update(path.read_bytes())
    config = hashlib.sha256()
    for path in _CONFIGS:
        config.update(path.name.encode("utf-8"))
        config.update(path.read_bytes())
    return {"git_commit": commit, "dirty_hash": digest.hexdigest(), "config_hash": config.hexdigest()}
