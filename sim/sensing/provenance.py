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
