"""TVT seed sets (configs/seeds_tvt.yaml) and the held-out guard.

``load()`` returns tuning, development and training seeds. The held-out seeds
4001-4010 are returned only by ``heldout()``, which refuses unless the git tag
``tvt-freeze`` exists (read from .git directly, no git binary needed). Every
TVT script resolves seeds through ``check()``, which refuses held-out seeds
before the freeze and the consumed conference held-out sets always.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE_TAG = "tvt-freeze"


class HeldOutRefused(RuntimeError):
    """A held-out or consumed seed was requested before it may be opened."""


def _raw() -> dict:
    from sim.scenes.config import load_yaml

    raw = load_yaml(ROOT / "configs" / "seeds_tvt.yaml")
    sets = {k: [int(s) for s in v] for k, v in raw.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(sets[a]) & set(sets[b]):
                raise SystemExit(f"seed sets {a} and {b} overlap")
    return sets


def freeze_tag_exists(git_dir: Path | None = None) -> bool:
    """True when refs/tags/tvt-freeze exists (loose or packed ref)."""
    g = git_dir or ROOT / ".git"
    if (g / "refs" / "tags" / FREEZE_TAG).exists():
        return True
    packed = g / "packed-refs"
    if packed.exists():
        return any(line.strip().endswith(f"refs/tags/{FREEZE_TAG}") for line in packed.read_text().splitlines())
    return False


def load() -> dict[str, list[int]]:
    """Tuning, development and training seeds (never the held-out or consumed ones)."""
    s = _raw()
    return {k: s[k] for k in ("tuning", "development", "training")}


def heldout(git_dir: Path | None = None) -> list[int]:
    if not freeze_tag_exists(git_dir):
        raise HeldOutRefused(f"held-out seeds are sealed until the tag {FREEZE_TAG} exists")
    return _raw()["heldout"]


def check(seeds, git_dir: Path | None = None) -> list[int]:
    """Return ``seeds`` as ints; refuse consumed seeds always and held-out seeds before the freeze."""
    s = _raw()
    out = [int(x) for x in seeds]
    bad = set(out) & set(s["consumed"])
    if bad:
        raise HeldOutRefused(f"seeds {sorted(bad)} belong to consumed conference held-out sets")
    held = set(out) & set(s["heldout"])
    if held and not freeze_tag_exists(git_dir):
        raise HeldOutRefused(f"held-out seeds {sorted(held)} are sealed until the tag {FREEZE_TAG} exists")
    return out
