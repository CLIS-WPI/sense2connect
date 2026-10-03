"""Rewrite provenance on finished traces so it matches this tree.

The ray traces are unchanged. Workers stamped different dirty hashes
because source files were added while they were running.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.sensing.provenance import attach, version

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"
EXTRA = {"min_gap_s": 0.5, "los_threshold_db": 10.0}


def main() -> None:
    metas = 0
    events = 0
    for mount in ("lamppost", "facade"):
        for density in ("low", "high"):
            for path in (CACHE / mount / density).glob("seed_*/meta.json"):
                meta = json.loads(path.read_text(encoding="utf-8"))
                path.write_text(json.dumps(attach(meta, kind="trace"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
                metas += 1
                event_path = path.parent / "events.json"
                if not event_path.exists():
                    continue
                payload = json.loads(event_path.read_text(encoding="utf-8"))
                event_path.write_text(
                    json.dumps(attach(payload, kind="events", extra=EXTRA), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                events += 1
    print("restamped", metas, events, version()["dirty_hash"], flush=True)


if __name__ == "__main__":
    main()
