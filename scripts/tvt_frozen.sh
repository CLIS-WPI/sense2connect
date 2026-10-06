#!/usr/bin/env bash
# Run a provenance-checked trace step from a FROZEN clone of the committed tree.
# The sensing-trace caches carry whole-tree provenance (git commit + hash of uncommitted changes,
# sim/sensing/provenance.py); a trace and the detections computed from it must therefore run in a tree
# that does not change in between. This script clones the current HEAD into results/TVT/frozen_tree
# (inside the repository, git-ignored; its own .git, so it is never dirty), runs the given command there
# in the sionna container (GPU 1), and copies the new cache directories back into the main results tree
# without overwriting anything that exists (results/cache/<mount>/<density>/seed_<s>, results/P2/cache/...,
# results/TVT/T6/si/...). Downstream TVT code uses only scope-checked caches (comm geometry, path
# coefficients, timelines) and the radar detections, which carry their stage provenance but are not
# refused across trees.
# Usage: bash scripts/tvt_frozen.sh <log-name> <python script and args, relative to the clone>
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift
fz="results/TVT/frozen_tree"
head=$(git rev-parse HEAD)
if [ ! -d "$fz/.git" ]; then
  git clone -q --no-hardlinks . "$fz"
fi
git -C "$fz" fetch -q "$PWD" "$head" 2>/dev/null || git -C "$fz" fetch -q origin
git -C "$fz" checkout -q "$head"
test -z "$(git -C "$fz" status --porcelain)" || { echo "frozen clone is dirty"; exit 1; }
mkdir -p results/TVT/logs
echo "frozen tree at $head" > "results/TVT/logs/frozen_${name}.log"
docker compose run --rm -w "/workspace/$fz" sionna "$@" >> "results/TVT/logs/frozen_${name}.log" 2>&1
# copy new cache directories back (never overwrite)
docker compose run --rm sionna sh -c "cp -rn $fz/results/cache/. results/cache/ 2>/dev/null; mkdir -p results/P2/cache results/TVT/T6; cp -rn $fz/results/P2/cache/. results/P2/cache/ 2>/dev/null; cp -rn $fz/results/TVT/T6/. results/TVT/T6/ 2>/dev/null; true"
echo "done $name" >> "results/TVT/logs/frozen_${name}.log"
