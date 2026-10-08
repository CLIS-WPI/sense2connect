#!/usr/bin/env bash
# TVT freeze: fill and check the git commit of configs/tvt_frozen/manifest.json FROM THE HOST (git is not available in the
# container, so scripts/tvt_freeze_params.py writes an empty "git_commit"). Human decision, fourth TVT round.
#   bash scripts/tvt_freeze_stamp.sh stamp   after scripts/tvt_freeze_params.py: writes HEAD into the manifest; refuses if
#                                            any tracked file outside configs/tvt_frozen/ and *.md (documentation) has
#                                            uncommitted changes, so the stamped commit holds exactly the code of the freeze
#   bash scripts/tvt_freeze_stamp.sh check   the stamped commit exists, is an ancestor of HEAD, and HEAD differs from it
#                                            only in configs/tvt_frozen/ and *.md; the working tree is clean there too
# Exit status 0 = pass. No Python (runs on the host).
set -euo pipefail
cd "$(dirname "$0")/.."
M=configs/tvt_frozen/manifest.json
ALLOWED='^(configs/tvt_frozen/|.*\.md$)'
mode="${1:-check}"
dirty=$(git status --porcelain --untracked-files=no | awk '{print $NF}' | grep -Ev "$ALLOWED" || true)
case "$mode" in
  stamp)
    [ -z "$dirty" ] || { echo "refused: uncommitted changes outside configs/tvt_frozen/ and *.md:"; echo "$dirty"; exit 1; }
    head=$(git rev-parse HEAD)
    grep -q '"git_commit": "[0-9a-f]*"' "$M" || { echo "refused: no git_commit field in $M"; exit 1; }
    sed -i -E "s/\"git_commit\": \"[0-9a-f]*\"/\"git_commit\": \"$head\"/" "$M"
    echo "stamped $M with $head"
    ;;
  check)
    c=$(sed -n -E 's/.*"git_commit": "([0-9a-f]*)".*/\1/p' "$M")
    [[ "$c" =~ ^[0-9a-f]{40}$ ]] || { echo "FAIL: git_commit '$c' is not a full commit hash (stamp from the host)"; exit 1; }
    git cat-file -e "$c^{commit}" 2>/dev/null || { echo "FAIL: commit $c does not exist"; exit 1; }
    git merge-base --is-ancestor "$c" HEAD || { echo "FAIL: $c is not an ancestor of HEAD"; exit 1; }
    changed=$(git diff --name-only "$c" HEAD | grep -Ev "$ALLOWED" || true)
    [ -z "$changed" ] || { echo "FAIL: files changed since the freeze commit $c:"; echo "$changed"; exit 1; }
    [ -z "$dirty" ] || { echo "FAIL: uncommitted changes outside configs/tvt_frozen/ and *.md:"; echo "$dirty"; exit 1; }
    echo "OK: configs/tvt_frozen frozen at $c; code unchanged since ($(git rev-list --count "$c"..HEAD) later commit(s), frozen files / docs only)"
    ;;
  *) echo "usage: $0 stamp|check"; exit 2 ;;
esac
