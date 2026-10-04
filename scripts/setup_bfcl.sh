#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${BFCL_CHECKOUT:-"$ROOT/gorilla"}
REVISION=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8

if [[ -e "$DEST" ]]; then
  echo "Refusing to overwrite existing path: $DEST" >&2
  exit 1
fi

git clone --filter=blob:none --no-checkout https://github.com/ShishirPatil/gorilla.git "$DEST"
git -C "$DEST" sparse-checkout set berkeley-function-call-leaderboard
git -C "$DEST" checkout "$REVISION"

echo "BFCL checkout ready at $DEST/berkeley-function-call-leaderboard"
echo "export BFCL_ROOT='$DEST/berkeley-function-call-leaderboard'"
