#!/usr/bin/env bash
# Unpack the Task 4 release into data/raw. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/narrative-similarity-dataset"
ZIP="${1:-$ROOT/../narrative-similarity-dataset-main.zip}"

if [ -d "$DEST/dev" ]; then echo "already unpacked at $DEST"; exit 0; fi
mkdir -p "$ROOT/data/raw"
if [ -f "$ZIP" ]; then
  echo "unpacking $ZIP"
  TMP="$(mktemp -d)"
  unzip -q "$ZIP" -d "$TMP"
  SRC="$(find "$TMP" -maxdepth 2 -type d -name dev | head -1 | xargs dirname)"
  mkdir -p "$DEST" && cp -R "$SRC"/. "$DEST"/
  rm -rf "$TMP"
else
  echo "no local zip; cloning the public repo"
  git clone --depth 1 https://github.com/narrative-similarity-task/narrative-similarity-dataset "$DEST"
fi
echo "splits now in $DEST:"; ls "$DEST"
