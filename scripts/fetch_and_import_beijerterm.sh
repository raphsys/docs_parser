#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_DIR="${1:-datasets/terminology_sources/external/beijerterm}"
OUT_DIR="${2:-ai_models/translation/glossaries}"

mkdir -p "$(dirname "$REPO_DIR")"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[sync] Updating existing beijerterm repo..."
  git -C "$REPO_DIR" pull --ff-only
else
  echo "[sync] Cloning beijerterm repo..."
  git clone --depth 1 https://github.com/michaelbeijer/beijerterm.git "$REPO_DIR"
fi

echo "[import] Parsing markdown glossaries..."
./.docs-parser/bin/python scripts/import_beijerterm.py --repo-dir "$REPO_DIR" --output-dir "$OUT_DIR"

echo "[ok] beijerterm import done"
