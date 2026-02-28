#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${1:-datasets/terminology_sources}"
OUTPUT_DIR="${2:-ai_models/translation/glossaries}"
DEFAULT_DOMAIN="${DEFAULT_DOMAIN:-general}"
DEFAULT_SOURCE_LANG="${DEFAULT_SOURCE_LANG:-en}"
DEFAULT_TARGET_LANG="${DEFAULT_TARGET_LANG:-fr}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[error] input dir not found: $INPUT_DIR" >&2
  exit 1
fi

echo "[build] input: $INPUT_DIR"
echo "[build] output: $OUTPUT_DIR"

./.docs-parser/bin/python scripts/import_terminology_sources.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --domain "$DEFAULT_DOMAIN" \
  --source-lang "$DEFAULT_SOURCE_LANG" \
  --target-lang "$DEFAULT_TARGET_LANG"

echo "[ok] glossary build completed"
