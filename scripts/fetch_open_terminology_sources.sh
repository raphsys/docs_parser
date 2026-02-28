#!/usr/bin/env bash
set -euo pipefail

# Download open terminology resources to datasets/terminology_sources/external
# Then convert them to project glossaries via build_glossaries_from_sources.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-datasets/terminology_sources/external}"
mkdir -p "$OUT_DIR"

# NOTE:
# Keep only sources compatible with commercial usage and clear licensing.
# You can add/remove URLs in this list as needed.

URLS=(
  # IATE exports (EU terminology): user must provide direct export URLs if required.
  # "https://.../iate_export_science_en_fr.tbx"

  # Public domain / open dictionaries can be dropped here as csv/tsv/json/tmx/tbx.
)

if [[ ${#URLS[@]} -eq 0 ]]; then
  echo "[info] No URLs configured in scripts/fetch_open_terminology_sources.sh"
  echo "[info] Add URLs to URLS=(...) then rerun."
  exit 0
fi

for url in "${URLS[@]}"; do
  fname="$(basename "${url%%\?*}")"
  target="$OUT_DIR/$fname"
  echo "[download] $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 -o "$target" "$url"
  else
    wget -O "$target" "$url"
  fi
  echo "[ok] $target"
done

echo "[done] downloads in $OUT_DIR"
