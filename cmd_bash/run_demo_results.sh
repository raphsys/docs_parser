#!/usr/bin/env bash
set -Eeuo pipefail

# Lance un essai local PAGEPRINT -> PAGETRANSLATE -> PAGERECONSTRUCT
# avec l'environnement Python officiel du projet :
#   .docs-parser/bin/python
#
# Résultats dans results/demo_YYYYMMDD_HHMMSS/.
#
# À lancer depuis n'importe où :
#   bash ~/Mes_Projets/docs_parser/cmd_bash/run_demo_results.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PROJECT_PYTHON="$ROOT/.docs-parser/bin/python"

if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "ERREUR: environnement Python du projet introuvable ou non exécutable." >&2
  echo "Attendu exactement : $PROJECT_PYTHON" >&2
  exit 2
fi

PYTHON_BIN="$PROJECT_PYTHON"

if [[ ! -f "tools/run_pipeline_full_demo.py" ]]; then
  echo "ERREUR: fichier absent: tools/run_pipeline_full_demo.py" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/demo_${TS}}"
COUNT="${COUNT:-5}"
SEED="${SEED:-20260616}"
MIN_PAGES="${MIN_PAGES:-20}"
PDF_DIR="${PDF_DIR:-tests/doc_pdf}"
ENGINE="${ENGINE:-ct2}"
MODEL="${MODEL:-opus_mt_tc_big_en_fr}"
SOURCE_LANG="${SOURCE_LANG:-en}"
TARGET_LANG="${TARGET_LANG:-fr}"
PUBREADY_MODE="${PUBREADY_MODE:-review}"
TID_CACHE="${TID_CACHE:-results/_tid_cache}"
RUN_TESTS="${RUN_TESTS:-0}"
REUSE_TID="${REUSE_TID:-0}"

mkdir -p "$OUT"

LOG="$OUT/run.log"
REPORT="$OUT/report.md"

{
  echo "=== vSense Studio local demo ==="
  echo "date           : $(date -Is)"
  echo "root           : $ROOT"
  echo "python         : $PYTHON_BIN"
  "$PYTHON_BIN" -c 'import sys; print("python_exe     :", sys.executable)'
  echo "out            : $OUT"
  echo "count          : $COUNT"
  echo "seed           : $SEED"
  echo "pdf_dir        : $PDF_DIR"
  echo "engine         : $ENGINE"
  echo "model          : $MODEL"
  echo "source_lang    : $SOURCE_LANG"
  echo "target_lang    : $TARGET_LANG"
  echo "pubready_mode  : $PUBREADY_MODE"
  echo "tid_cache      : $TID_CACHE"
  echo "reuse_tid      : $REUSE_TID"
  echo "run_tests      : $RUN_TESTS"
  echo
} | tee "$LOG"

echo "==> Vérification syntaxe Python..." | tee -a "$LOG"
"$PYTHON_BIN" -m compileall -q pageprint pagetranslate pagereconstruct pipelines tools 2>&1 | tee -a "$LOG"

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "==> Tests rapides..." | tee -a "$LOG"
  "$PYTHON_BIN" -m pytest -q tests/pageprint tests/pagetranslate tests/pagereconstruct 2>&1 | tee -a "$LOG"
fi

CMD=("$PYTHON_BIN" tools/run_pipeline_full_demo.py
  --out "$OUT"
  --engine "$ENGINE"
  --model "$MODEL"
  --source-lang "$SOURCE_LANG"
  --target-lang "$TARGET_LANG"
  --pubready-mode "$PUBREADY_MODE"
  --tid-cache "$TID_CACHE"
)

if [[ "$REUSE_TID" == "1" ]]; then
  CMD+=(--reuse-tid)
fi

if [[ -n "${PDF:-}" && -n "${PAGE:-}" ]]; then
  CMD+=(--pdf "$PDF" --page "$PAGE")
else
  CMD+=(--pdf-dir "$PDF_DIR" --count "$COUNT" --seed "$SEED" --min-pages "$MIN_PAGES")
fi

echo "==> Commande pipeline:" | tee -a "$LOG"
printf ' %q' "${CMD[@]}" | tee -a "$LOG"
echo | tee -a "$LOG"
echo | tee -a "$LOG"

set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG"
PIPE_STATUS=${PIPESTATUS[0]}
set -e

echo | tee -a "$LOG"
echo "==> Statut pipeline: $PIPE_STATUS" | tee -a "$LOG"

echo "==> Génération rapport minimal..." | tee -a "$LOG"

"$PYTHON_BIN" - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary_path = out / "summary.json"
report_path = out / "report.md"

def load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

summary = load_json(summary_path) or []

lines = [
    "# Rapport vSense Studio",
    "",
    f"Dossier: `{out}`",
    "",
    "| page | translated | preserved | protected | findings | audit_status | pubready | blockers |",
    "|---|---:|---:|---:|---:|---|---:|---|",
]

for item in summary:
    tag = item.get("tag", "?")
    translated = item.get("translated_text_count", "")
    preserved = (item.get("preserved_overlay_count") or 0) + (item.get("preserved_underlay_count") or 0)
    protected = item.get("protected_region_count", "")
    findings = item.get("finding_count", "")
    status = item.get("status", "")
    pr = item.get("pubready") or {}
    score = pr.get("score", "")
    blockers = ", ".join(pr.get("hard_blockers") or [])
    lines.append(f"| `{tag}` | {translated} | {preserved} | {protected} | {findings} | {status} | {score} | {blockers} |")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"rapport: {report_path}")
PY

echo | tee -a "$LOG"
echo "==> Terminé." | tee -a "$LOG"
echo "Résultats: $OUT" | tee -a "$LOG"
echo "Rapport  : $REPORT" | tee -a "$LOG"
echo "Log      : $LOG" | tee -a "$LOG"

exit "$PIPE_STATUS"
