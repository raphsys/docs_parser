#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(pwd)"
BACKUP_DIR="$PROJECT_ROOT/results/_patch_backups/background_purity_v1_1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

need() {
  if [[ ! -f "$PROJECT_ROOT/$1" ]]; then
    echo "ERREUR: $1 introuvable. Lance depuis la racine docs_parser/." >&2
    exit 2
  fi
}

need pipelines/background_cover.py
need pipelines/background_cleaner.py

cp -a "$PROJECT_ROOT/pipelines/background_cover.py" "$BACKUP_DIR/background_cover.py.bak"
cp -a "$PROJECT_ROOT/pipelines/background_cleaner.py" "$BACKUP_DIR/background_cleaner.py.bak"

cp -f "$PROJECT_ROOT/patchs/files/pipelines/background_cover.py" "$PROJECT_ROOT/pipelines/background_cover.py"

mkdir -p "$PROJECT_ROOT/tools"
cp -f "$PROJECT_ROOT/patchs/files/tools/patch_background_cleaner_v1_1.py" "$PROJECT_ROOT/tools/patch_background_cleaner_v1_1.py"
cp -f "$PROJECT_ROOT/patchs/files/tools/audit_background_purity.py" "$PROJECT_ROOT/tools/audit_background_purity.py"
chmod +x "$PROJECT_ROOT/tools/patch_background_cleaner_v1_1.py" "$PROJECT_ROOT/tools/audit_background_purity.py"

python3 "$PROJECT_ROOT/tools/patch_background_cleaner_v1_1.py" "$PROJECT_ROOT/pipelines/background_cleaner.py"

PY="$PROJECT_ROOT/.docs-parser/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

"$PY" -m compileall -q pipelines/background_cover.py pipelines/background_cleaner.py tools/audit_background_purity.py

echo
echo "==> Patch Background Purity v1.1 appliqué."
echo
echo "Contrat corrigé : cleanbg = substrat visuel sans texte/formule/code."
echo "  - tout texte effacé, traduisible ou non"
echo "  - page numbers / exact text / labels / captions effacés"
echo "  - formules / équations / code effacés"
echo "  - images / diagrammes / formes non textuelles conservés"
echo
echo "Relance :"
echo "  .docs-parser/bin/python -m pytest -q tests/pagereconstruct tests/pagetranslate tests/pageprint tests/pubready"
echo "  bash cmd_bash/run_vsense_studio.sh"
echo "  .docs-parser/bin/python tools/audit_background_purity.py results/<nouveau_demo_studio>/"
echo
echo "Backups : $BACKUP_DIR"
