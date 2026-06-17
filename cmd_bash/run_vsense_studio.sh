#!/usr/bin/env bash
set -Eeuo pipefail

# Lance vSense Studio avec l'environnement Python officiel du projet.
#
# Environnement Python OBLIGATOIRE par défaut :
#   .docs-parser/bin/python
#
# À lancer depuis n'importe où :
#   bash ~/Mes_Projets/docs_parser/cmd_bash/run_vsense_studio.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$ROOT/apps/demo_studio_flutter"
PROJECT_PYTHON="$ROOT/.docs-parser/bin/python"

if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "ERREUR: environnement Python du projet introuvable ou non exécutable." >&2
  echo "Attendu exactement : $PROJECT_PYTHON" >&2
  echo >&2
  echo "Vérifie avec :" >&2
  echo "  ls -l \"$PROJECT_PYTHON\"" >&2
  exit 2
fi

PYTHON_BIN="$PROJECT_PYTHON"

if ! command -v flutter >/dev/null 2>&1; then
  echo "ERREUR: Flutter n'est pas installé ou n'est pas dans le PATH." >&2
  echo "Pré-requis Debian: sudo apt install clang cmake ninja-build pkg-config libgtk-3-dev" >&2
  echo "Puis installe Flutter Desktop Linux et vérifie: flutter doctor" >&2
  exit 2
fi

if [[ ! -f "$ROOT/tools/demo_studio_backend.py" ]]; then
  echo "ERREUR: backend absent: $ROOT/tools/demo_studio_backend.py" >&2
  echo "Réapplique le lot vSense Studio." >&2
  exit 2
fi

echo "==> Application : vSense Studio"
echo "==> Projet      : $ROOT"
echo "==> Python      : $PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import sys
print("==> Python exe  :", sys.executable)
try:
    import fitz
    print("==> PyMuPDF     : OK")
except Exception as exc:
    print("==> PyMuPDF     : INDISPONIBLE:", exc)
    raise SystemExit(2)
PY

mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Les fichiers Dart sont livrés par les patchs. Flutter doit seulement générer
# l'enveloppe desktop linux/ si elle n'existe pas encore.
if [[ ! -d "linux" ]]; then
  echo "==> Initialisation Flutter Desktop Linux..."
  flutter create --platforms=linux --project-name vsense_studio .
fi

echo "==> Récupération des dépendances Flutter..."
flutter pub get

echo "==> Lancement vSense Studio..."
flutter run -d linux \
  --dart-define=DOCS_PARSER_ROOT="$ROOT" \
  --dart-define=DOCS_PARSER_PYTHON="$PYTHON_BIN" \
  --dart-define=VSENSE_APP_NAME="vSense Studio"
