#!/usr/bin/env bash
set -Eeuo pipefail

# Crée à la racine du projet docs_parser :
#   - Demo_Studio.desktop       : lanceur cliquable Linux
#   - LANCER_DEMO_STUDIO.sh     : lien symbolique de secours vers cmd_bash/run_demo_studio.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT_ROOT/cmd_bash/run_demo_studio.sh"
DESKTOP_FILE="$PROJECT_ROOT/Demo_Studio.desktop"
SYMLINK="$PROJECT_ROOT/LANCER_DEMO_STUDIO.sh"

if [[ ! -f "$RUNNER" ]]; then
  echo "ERREUR: lanceur absent: $RUNNER" >&2
  echo "Installe d'abord Demo Studio Flutter." >&2
  exit 2
fi

chmod +x "$RUNNER"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Docs Parser Demo Studio
Comment=Lancer l'application locale Demo Studio Flutter
Exec=bash -lc 'cd "$PROJECT_ROOT" && bash "$RUNNER"'
Path=$PROJECT_ROOT
Terminal=false
Categories=Development;Utility;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"

# Marquer comme fiable dans GNOME/Nautilus quand possible.
if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

# Lien symbolique de secours.
ln -sfn "cmd_bash/run_demo_studio.sh" "$SYMLINK"
chmod +x "$SYMLINK" 2>/dev/null || true

echo "Créé : $DESKTOP_FILE"
echo "Créé : $SYMLINK"
echo
echo "Utilisation :"
echo "  - Double-clique sur Demo_Studio.desktop"
echo "  - ou double-clique sur LANCER_DEMO_STUDIO.sh si ton gestionnaire l'exécute"
echo
echo "Si GNOME affiche un avertissement, clic droit sur Demo_Studio.desktop puis 'Autoriser le lancement'."
