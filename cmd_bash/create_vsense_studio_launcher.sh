#!/usr/bin/env bash
set -Eeuo pipefail

# Crée les lanceurs locaux de vSense Studio.
#
# Fichiers créés à la racine du projet :
#   - vSense_Studio.desktop
#   - LANCER_vSense_Studio.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT_ROOT/cmd_bash/run_vsense_studio.sh"
DESKTOP_FILE="$PROJECT_ROOT/vSense_Studio.desktop"
SYMLINK="$PROJECT_ROOT/LANCER_vSense_Studio.sh"
APP_DESKTOP="$HOME/.local/share/applications/vsense-studio.desktop"

if [[ ! -f "$RUNNER" ]]; then
  echo "ERREUR: lanceur absent: $RUNNER" >&2
  exit 2
fi

chmod +x "$RUNNER"
mkdir -p "$HOME/.local/share/applications"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=vSense Studio
Comment=Studio local de diagnostic, traduction et reconstruction WYSIWYG
Exec=bash -lc 'cd "$PROJECT_ROOT" && bash "$RUNNER"'
Path=$PROJECT_ROOT
Terminal=false
Categories=Development;Utility;
StartupNotify=true
EOF

cp "$DESKTOP_FILE" "$APP_DESKTOP"

chmod +x "$DESKTOP_FILE" "$APP_DESKTOP"

if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
  gio set "$APP_DESKTOP" metadata::trusted true 2>/dev/null || true
fi

ln -sfn "cmd_bash/run_vsense_studio.sh" "$SYMLINK"
chmod +x "$SYMLINK" 2>/dev/null || true

update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

# Nettoyage éventuel des anciens noms à la racine, sans supprimer l'historique applicatif utilisateur.
rm -f "$PROJECT_ROOT/Demo_Studio.desktop" "$PROJECT_ROOT/LANCER_DEMO_STUDIO.sh" 2>/dev/null || true

echo "Créé : $DESKTOP_FILE"
echo "Créé : $SYMLINK"
echo "Créé : $APP_DESKTOP"
echo
echo "Tu peux lancer depuis le terminal avec :"
echo "  gtk-launch vsense-studio"
echo
echo "Ou chercher dans le menu Applications :"
echo "  vSense Studio"
