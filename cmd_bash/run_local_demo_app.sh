#!/usr/bin/env bash
set -Eeuo pipefail
# Script situé dans cmd_bash/ ; se positionne tout seul à la racine du projet.
#   bash cmd_bash/run_local_demo_app.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 tools/local_demo_app.py
