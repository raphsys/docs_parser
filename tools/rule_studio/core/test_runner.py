"""Lancement ciblé de la suite de tests (§12.6).

Wrapper léger autour de pytest. Permet de relancer les tests liés à une règle
après application d'un patch, et de retourner un résumé exploitable par l'UI.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestRun:
    ok: bool
    returncode: int
    summary: str
    output: str


def run_tests(root: str | Path, targets: list[str] | None = None, timeout: int = 600) -> TestRun:
    """Exécute pytest sur `targets` (ou toute la suite si None)."""
    root = Path(root)
    cmd = ["python", "-m", "pytest", "-q"]
    if targets:
        cmd += [t for t in targets if (root / t).exists() or "::" in t]
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return TestRun(False, -1, "Timeout", "pytest a dépassé le délai imparti.")
    except FileNotFoundError:
        return TestRun(False, -1, "pytest introuvable", "Installer pytest.")

    out = proc.stdout + "\n" + proc.stderr
    # Dernière ligne non vide = résumé pytest habituel.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    summary = lines[-1] if lines else "(aucune sortie)"
    return TestRun(proc.returncode == 0, proc.returncode, summary, out[-8000:])
