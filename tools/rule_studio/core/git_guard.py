"""Garde-fous Git avant toute modification du dépôt (§12).

Aucune écriture silencieuse : on vérifie que le dépôt est sous Git, on inspecte
son état, et on fournit branche de sauvegarde + rollback. Le patcher s'appuie
dessus.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitState:
    is_repo: bool
    branch: str = ""
    dirty: bool = False
    untracked: list[str] = None  # type: ignore[assignment]
    modified: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.untracked = self.untracked or []
        self.modified = self.modified or []


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=str(cwd), capture_output=True, text=True, check=False
    )


def inspect(root: str | Path) -> GitState:
    root = Path(root)
    head = _run(["git", "rev-parse", "--is-inside-work-tree"], root)
    if head.returncode != 0 or head.stdout.strip() != "true":
        return GitState(is_repo=False)

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip()
    status = _run(["git", "status", "--porcelain"], root).stdout.splitlines()
    untracked, modified = [], []
    for line in status:
        code, _, name = line[:2], line[2:3], line[3:]
        if code.strip() == "??":
            untracked.append(name)
        else:
            modified.append(name)
    return GitState(
        is_repo=True,
        branch=branch,
        dirty=bool(status),
        untracked=untracked,
        modified=modified,
    )


def create_backup_branch(root: str | Path, prefix: str = "rule-studio-backup") -> str:
    """Crée une branche de sauvegarde pointant sur HEAD. Renvoie son nom."""
    root = Path(root)
    name = f"{prefix}/{time.strftime('%Y%m%d-%H%M%S')}"
    res = _run(["git", "branch", name], root)
    if res.returncode != 0:
        raise RuntimeError(f"Échec création branche backup : {res.stderr.strip()}")
    return name


def stash_snapshot(root: str | Path, label: str = "rule-studio") -> str | None:
    """Crée un stash horodaté (sans toucher l'index courant). Renvoie le ref."""
    root = Path(root)
    res = _run(
        ["git", "stash", "push", "--include-untracked", "-m", f"{label}-{time.time():.0f}"],
        root,
    )
    if res.returncode != 0 or "No local changes" in res.stdout:
        return None
    return _run(["git", "rev-parse", "stash@{0}"], root).stdout.strip() or None


def can_safely_patch(state: GitState, target_file: str) -> tuple[bool, str]:
    """Refuse d'écraser un fichier non suivi (§12.3)."""
    if not state.is_repo:
        return False, "Dépôt non Git : modification refusée par sécurité."
    if target_file in state.untracked:
        return (
            False,
            f"`{target_file}` est non suivi par Git : "
            "confirmation explicite requise avant écrasement.",
        )
    return True, "OK"
