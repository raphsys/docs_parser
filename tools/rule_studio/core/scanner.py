"""Scanner de projet : parcourt l'arborescence et route chaque fichier vers
l'extracteur adéquat.

Le scanner ne décide rien sur le contenu des règles : il collecte les chemins,
applique des filtres d'exclusion, puis délègue aux extracteurs. Il agrège enfin
les `RuleRecord` produits.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .models import RuleRecord

# Répertoires jamais scannés (bruit, artefacts, gros volumes)
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "ocr_results",
    "results",
    "converted_pages",
    "uploads",
    "models",
    "ai_models",
    "frontend-expo",
    "revisions",
    # le studio lui-même ne s'auto-scanne pas par défaut
    "rule_studio",
}

PY_EXT = {".py"}
CONFIG_EXT = {".yaml", ".yml", ".json", ".toml", ".ini"}
MD_EXT = {".md"}

# Fichiers trop gros = probablement générés/données, pas des règles lisibles.
MAX_FILE_BYTES = 1_500_000


@dataclass
class ScanResult:
    rules: list[RuleRecord] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def iter_source_files(
    root: str | Path,
    exclude_dirs: set[str] | None = None,
) -> Iterable[Path]:
    """Itère les fichiers candidats (py/config/markdown) sous `root`."""
    root = Path(root)
    excl = DEFAULT_EXCLUDE_DIRS if exclude_dirs is None else exclude_dirs
    wanted = PY_EXT | CONFIG_EXT | MD_EXT
    for dirpath, dirnames, filenames in os.walk(root):
        # élague in-place pour ne pas descendre dans les exclus
        dirnames[:] = [d for d in dirnames if d not in excl and not d.startswith(".")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in wanted:
                yield p


def scan_project(
    root: str | Path,
    exclude_dirs: set[str] | None = None,
    include_documentation: bool = True,
) -> ScanResult:
    """Scanne `root` et renvoie tous les `RuleRecord` détectés."""
    # imports tardifs pour éviter les cycles
    from ..extractors.python_ast_extractor import extract_python_rules
    from ..extractors.config_extractor import extract_config_rules
    from ..extractors.markdown_extractor import extract_markdown_rules
    from ..extractors.comment_rule_extractor import extract_comment_rules

    root = Path(root).resolve()
    result = ScanResult()

    for path in iter_source_files(root, exclude_dirs):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                result.files_skipped += 1
                continue
            rel = str(path.relative_to(root))
            text = path.read_text(encoding="utf-8", errors="ignore")
            suffix = path.suffix.lower()

            if suffix in PY_EXT:
                result.rules.extend(extract_python_rules(text, rel))
                result.rules.extend(extract_comment_rules(text, rel))
            elif suffix in CONFIG_EXT:
                result.rules.extend(extract_config_rules(text, rel, suffix))
            elif suffix in MD_EXT and include_documentation:
                result.rules.extend(extract_markdown_rules(text, rel))

            result.files_scanned += 1
        except Exception as exc:  # noqa: BLE001 — on agrège, on ne casse pas le scan
            result.errors.append(f"{path}: {exc}")

    return result
