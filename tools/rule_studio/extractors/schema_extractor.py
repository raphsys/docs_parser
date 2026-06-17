"""Extracteur de schémas (Enum / dataclass / Pydantic / TypedDict).

Note d'implémentation : la détection structurelle des schémas est déjà assurée
par `python_ast_extractor` (visit_ClassDef émet des règles `schema`). Ce module
expose une API dédiée pour usage explicite/tests, en réutilisant l'AST.
"""

from __future__ import annotations

from ..core.models import RuleRecord
from .python_ast_extractor import extract_python_rules


def extract_schema_rules(source: str, file_path: str) -> list[RuleRecord]:
    """Renvoie uniquement les règles de type `schema`/`contract` du fichier."""
    return [
        r
        for r in extract_python_rules(source, file_path)
        if r.rule_type in {"contract"} or r.title.startswith("[schema]")
    ]
