"""Exports (CSV/JSON/Markdown) et rapport d'audit (§13)."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict

from .models import RuleRecord

_CSV_COLS = [
    "id",
    "pipeline_unit",
    "secondary_units",
    "rule_type",
    "source_type",
    "title",
    "file_path",
    "line_start",
    "line_end",
    "symbol_name",
    "natural_language_title",
    "natural_language_rule",
    "objective",
    "understanding",
    "impact",
    "risk_explanation",
    "human_decision",
    "recommended_decision",
    "proposed_modification",
    "keep_decision",
    "modification_decision",
    "lifecycle_status",
    "status",
    "usage_status",
    "validation_status",
    "ai_interpreted",
    "confidence",
    "risk_level",
    "blocking",
    "managed",
    "tests_linked",
]


def to_csv(records: list[RuleRecord]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        row = r.to_row()
        writer.writerow({k: row.get(k, "") for k in _CSV_COLS})
    return buf.getvalue()


def to_json(records: list[RuleRecord]) -> str:
    return json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2)


def to_markdown(records: list[RuleRecord]) -> str:
    cols = ["id", "pipeline_unit", "rule_type", "natural_language_title",
            "natural_language_rule", "recommended_decision", "lifecycle_status",
            "status", "confidence"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in records:
        row = r.to_row()
        cells = [str(row.get(c, "")).replace("|", "\\|")[:80] for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def audit_report(records: list[RuleRecord]) -> str:
    """Génère `RULE_AUDIT_REPORT.md` (14 sections du §13)."""
    n = len(records)
    by_unit: dict[str, int] = Counter(r.pipeline_unit for r in records)
    by_status: dict[str, list[RuleRecord]] = defaultdict(list)
    for r in records:
        by_status[r.status].append(r)

    active = by_status.get("active", [])
    candidate = by_status.get("candidate", [])
    conflicting = by_status.get("conflicting", [])
    duplicate = by_status.get("duplicate", [])

    orphans = [r for r in records if r.usage_status == "not_referenced"]
    untested = [r for r in records if not r.tests_linked and r.source_type != "documentation"]
    documented = [r for r in records if r.source_type == "documentation"]
    code_rules = [r for r in records if r.source_type in {"python_ast", "comment_block"}]

    # documenté mais non implémenté : règle doc dont aucun code ne partage l'unité+type
    impl_keys = {(r.pipeline_unit, r.rule_type) for r in code_rules}
    doc_not_impl = [r for r in documented if (r.pipeline_unit, r.rule_type) not in impl_keys]
    # implémenté mais non documenté : règle code dont aucune doc ne partage l'unité
    doc_units = {r.pipeline_unit for r in documented}
    impl_not_doc = [r for r in code_rules if r.pipeline_unit not in doc_units]

    keep = [r for r in records if r.keep_decision == "keep"]
    modify = [r for r in records if r.keep_decision == "modify"]
    delete = [r for r in records if r.keep_decision == "delete"]

    def _bullet(rs: list[RuleRecord], limit: int = 25) -> str:
        if not rs:
            return "_(aucune)_\n"
        out = "\n".join(
            f"- `{r.id}` [{r.pipeline_unit}] {r.title[:90]} — {r.file_path}:{r.line_start or ''}"
            for r in rs[:limit]
        )
        if len(rs) > limit:
            out += f"\n- … et {len(rs) - limit} autres"
        return out + "\n"

    md: list[str] = []
    md.append("# Rapport d'audit des règles — Rule Studio\n")
    md.append("## 1. Nombre total de règles détectées\n")
    md.append(f"**{n}** règles au total.\n")

    md.append("## 2. Règles par unité de pipeline\n")
    for unit, c in sorted(by_unit.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- **{unit}** : {c}")
    md.append("")

    md.append("## 3. Règles actives\n")
    md.append(f"{len(active)} actives.\n" + _bullet(active))
    md.append("## 4. Règles candidates\n")
    md.append(f"{len(candidate)} candidates.\n" + _bullet(candidate))
    md.append("## 5. Règles orphelines (non référencées)\n")
    md.append(f"{len(orphans)} orphelines.\n" + _bullet(orphans))
    md.append("## 6. Règles contradictoires\n")
    md.append(_bullet(conflicting))
    md.append("## 7. Règles dupliquées\n")
    md.append(_bullet(duplicate))
    md.append("## 8. Règles non testées\n")
    md.append(f"{len(untested)} sans test lié.\n" + _bullet(untested))
    md.append("## 9. Règles documentées mais non implémentées\n")
    md.append(_bullet(doc_not_impl))
    md.append("## 10. Règles implémentées mais non documentées\n")
    md.append(_bullet(impl_not_doc))
    md.append("## 11. Règles à garder\n")
    md.append(f"{len(keep)} marquées keep.\n" + _bullet(keep))
    md.append("## 12. Règles à modifier\n")
    md.append(f"{len(modify)} marquées modify.\n" + _bullet(modify))
    md.append("## 13. Règles à supprimer\n")
    md.append(f"{len(delete)} marquées delete.\n" + _bullet(delete))

    md.append("## 14. Recommandations\n")
    recos = []
    if orphans:
        recos.append(f"- Examiner les {len(orphans)} règles orphelines : code mort potentiel.")
    if untested:
        recos.append(f"- Ajouter des tests pour les {len(untested)} règles non couvertes.")
    if doc_not_impl:
        recos.append(f"- {len(doc_not_impl)} règles documentées sans implémentation détectée : à implémenter ou archiver.")
    if impl_not_doc:
        recos.append(f"- {len(impl_not_doc)} règles codées sans documentation : à documenter.")
    if candidate:
        recos.append(f"- Trier les {len(candidate)} candidates : promouvoir en gérées ou rejeter.")
    if not recos:
        recos.append("- Inventaire sain : poursuivre la promotion des candidates en règles gérées.")
    md.append("\n".join(recos) + "\n")

    return "\n".join(md)
