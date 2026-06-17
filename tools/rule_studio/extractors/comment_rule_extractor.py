"""Extracteur de blocs de commentaires structurés `# RULE-... / # END-RULE`.

Format standard (§6.4) :

    # RULE-ID: PP-COVER-001
    # RULE-UNIT: PAGE_CLASSIFICATION
    # RULE-TYPE: classification
    # RULE-OBJECTIVE: Détecter les couvertures visuelles.
    # RULE-STATUS: active
    # RULE-MANAGED: true
    # RULE-TEST: tests/test_cover_page.py
    if page_index == 0 and visual_density > 0.7 and text_density < 0.2:
        page_role = "cover"
    # END-RULE

Ces blocs sont la passerelle entre code et règles gérées : ils portent un ID
explicite et peuvent être marqués `managed`.
"""

from __future__ import annotations

import re

from ..core.models import RuleRecord, make_rule_id

_HEADER = re.compile(r"^\s*#\s*RULE-([A-Z]+)\s*:\s*(.*)$")
_END = re.compile(r"^\s*#\s*END-RULE\s*$")
_START = re.compile(r"^\s*#\s*RULE-ID\s*:\s*(.+)$")


def extract_comment_rules(source: str, file_path: str) -> list[RuleRecord]:
    lines = source.splitlines()
    out: list[RuleRecord] = []
    i = 0
    n = len(lines)
    while i < n:
        if _START.match(lines[i]):
            meta: dict[str, str] = {}
            code_lines: list[str] = []
            start_line = i + 1
            j = i
            while j < n and not _END.match(lines[j]):
                m = _HEADER.match(lines[j])
                if m:
                    meta[m.group(1).upper()] = m.group(2).strip()
                elif not lines[j].lstrip().startswith("#"):
                    code_lines.append(lines[j])
                j += 1
            end_line = j + 1 if j < n else start_line + len(code_lines)

            rid = meta.get("ID") or make_rule_id("CMT", f"{file_path}:{start_line}")
            managed = meta.get("MANAGED", "").lower() in {"true", "1", "yes"}
            rec = RuleRecord(
                id=rid,
                title=meta.get("OBJECTIVE", rid)[:120] or rid,
                description=meta.get("OBJECTIVE", ""),
                pipeline_unit=meta.get("UNIT", "UNKNOWN").upper(),
                rule_type=meta.get("TYPE", "unknown").lower(),
                source_type="comment_block",
                file_path=file_path,
                line_start=start_line,
                line_end=end_line,
                raw_code="\n".join(code_lines)[:2000] or None,
                normalized_expression=(code_lines[0].strip() if code_lines else None),
                objective=meta.get("OBJECTIVE", ""),
                status=meta.get("STATUS", "active").lower(),
                usage_status="unknown",
                validation_status="needs_review",
                confidence=0.95,  # déclaration explicite = haute confiance
                risk_level="medium",
                managed=managed,
                editable=True,
                deletable=True,
                simulable=bool(code_lines),
                tests_linked=[meta["TEST"]] if "TEST" in meta else [],
            )
            rec.last_seen_hash = rec.compute_hash()
            rec.touch()
            out.append(rec)
            i = j + 1
        else:
            i += 1
    return out
