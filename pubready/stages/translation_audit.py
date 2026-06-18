"""Audit traduction GRANULAIRE par bloc: présence, non-vide, troncature,
tokens protégés conservés, code/formule non traduits."""

from __future__ import annotations

import re
from ..schema import StageAuditResult, ElementAudit, DimensionScore, Finding, OK, REVIEW, KO

_PRESERVE_ROLES = {"formula_expression", "code_line", "code_block", "command_name", "path"}
# Tokens à préserver: acronymes TOUT-MAJUSCULE (≥2, ex: SQL, API, HTTP) + nombres.
# PAS les mots simplement capitalisés en début de phrase (The, When, Data...) ni
# les noms propres mixtes (Postgres) — sinon faux positifs massifs en traduction.
_TOKEN_RE = re.compile(r"\b[A-Z]{2,6}[0-9]*\b|\b\d+(?:[.,]\d+)?\b")


def _dim_presence(src, tr, role) -> DimensionScore:
    if role in _PRESERVE_ROLES:
        return DimensionScore("translation", 1.0, "preserve", "preserve", OK, 1.0)
    if src and not tr:
        return DimensionScore("translation", 0.0, "non-empty", "empty", KO, 1.5, "empty_translation")
    if src and tr == src and len(src.split()) > 3:
        return DimensionScore("translation", 0.5, "translated", "unchanged", REVIEW, 1.5, "unchanged_suspect")
    # troncature: traduction beaucoup plus courte qu'attendu
    if src and tr and len(src) > 40 and len(tr) < 0.4 * len(src):
        return DimensionScore("translation", 0.3, f"~{len(src)}c", f"{len(tr)}c", KO, 1.5, "translation_truncated")
    return DimensionScore("translation", 1.0, "ok", "ok", OK, 1.5)


def _dim_protected_tokens(src, tr, role) -> DimensionScore:
    if role in _PRESERVE_ROLES or not src or not tr:
        return DimensionScore("protected_tokens", 1.0, weight=0.8)
    # nombres / acronymes présents dans la source doivent rester dans la trad.
    src_tok = set(_TOKEN_RE.findall(src))
    if not src_tok:
        return DimensionScore("protected_tokens", 1.0, weight=0.8)
    kept = sum(1 for t in src_tok if t in tr)
    ratio = kept / len(src_tok)
    return DimensionScore("protected_tokens", round(ratio, 3), len(src_tok), kept,
                          OK if ratio >= 0.85 else REVIEW if ratio >= 0.6 else KO, 0.8,
                          None if ratio >= 0.85 else "protected_token_changed")


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    res = StageAuditResult(stage_name="pagetranslate")
    blocks = (plan.get("layers") or {}).get("translated_text") or []
    if not blocks:
        res.score = 1.0
        return res
    auds = []
    for b in blocks:
        role = b.get("role") or ""
        src = (b.get("source_text") or "").strip()
        tr = (b.get("translated_text") or "").strip()
        dims = [_dim_presence(src, tr, role), _dim_protected_tokens(src, tr, role)]
        el = ElementAudit(element_id=b.get("id"), level="block", role=role,
                          source_text=src, translated_text=tr, dimensions=dims)
        el.combine(); auds.append(el)
    res.elements = auds
    res.score = round(sum(a.score for a in auds) / len(auds), 3)
    res.status = KO if any(a.status == KO for a in auds) else (REVIEW if res.score < 0.98 else OK)
    for a in auds:
        for d in a.dimensions:
            if d.status == KO and d.finding:
                hb = {"empty_translation": "missing_translatable_text",
                      "translation_truncated": "translation_truncated",
                      "protected_token_changed": "protected_token_changed"}.get(d.finding)
                if hb and hb not in res.hard_blockers:
                    res.hard_blockers.append(hb)
                res.findings.append(Finding(type=d.finding, severity=KO, element_id=a.element_id))
    return res
