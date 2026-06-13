"""Resolve competing PAGEPRINT evidence claims into unit understanding."""

from __future__ import annotations

from collections import defaultdict

from pageprint.normalizer import clamp_confidence


STRONG_TEXT_ROLES = {
    "body_paragraph",
    "title",
    "subtitle",
    "section_heading",
    "subsection_heading",
    "figure_caption",
    "table_caption",
    "toc_entry",
    "toc_entry_title",
    "index_entry",
    "index_head_term",
    "table_body_cell",
    "table_header_cell",
}

CLAIM_TO_OBJECT = {
    "natural_text": ("natural_text", "text"),
    "formula_candidate": ("formula_expression", "formula"),
    "formula_confirmed": ("formula_expression", "formula"),
    "code_candidate": ("code_line", "code"),
    "code_confirmed": ("code_line", "code"),
    "table_candidate": ("table_cell", "table"),
    "table_confirmed": ("table_cell", "table"),
    "toc_candidate": ("toc_entry", "toc"),
    "toc_confirmed": ("toc_entry", "toc"),
    "index_candidate": ("index_entry", "index"),
    "index_confirmed": ("index_entry", "index"),
    "caption_candidate": ("caption", "caption"),
    "caption_confirmed": ("caption", "caption"),
    "publisher_mark_candidate": ("publisher_mark", "artifact"),
    "publisher_mark_confirmed": ("publisher_mark", "artifact"),
    "watermark": ("watermark", "artifact"),
    "command_name": ("command_name", "code_token"),
    "file_path": ("path", "protected_token"),
    "url": ("url", "protected_token"),
    "email": ("email", "protected_token"),
    "acronym": ("acronym", "protected_token"),
    "proper_name": ("proper_name", "protected_token"),
    "page_reference": ("page_reference", "reference"),
    "section_number": ("section_number", "reference"),
}


def resolve_unit_evidence(unit: dict) -> dict:
    """Resolve claims for one unit and attach ``resolved_understanding``.

    A region candidate can influence a unit, but weak/partial region evidence
    cannot override strong natural text, TOC/index/table/caption roles, or a
    long text extraction.
    """
    claims = list((unit.get("evidence") or {}).get("claims") or [])
    extraction = unit.get("extraction") or {}
    understanding = unit.get("understanding") or {}
    text = str((unit.get("content") or {}).get("text") or "").strip()
    role = understanding.get("role")

    if not claims:
        claims = [{
            "source": extraction.get("source") or "heuristic",
            "claim_type": "natural_text" if text else "empty",
            "confidence": extraction.get("confidence") or 0.6,
            "reason": "fallback_extraction_default",
            "value": text,
        }]

    scored = []
    for claim in claims:
        claim_type = claim.get("claim_type") or "unknown"
        score = float(clamp_confidence(claim.get("confidence")) or 0.0)
        evidence = claim.get("evidence") or {}
        coverage_mode = evidence.get("coverage_mode")
        if claim_type.endswith("_candidate") and coverage_mode in {"incidental_overlap", "partial_inline"}:
            score *= 0.55
        if claim_type in {"formula_candidate", "code_candidate"} and _has_strong_natural_text(unit, text, role):
            score *= 0.45
        if claim_type == "natural_text" and _has_strong_natural_text(unit, text, role):
            score = max(score, 0.88)
        if claim_type in {"toc_candidate", "index_candidate", "caption_candidate"} and role:
            score = max(score, 0.84)
        scored.append((score, claim))

    scored.sort(key=lambda item: item[0], reverse=True)
    winning_score, winning_claim = scored[0]
    rejected = [claim for _, claim in scored[1:]]
    object_type, semantic_kind = CLAIM_TO_OBJECT.get(
        winning_claim.get("claim_type"),
        (understanding.get("object_type") or "natural_text", understanding.get("semantic_kind") or "text"),
    )

    resolved = {
        "role": role or "unknown",
        "object_type": object_type,
        "semantic_kind": semantic_kind,
        "confidence": round(winning_score, 3),
        "reason": f"winning_claim:{winning_claim.get('claim_type')}",
        "winning_claims": [winning_claim],
        "rejected_claims": rejected,
    }
    unit.setdefault("evidence", {})["resolved_understanding"] = resolved
    unit["evidence"]["sources"] = [
        {
            "source": claim.get("source"),
            "claim": claim.get("claim_type"),
            "confidence": claim.get("confidence"),
        }
        for claim in claims
    ]
    unit["evidence"]["resolved_as"] = object_type
    unit["evidence"]["resolution_rule"] = resolved["reason"]
    unit["evidence"]["confidence"] = resolved["confidence"]
    return resolved


def resolve_all(units: list[dict]) -> list[dict]:
    """Resolve all units in-place and return decision traces."""
    decisions = []
    for unit in units:
        resolved = resolve_unit_evidence(unit)
        if resolved.get("confidence", 0.0) < 0.65 or resolved.get("role") == "unknown":
            decision = {
                "stage": "evidence_resolution",
                "target_id": unit.get("unit_id"),
                "decision": f"resolved_as={resolved.get('object_type')}",
                "reason": resolved.get("reason"),
                "confidence": resolved.get("confidence"),
            }
            unit.setdefault("provenance", {}).setdefault("decision_trace", []).append(decision)
            decisions.append(decision)
    return decisions


def summarize_claims_by_type(claims: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for claim in claims or []:
        counts[str(claim.get("claim_type") or "unknown")] += 1
    return dict(counts)


def _has_strong_natural_text(unit: dict, text: str, role: str | None) -> bool:
    words = [w for w in text.split() if any(ch.isalpha() for ch in w)]
    if role in STRONG_TEXT_ROLES:
        return True
    if len(words) >= 6:
        return True
    return bool(unit.get("level") in {"phrase", "line", "block"} and len(words) >= 4)
