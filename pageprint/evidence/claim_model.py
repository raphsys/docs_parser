"""Small claim DTO helpers for PAGEPRINT evidence."""

from __future__ import annotations

import itertools

from pageprint.normalizer import clamp_confidence


CLAIM_TYPES = {
    "natural_text",
    "formula_candidate",
    "formula_confirmed",
    "code_candidate",
    "code_confirmed",
    "table_candidate",
    "table_confirmed",
    "toc_candidate",
    "toc_confirmed",
    "index_candidate",
    "index_confirmed",
    "caption_candidate",
    "caption_confirmed",
    "publisher_mark_candidate",
    "publisher_mark_confirmed",
    "author_name_candidate",
    "page_reference",
    "section_number",
    "command_name",
    "file_path",
    "url",
    "email",
    "acronym",
    "proper_name",
    "watermark",
}

_COUNTER = itertools.count(1)


def make_claim(
    *,
    source: str,
    target_unit_id: str | None,
    claim_type: str,
    value: str,
    confidence: float,
    reason: str,
    evidence: dict | None = None,
    bbox: list[float] | None = None,
) -> dict:
    """Build a normalized evidence claim.

    Unknown claim types are kept, but marked so the resolver can still expose
    provenance without accepting the value as a strong ontology signal.
    """
    claim_confidence = clamp_confidence(confidence)
    if claim_confidence is None:
        claim_confidence = 0.5
    normalized_type = str(claim_type or "unknown").strip() or "unknown"
    return {
        "claim_id": f"claim_{next(_COUNTER):06d}",
        "source": source or "unknown",
        "target_unit_id": target_unit_id,
        "claim_type": normalized_type,
        "value": value,
        "confidence": round(float(claim_confidence), 3),
        "reason": reason or "unspecified",
        "evidence": dict(evidence or {}),
        "bbox": list(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
        "known_claim_type": normalized_type in CLAIM_TYPES,
    }
