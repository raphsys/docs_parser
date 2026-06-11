"""PAGEPRINT Evidence Resolver — résout les conflits entre sources.

Plusieurs sources peuvent se contredire :
    native_pdf / ocr / layout_ai / special_region_detector / heuristics /
    postprocessors / LLM-ONNX.

Ce module décide quelle source gagne, dans quel contexte, avec quelle
confiance — et produit une décision traçable pour chaque unité.
Sans cette couche, le pipeline reste fragile.
"""

from __future__ import annotations

from .normalizer import clamp_confidence

SPECIAL_REGION_OVERLAP_RULE = "special_region_over_text_when_overlap_gt_0.65"
SPECIAL_REGION_OVERLAP_THRESHOLD = 0.65

# Régions spéciales qui priment sur la classification textuelle.
SPECIAL_REGION_CLAIMS = {
    "formula": "formula",
    "code": "code",
    "table_cell": "table_cell",
    "chart_tick": "chart_tick_label",
}

DEFAULT_SOURCE_CONFIDENCE = {
    "native_pdf": 0.98,
    "merged": 0.92,
    "ocr": 0.86,
    "layout_ai": 0.75,
    "heuristic": 0.6,
}


def resolve_unit_evidence(unit: dict) -> dict:
    """Construit la couche evidence d'une unité et résout sa nature.

    Résolution :
    1. région spéciale recouvrante (> 0.65) → la région gagne ;
    2. sinon → la source d'extraction la plus fiable gagne.
    """
    extraction = unit.get("extraction") or {}
    understanding = unit.get("understanding") or {}

    source = extraction.get("source") or "heuristic"
    base_source = "native_pdf" if "native" in str(source) else (
        "ocr" if "ocr" in str(source) else str(source)
    )
    source_confidence = clamp_confidence(extraction.get("confidence"))
    if source_confidence is None:
        source_confidence = DEFAULT_SOURCE_CONFIDENCE.get(base_source, 0.6)

    sources = [{
        "source": base_source,
        "claim": understanding.get("object_type") or "text_block",
        "confidence": source_confidence,
    }]

    resolved_as = understanding.get("object_type") or "text_block"
    resolution_rule = "extraction_source_default"
    resolved_confidence = source_confidence

    for membership in understanding.get("region_memberships") or []:
        region_type = membership.get("region_type")
        overlap = float(membership.get("overlap_ratio") or 0.0)
        claim = SPECIAL_REGION_CLAIMS.get(region_type)
        if claim is None:
            continue
        region_confidence = clamp_confidence(membership.get("confidence")) or 0.9
        sources.append({
            "source": "special_region_detector",
            "claim": f"{region_type}_region",
            "confidence": region_confidence,
        })
        if overlap > SPECIAL_REGION_OVERLAP_THRESHOLD:
            resolved_as = claim
            resolution_rule = SPECIAL_REGION_OVERLAP_RULE
            resolved_confidence = region_confidence

    evidence = {
        "sources": sources,
        "resolved_as": resolved_as,
        "resolution_rule": resolution_rule,
        "confidence": resolved_confidence,
    }
    unit["evidence"] = evidence
    return evidence


def resolve_all(units: list[dict]) -> list[dict]:
    """Résout l'évidence de toutes les unités (in place) et journalise la décision."""
    decisions = []
    for unit in units:
        evidence = resolve_unit_evidence(unit)
        if evidence["resolution_rule"] != "extraction_source_default":
            decision = {
                "stage": "evidence_resolution",
                "target_id": unit["unit_id"],
                "decision": f"resolved_as={evidence['resolved_as']}",
                "reason": evidence["resolution_rule"],
                "confidence": evidence["confidence"],
            }
            unit["provenance"]["decision_trace"].append(decision)
            decisions.append(decision)
    return decisions
