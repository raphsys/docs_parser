"""PAGEPRINT Quality Auditor — évalue confiance et risques.

Produit : score de confiance, risques, fragments suspects, unités faibles,
zones à revoir, vecteur de confiance par unité et risques aval
(downstream_risks) — l'extraction prépare déjà la reconstruction.
"""

from __future__ import annotations

from .normalizer import clamp_confidence

WEAK_UNIT_CONFIDENCE_THRESHOLD = 0.55
MANUAL_REVIEW_SEVERITY_THRESHOLD = 0.85


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def assess_unit_confidence(unit: dict) -> dict:
    """Remplit le vecteur de confiance d'une unité."""
    extraction = unit.get("extraction") or {}
    visual = unit.get("visual") or {}
    understanding = unit.get("understanding") or {}

    text_conf = clamp_confidence(extraction.get("confidence"))
    if text_conf is None:
        text_conf = 0.97 if "native" in str(extraction.get("source") or "") else 0.7
    bbox_conf = 0.95 if (unit.get("geometry") or {}).get("bbox") else 0.0
    style_conf = clamp_confidence(visual.get("style_confidence"))
    if style_conf is None:
        style_conf = 0.8 if (visual.get("style") or {}).get("font_size_pt") else 0.5
    role_conf = 0.85 if understanding.get("role") else 0.5
    reading_conf = 0.9 if (unit.get("geometry") or {}).get("reading_order_index") is not None else 0.5
    memberships = understanding.get("region_memberships") or []
    region_conf = max((float(m.get("overlap_ratio") or 0.0) for m in memberships), default=0.5)
    policy_conf = 0.9 if (unit.get("policy") or {}).get("translation_strategy") else 0.5

    vector = {
        "text": round(text_conf, 3),
        "bbox": round(bbox_conf, 3),
        "style": round(style_conf, 3),
        "role": round(role_conf, 3),
        "reading_order": round(reading_conf, 3),
        "region_membership": round(region_conf, 3),
        "translation_policy": round(policy_conf, 3),
    }
    vector["overall"] = round(sum(vector.values()) / len(vector), 3)
    unit["confidence"] = vector
    return vector


def assess(units: list[dict], regions: list[dict],
           page_structure: dict | None = None,
           page_intelligence: dict | None = None) -> tuple[dict, dict]:
    """Évalue la qualité globale. Retourne (quality, risks)."""
    structure = page_structure or {}
    intelligence = (
        page_intelligence
        or structure.get("page_intelligence")
        or {}
    )
    if not intelligence.get("risk_flags"):
        intelligence = dict(intelligence)
        intelligence["risk_flags"] = (
            (structure.get("page_case_v2") or {}).get("risk_flags")
            or structure.get("risk_flags")
            or []
        )

    overall_values: list[float] = []
    weak_units: list[str] = []
    native_count = 0
    ocr_confidences: list[float] = []
    style_confidences: list[float] = []

    for unit in units:
        vector = assess_unit_confidence(unit)
        overall_values.append(vector["overall"])
        style_confidences.append(vector["style"])
        if vector["overall"] < WEAK_UNIT_CONFIDENCE_THRESHOLD:
            weak_units.append(unit["unit_id"])
        source = str((unit.get("extraction") or {}).get("source") or "")
        if "native" in source:
            native_count += 1
        elif "ocr" in source:
            conf = clamp_confidence((unit.get("extraction") or {}).get("confidence"))
            if conf is not None:
                ocr_confidences.append(conf)

    total = max(1, len(units))
    risk_flags = []
    for flag in intelligence.get("risk_flags") or structure.get("risk_flags") or []:
        if isinstance(flag, dict):
            risk_flags.append({
                "code": flag.get("code"),
                "severity": clamp_confidence(flag.get("severity")) or 0.5,
                "affected_units": flag.get("affected_units") or [],
            })
        elif isinstance(flag, str):
            risk_flags.append({"code": flag, "severity": 0.5, "affected_units": []})

    sensitivity = intelligence.get("translation_sensitivity") or {}
    downstream_risks = {
        "translation_overflow": clamp_confidence(sensitivity.get("overflow_risk")) or 0.0,
        "anchor_drift": clamp_confidence(sensitivity.get("anchoring_sensitivity")) or 0.0,
        "grid_alignment_loss": clamp_confidence(sensitivity.get("grid_alignment_sensitivity")) or 0.0,
        "toc_fragmentation": 0.0,
        "manual_review_required": any(
            (f.get("severity") or 0.0) >= MANUAL_REVIEW_SEVERITY_THRESHOLD
            for f in risk_flags
        ),
    }

    quality = {
        "overall_confidence": _mean(overall_values) or 0.0,
        "extraction_quality": {
            "native_text_coverage": round(native_count / total, 4),
            "ocr_confidence_mean": _mean(ocr_confidences),
            "block_fragmentation_score": None,
            "style_confidence_mean": _mean(style_confidences),
        },
        "layout_quality": {
            "classification_confidence": intelligence.get("classification_confidence") or {},
            "region_detection_confidence": _mean(
                [clamp_confidence(r.get("confidence")) for r in regions]
            ),
            "reading_order_confidence": 0.9 if units else 0.0,
        },
        "weak_units": weak_units,
        "risks": risk_flags,
        "manual_review_required": downstream_risks["manual_review_required"],
    }
    risks = {
        "risk_flags": risk_flags,
        "downstream_risks": downstream_risks,
    }
    return quality, risks
