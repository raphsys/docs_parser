"""Terminology support for PAGETRANSLATE."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


DEFAULT_TERMINOLOGY_PATH = Path("ai_models/translation/terminology_en_fr.json")

DEFAULT_TERMINOLOGY = {
    "terms": {
        "MLP": {"policy": "preserve"},
        "CNN": {"policy": "preserve"},
        "ReLU": {"policy": "preserve"},
        "Softmax": {"policy": "preserve"},
        "SQL": {"policy": "preserve"},
        "API": {"policy": "preserve"},
        "OCR": {"policy": "preserve"},
        "precision": {"policy": "preferred_translation", "target": "précision"},
        "recall": {"policy": "preferred_translation", "target": "rappel"},
        "dropout": {"policy": "contextual", "preferred": "dropout", "alternatives": ["abandon"]},
        "pooling": {"policy": "contextual", "preferred": "pooling", "alternatives": ["sous-échantillonnage"]},
        "F-score": {"policy": "contextual", "preferred": "F-score", "alternatives": ["score F"]},
    }
}


def explicit_protected_terms(profile: dict) -> list[str]:
    terminology = profile.get("terminology") or {}
    loaded = load_terminology_table(terminology.get("terms_path"))
    locked = terminology.get("locked_terms") or terminology.get("locked") or []
    protected = profile.get("protected_tokens") or []
    preserve_terms = [term for term, spec in loaded.get("terms", {}).items() if spec.get("policy") == "preserve"]
    return list(dict.fromkeys(str(term) for term in [*protected, *locked, *preserve_terms] if str(term or "").strip()))


def apply_post_translation_glossary(text: str, profile: dict) -> str:
    terminology = profile.get("terminology") or {}
    preferred = _preferred_map(terminology)
    if not preferred:
        return text
    output = text or ""
    for source, target in preferred.items():
        if not source or not target:
            continue
        output = re.sub(re.escape(str(source)), str(target), output, flags=re.IGNORECASE)
    return output


def check_terminology_consistency(source_text: str, translated_text: str, profile: dict) -> dict:
    terminology = profile.get("terminology") or {}
    table = load_terminology_table(terminology.get("terms_path"))
    preferred = _preferred_map(terminology, table)
    reserved = _reserved_terms(terminology, table)
    missing_preferred = []
    for source, target in preferred.items():
        if source and target and re.search(re.escape(str(source)), source_text or "", flags=re.IGNORECASE):
            if not re.search(re.escape(str(target)), translated_text or "", flags=re.IGNORECASE):
                missing_preferred.append(str(target))
    reserved_hits = []
    for term in reserved:
        if term and re.search(re.escape(str(term)), translated_text or "", flags=re.IGNORECASE):
            reserved_hits.append(str(term))
    return {
        "missing_preferred_terms": missing_preferred,
        "reserved_term_hits": reserved_hits,
        "terminology_issue": bool(missing_preferred or reserved_hits),
    }


def load_terminology_table(path: str | None = None) -> dict:
    candidate = Path(path or os.getenv("TRANSLATION_TERMINOLOGY_PATH") or DEFAULT_TERMINOLOGY_PATH)
    if not candidate.is_file():
        return DEFAULT_TERMINOLOGY
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_TERMINOLOGY
    if not isinstance(payload, dict) or not isinstance(payload.get("terms"), dict):
        return DEFAULT_TERMINOLOGY
    normalized = {}
    for term, spec in payload["terms"].items():
        if not isinstance(spec, dict):
            continue
        normalized[str(term)] = {
            "policy": str(spec.get("policy") or "preserve"),
            "target": spec.get("target"),
            "preferred": spec.get("preferred"),
            "alternatives": spec.get("alternatives") or [],
        }
    return {"terms": normalized}


def _preferred_map(terminology: dict, table: dict | None = None) -> dict[str, str]:
    preferred = {}
    table = table or load_terminology_table(terminology.get("terms_path"))
    for term, spec in (table.get("terms") or {}).items():
        policy = spec.get("policy")
        if policy == "preferred_translation" and spec.get("target"):
            preferred[str(term)] = str(spec.get("target"))
        elif policy == "contextual" and spec.get("preferred"):
            preferred[str(term)] = str(spec.get("preferred"))
    legacy_preferred = terminology.get("preferred_terms") or terminology.get("preferred") or {}
    if isinstance(legacy_preferred, dict):
        for source, target in legacy_preferred.items():
            if source and target:
                preferred[str(source)] = str(target)
    return preferred


def _reserved_terms(terminology: dict, table: dict | None = None) -> list[str]:
    reserved = []
    table = table or load_terminology_table(terminology.get("terms_path"))
    for term, spec in (table.get("terms") or {}).items():
        if spec.get("policy") == "preserve":
            reserved.append(str(term))
    legacy_reserved = terminology.get("reserved_terms") or terminology.get("reserved") or []
    if isinstance(legacy_reserved, dict):
        reserved.extend(str(term) for term in legacy_reserved)
    elif isinstance(legacy_reserved, list):
        reserved.extend(str(term) for term in legacy_reserved)
    return list(dict.fromkeys(reserved))
