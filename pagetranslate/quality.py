"""Translation quality checks for PAGETRANSLATE."""

from __future__ import annotations

import re

from .text_utils import normalize_spaces, word_count
from .terminology import check_terminology_consistency

try:
    from pageprint.text_postprocessors import ends_with_break_hyphen, has_repeated_ngram, looks_like_truncated_fragment
except Exception:  # pragma: no cover - pageprint always available in this repo
    def ends_with_break_hyphen(_text):
        return None

    def has_repeated_ngram(_text, **_kwargs):
        return False

    def looks_like_truncated_fragment(_text):
        return False


NUMBER_RE = re.compile(r"(?<![\w./-])\d+(?:[.,]\d+)?(?![\w./-])")
UNIT_RE = re.compile(r"\b(?:%|kg|g|mg|µg|ug|cm|mm|m|m²|m³|km|km/h|mol/L|mg/dL|mmHg|bpm|UI|MB|GB|GHz|Hz|ms|s|h|°C|°F|K|px|pt|FCFA|XOF|USD|EUR)\b", re.IGNORECASE)
SOURCE_EN_MARKERS = re.compile(r"\b(the|and|with|from|this|that|which|shall|will|can|must)\b", re.IGNORECASE)


def unit_quality(source_text: str, translated_text: str, item: dict, profile: dict) -> dict:
    source_words = max(1, word_count(source_text))
    translated_words = word_count(translated_text)
    expansion = round(translated_words / source_words, 3)
    source_numbers = [_normalize_number(n) for n in NUMBER_RE.findall(source_text or "")]
    translated_numbers = [_normalize_number(n) for n in NUMBER_RE.findall(translated_text or "")]
    source_units = [u.lower() for u in UNIT_RE.findall(source_text or "")]
    translated_units = [u.lower() for u in UNIT_RE.findall(translated_text or "")]
    protections = item.get("protections") or []
    missing_protections = [
        token.get("text")
        for token in protections
        if token.get("text") and token.get("text") not in (translated_text or "")
    ]
    placeholders_left = [
        token.get("placeholder")
        for token in protections
        if token.get("placeholder") and token.get("placeholder") in (translated_text or "")
    ]
    overflow_risk = _overflow_risk(source_text, translated_text, item)
    source_leak = _source_language_leak(translated_text, profile)
    empty = not bool(normalize_spaces(translated_text))
    unchanged = normalize_spaces(source_text).lower() == normalize_spaces(translated_text).lower()
    number_mismatch = source_numbers != translated_numbers
    unit_mismatch = source_units != translated_units
    protected_mismatch = bool(missing_protections or placeholders_left)
    unchanged_problem = _unchanged_problem(unchanged, item, profile)
    terminology = check_terminology_consistency(source_text, translated_text, profile)
    repeated_output = has_repeated_ngram(translated_text)
    source_fragment = looks_like_truncated_fragment(source_text)
    dehyphenation_needed = bool(ends_with_break_hyphen(source_text))
    needs_review = any([
        empty,
        unchanged_problem,
        number_mismatch,
        unit_mismatch,
        protected_mismatch,
        source_leak,
        terminology["terminology_issue"],
        overflow_risk == "high",
        repeated_output,
        source_fragment,
    ])
    qa_reasons = []
    if empty:
        qa_reasons.append("empty_translation")
    if dehyphenation_needed:
        qa_reasons.append("dehyphenation_needed")
    if source_fragment:
        qa_reasons.append("source_fragment")
    if repeated_output:
        qa_reasons.append("repeated_output")
    if number_mismatch:
        qa_reasons.append("number_mismatch")
    if unit_mismatch:
        qa_reasons.append("unit_mismatch")
    if protected_mismatch:
        qa_reasons.append("protected_token_mismatch")
    if source_leak:
        qa_reasons.append("source_leak")
    if unchanged_problem:
        qa_reasons.append("unchanged_suspect")
    if terminology["terminology_issue"]:
        qa_reasons.append("terminology_issue")
    if overflow_risk == "high":
        qa_reasons.append("overflow_risk")
    return {
        "source_word_count": source_words,
        "translated_word_count": translated_words,
        "word_expansion_ratio": expansion,
        "char_expansion_ratio": _char_ratio(source_text, translated_text),
        "empty_translation": empty,
        "unchanged": unchanged,
        "unchanged_problem": unchanged_problem,
        "target_lang": profile.get("target_lang"),
        "number_mismatch": number_mismatch,
        "unit_mismatch": unit_mismatch,
        "protected_token_mismatch": protected_mismatch,
        "placeholder_corruption_count": int(protected_mismatch),
        "missing_protected_tokens": missing_protections,
        "unrestored_placeholders": placeholders_left,
        "source_language_leak": source_leak,
        "terminology": terminology,
        "wysiwyg_overflow_risk": overflow_risk,
        "repeated_output": repeated_output,
        "source_fragment": source_fragment,
        "dehyphenation_needed": dehyphenation_needed,
        "qa_reasons": qa_reasons,
        "needs_review": needs_review,
        "checks": {
            "numbers": {"source": source_numbers, "translated": translated_numbers},
            "units": {"source": source_units, "translated": translated_units},
            "protections_count": len(protections),
        },
    }


def assess_translation_quality(translated_units: list[dict]) -> dict:
    total = len(translated_units)
    if not total:
        return {
            "unit_count": 0,
            "translated_count": 0,
            "preserved_count": 0,
            "unchanged_suspect_count": 0,
            "dry_run_count": 0,
            "unchanged_count": 0,
            "empty_count": 0,
            "needs_review_count": 0,
            "protected_token_mismatch_count": 0,
            "number_mismatch_count": 0,
            "terminology_warning_count": 0,
            "placeholder_corruption_count": 0,
            "average_word_expansion_ratio": None,
        }
    ratios = [
        (u.get("quality") or {}).get("word_expansion_ratio")
        for u in translated_units
        if (u.get("quality") or {}).get("word_expansion_ratio") is not None
    ]
    return {
        "unit_count": total,
        "translated_count": sum(1 for u in translated_units if u.get("status") == "translated"),
        "preserved_count": sum(1 for u in translated_units if u.get("status") == "preserved"),
        "unchanged_suspect_count": sum(1 for u in translated_units if u.get("status") == "unchanged_suspect"),
        "dry_run_count": sum(1 for u in translated_units if u.get("status") == "dry_run"),
        "unchanged_count": sum(1 for u in translated_units if (u.get("quality") or {}).get("unchanged")),
        "empty_count": sum(1 for u in translated_units if (u.get("quality") or {}).get("empty_translation")),
        "needs_review_count": sum(1 for u in translated_units if (u.get("quality") or {}).get("needs_review")),
        "number_mismatch_count": sum(1 for u in translated_units if (u.get("quality") or {}).get("number_mismatch")),
        "unit_mismatch_count": sum(1 for u in translated_units if (u.get("quality") or {}).get("unit_mismatch")),
        "protected_token_mismatch_count": sum(1 for u in translated_units if (u.get("quality") or {}).get("protected_token_mismatch")),
        "placeholder_corruption_count": sum(1 for u in translated_units if (u.get("quality") or {}).get("placeholder_corruption_count")),
        "terminology_warning_count": sum(1 for u in translated_units if ((u.get("quality") or {}).get("terminology") or {}).get("terminology_issue")),
        "average_word_expansion_ratio": round(sum(ratios) / len(ratios), 3) if ratios else None,
    }


def _char_ratio(source_text: str, translated_text: str) -> float:
    return round(len(translated_text or "") / max(1, len(source_text or "")), 3)


def _normalize_number(value: str) -> str:
    s = str(value or "").strip().replace(" ", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    return s


def _overflow_risk(source_text: str, translated_text: str, item: dict) -> str:
    ratio = _char_ratio(source_text, translated_text)
    strategy = str(item.get("strategy") or "")
    constraints = ((item.get("context") or {}).get("wysiwyg_constraints") or {})
    budget = constraints.get("transformation_budget") or {}
    layout_freedom = constraints.get("layout_freedom") or {}
    forecast = constraints.get("translation_forecast") or {}
    max_ratio = budget.get("max_text_expansion_ratio")
    if isinstance(max_ratio, (int, float)) and ratio > float(max_ratio):
        return "high"
    if layout_freedom.get("allow_reflow") is False and ratio > 1.15:
        return "high"
    if float(forecast.get("overflow_probability") or 0.0) >= 0.5 and ratio > 1.15:
        return "medium"
    bbox = item.get("bbox")
    narrow = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and (float(bbox[2]) - float(bbox[0])) < 180
    if ratio > 1.7 and (narrow or strategy in {"anchored_text", "layout_constrained"}):
        return "high"
    if ratio > 1.35:
        return "medium"
    return "low"


def _source_language_leak(translated_text: str, profile: dict) -> bool:
    target = str(profile.get("target_lang") or "").lower()
    if not target.startswith("fr"):
        return False
    text = normalize_spaces(translated_text)
    return len(SOURCE_EN_MARKERS.findall(text)) >= 3


def _unchanged_problem(unchanged: bool, item: dict, profile: dict) -> bool:
    if not unchanged:
        return False
    if item.get("dry_run"):
        return False
    source_lang = str(profile.get("source_lang") or "auto").lower()
    target_lang = str(profile.get("target_lang") or "").lower()
    if source_lang and target_lang and source_lang.split("-")[0] == target_lang.split("-")[0]:
        return False
    strategy = str(item.get("strategy") or "").lower()
    object_type = str(item.get("object_type") or "").lower()
    semantic_kind = str(item.get("semantic_kind") or "").lower()
    if strategy in {"exact_preserve", "keep_original", "background_only"}:
        return False
    if object_type in {"formula", "code", "url", "reference_link", "citation"}:
        return False
    if semantic_kind in {"formula", "code", "reference"}:
        return False
    return True
