"""Translation context/profile construction."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict

from .schema import TranslationProfile
from .text_utils import normalize_spaces


def build_translation_profile(
    input_data: dict,
    *,
    target_lang: str,
    source_lang: str | None,
    style: str | None,
    tone: str | None,
) -> dict:
    document = input_data.get("document") or {}
    page_intelligence = input_data.get("page_intelligence") or {}
    context = input_data.get("translation_context") or {}
    language = document.get("language") or {}
    profile = TranslationProfile(
        source_lang=source_lang or context.get("source_lang") or language.get("source_lang") or "auto",
        target_lang=target_lang or context.get("target_lang") or language.get("target_lang") or "fr",
        target_variant=context.get("target_variant") or "standard",
        domain=context.get("document_domain") or document.get("document_domain"),
        subdomain=context.get("document_subdomain") or document.get("document_subdomain"),
        document_type=document.get("detected_document_type") or page_intelligence.get("document_type"),
        page_role=page_intelligence.get("page_role") or (input_data.get("page") or {}).get("page_role"),
        page_family=page_intelligence.get("page_family"),
        layout_type=page_intelligence.get("layout_type"),
        style=style or context.get("translation_style") or "professionnel",
        tone=tone or context.get("translation_tone") or "neutre",
        terminology=copy.deepcopy(context.get("terminology") or {}),
        protected_tokens=list(context.get("protected_tokens") or []),
        methodology={
            "source": "ocr_server.translation_methodology + scripts/perfect_traduction.md",
            "principles": [
                "fidelity_to_meaning",
                "target_language_naturality",
                "semantic_unit_before_visual_line",
                "document_context",
                "domain_terminology",
                "style_tone_function",
                "layout_constraints",
                "protected_non_translatable_elements",
                "wysiwyg_reconstruction_compatibility",
            ],
        },
    )
    profile_dict = asdict(profile)
    profile_dict["terminology"] = dict(profile_dict.get("terminology") or {})
    profile_dict["terminology"].setdefault(
        "terms_path",
        context.get("terminology", {}).get("terms_path")
        or os.getenv("TRANSLATION_TERMINOLOGY_PATH")
        or "ai_models/translation/terminology_en_fr.json",
    )
    profile_dict["engine_profile"] = _build_engine_profile(profile_dict)
    return profile_dict


def _build_engine_profile(profile_dict: dict) -> dict:
    """Load translation/style profiles and summarise them for the engine.

    Never raises: a missing profile file yields a minimal summary.
    """
    try:
        from translation_engines.profile_store import load_profile_store

        store = load_profile_store()
        return store.engine_profile(
            target_lang=profile_dict.get("target_lang"),
            style=profile_dict.get("style"),
            tone=profile_dict.get("tone"),
            domain=profile_dict.get("domain"),
        )
    except Exception:
        return {
            "target_lang": profile_dict.get("target_lang"),
            "style": profile_dict.get("style"),
            "tone": profile_dict.get("tone"),
            "domain": profile_dict.get("domain"),
            "quality_thresholds": {},
            "post_edit_enabled": False,
            "has_profile": False,
            "has_style_tone": False,
        }


def attach_unit_context(units: list[dict], input_data: dict, profile: dict) -> list[dict]:
    section_titles = _section_titles(input_data)
    unit_map = {
        unit.get("unit_id"): unit
        for unit in input_data.get("units") or []
        if isinstance(unit, dict) and unit.get("unit_id")
    }
    for idx, item in enumerate(units):
        prev_item = units[idx - 1] if idx else None
        next_item = units[idx + 1] if idx + 1 < len(units) else None
        source_unit = _primary_source_unit(item, unit_map)
        item["context"] = {
            "previous_text": normalize_spaces(prev_item.get("source_text")) if prev_item else "",
            "next_text": normalize_spaces(next_item.get("source_text")) if next_item else "",
            "section_title": _nearest_section_title(item, section_titles),
            "page_role": profile.get("page_role"),
            "page_family": profile.get("page_family"),
            "document_type": profile.get("document_type"),
            "domain": profile.get("domain"),
            "subdomain": profile.get("subdomain"),
            "style": profile.get("style"),
            "tone": profile.get("tone"),
            "wysiwyg_constraints": {
                "strategy": item.get("strategy"),
                "render_policy": item.get("render_policy"),
                "coverage_required": item.get("coverage_required"),
                "bbox": item.get("bbox"),
                "transformation_budget": (source_unit or {}).get("transformation_budget"),
                "layout_freedom": (source_unit or {}).get("layout_freedom"),
                "render_contract": (source_unit or {}).get("render_contract"),
                "translation_forecast": (source_unit or {}).get("translation_forecast"),
            },
        }
    return units


def _primary_source_unit(item: dict, unit_map: dict[str, dict]) -> dict | None:
    if item.get("unit_id") in unit_map:
        return unit_map[item["unit_id"]]
    for source_id in item.get("source_unit_ids") or []:
        if source_id in unit_map:
            return unit_map[source_id]
    return None


def _section_titles(input_data: dict) -> list[dict]:
    titles = []
    units = input_data.get("units") or []
    for unit in units:
        understanding = unit.get("understanding") or {}
        if understanding.get("role") in {"section_heading", "title"}:
            text = normalize_spaces((unit.get("content") or {}).get("text"))
            if text:
                titles.append({
                    "text": text,
                    "reading_order_index": (unit.get("geometry") or {}).get("reading_order_index") or 0,
                    "bbox": (unit.get("geometry") or {}).get("bbox"),
                })
    return sorted(titles, key=lambda item: item["reading_order_index"])


def _nearest_section_title(item: dict, titles: list[dict]) -> str:
    if not titles:
        return ""
    order = item.get("reading_order_index") or 0
    previous = [title for title in titles if title.get("reading_order_index", 0) <= order]
    if previous:
        return previous[-1]["text"]
    return titles[0]["text"]
