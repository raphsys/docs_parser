"""PAGETRANSLATE schema constants and runtime DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PAGETRANSLATE_SCHEMA_VERSION = "pagetranslate.output.v1"

SOURCE_SCHEMA_VERSION = "pageprint.input.v1"

PRIMARY_TEXT_LEVELS = ("semantic_phrase", "semantic_group", "phrase", "line", "block")

AUXILIARY_TEXT_LEVELS = ("word", "char")

TERMINAL_PUNCTUATION = (".", "!", "?", "…")


@dataclass(slots=True)
class TranslationProtection:
    placeholder: str
    text: str
    kind: str
    start: int
    end: int


@dataclass(slots=True)
class TranslationQuality:
    source_word_count: int
    translated_word_count: int
    word_expansion_ratio: float
    empty_translation: bool
    unchanged: bool
    number_mismatch: bool = False
    unit_mismatch: bool = False
    protected_token_mismatch: bool = False
    source_language_leak: bool = False
    wysiwyg_overflow_risk: str = "low"
    needs_review: bool = False
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationProfile:
    source_lang: str = "auto"
    target_lang: str = "fr"
    target_variant: str = "standard"
    domain: str | None = None
    subdomain: str | None = None
    document_type: str | None = None
    page_role: str | None = None
    page_family: str | None = None
    layout_type: str | None = None
    style: str = "professionnel"
    tone: str = "neutre"
    terminology: dict[str, Any] = field(default_factory=dict)
    protected_tokens: list[str] = field(default_factory=list)
    methodology: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationUnit:
    translation_unit_id: str
    unit_id: str
    level: str
    source_text: str
    parent_id: str | None = None
    source_unit_ids: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    role: str | None = None
    object_type: str | None = None
    object_class: str | None = None
    semantic_kind: str | None = None
    strategy: str = "layout_constrained"
    render_policy: str | None = None
    coverage_required: str | None = None
    protected: list[str] = field(default_factory=list)
    translatable: bool = True
    sentence: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    protections: list[dict[str, Any]] = field(default_factory=list)
    protected_source_text: str | None = None
    translated_text: str | None = None
    status: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationProjection:
    unit_id: str
    translation_unit_id: str
    status: str
    target_text: str
    source_unit_ids: list[str] = field(default_factory=list)
    reconstruction_compatible: bool = True
