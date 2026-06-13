"""Bridge between PAGETRANSLATE units and the historical DocumentTranslator."""

from __future__ import annotations

import os
import inspect

from .text_utils import normalize_spaces


class TranslationEngine:
    """Minimal engine contract for rev_04 translation providers."""

    profile = "external_model"

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        raise NotImplementedError


class MockTranslationEngine(TranslationEngine):
    profile = "mock"

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        return text


class TranslatorBridge:
    def __init__(self, translator=None):
        self.translator = translator

    def translate(self, item: dict, profile: dict, text: str) -> str:
        translated = self._translate_one(item, profile, text)
        if self._needs_retry(text, translated):
            retry_item = {**item, "strategy": "semantic_reflow"}
            retry_profile = {**profile, "retry_reason": "empty_or_identical_translation"}
            retried = self._translate_one(retry_item, retry_profile, text)
            if normalize_spaces(retried):
                translated = retried
        return normalize_spaces(translated)

    def translate_batch(self, items: list[dict], profile: dict) -> list[dict]:
        translator = self._translator()
        if hasattr(translator, "translate_batch"):
            requests = [self._request_from_item(item, profile) for item in items]
            raw_outputs = translator.translate_batch(requests)
            return [self._normalize_batch_item(item, output, profile) for item, output in zip(items, raw_outputs)]
        return [
            self._normalize_batch_item(item, {"translated_text": self.translate(item, profile, item.get("protected_source_text") or item.get("source_text") or "")}, profile)
            for item in items
        ]

    def _translate_one(self, item: dict, profile: dict, text: str) -> str:
        translator = self._translator()
        if not hasattr(translator, "translate_text") and not hasattr(translator, "translate"):
            raise TypeError("translator must expose translate_text(...) or translate(...)")
        return self._call(translator, item, profile, text)

    def _translator(self):
        if self.translator is None:
            try:
                from translation_engines import create_translation_engine

                self.translator = create_translation_engine(os.getenv("TRANSLATION_ENGINE") or "mock")
            except Exception:
                from translator import DocumentTranslator

                self.translator = DocumentTranslator()
        return self.translator

    def _call(self, translator, item: dict, profile: dict, text: str) -> str:
        context = item.get("context") or {}
        engine_context = {
            **context,
            "role": item.get("role"),
            "object_type": item.get("object_type"),
            "semantic_kind": item.get("semantic_kind"),
            "protected_tokens": item.get("protected") or [],
            "terminology": profile.get("terminology") or {},
            "translation_profile": profile.get("engine_profile") or {},
        }
        if hasattr(translator, "translate") and not hasattr(translator, "translate_text"):
            return translator.translate(
                text,
                source_lang=profile.get("source_lang") or "auto",
                target_lang=profile.get("target_lang") or "fr",
                context=engine_context,
            )
        kwargs = {
            "source_lang": profile.get("source_lang"),
            "target_lang": profile.get("target_lang") or "fr",
            "target_variant": profile.get("target_variant"),
            "block_role": item.get("role") or "body",
            "strategy": item.get("strategy") or "layout_constrained",
            "translatable": True,
            "domain": profile.get("domain"),
            "subdomain": profile.get("subdomain"),
            "document_type": profile.get("document_type"),
            "page_role": profile.get("page_role"),
            "page_family": profile.get("page_family"),
            "layout_type": profile.get("layout_type"),
            "style": profile.get("style"),
            "tone": profile.get("tone"),
            "terminology": profile.get("terminology") or {},
            "context_before": context.get("previous_text"),
            "context_after": context.get("next_text"),
            "section_title": context.get("section_title"),
            "protected_tokens": profile.get("protected_tokens") or [],
            "wysiwyg_constraints": context.get("wysiwyg_constraints") or {},
            "object_class": item.get("object_class") or "",
            "object_type": item.get("object_type") or "",
            "phrase_semantics": item.get("semantic_kind") or "",
        }
        return translator.translate_text(text, **self._supported_kwargs(translator.translate_text, kwargs))

    def _request_from_item(self, item: dict, profile: dict) -> dict:
        context = item.get("context") or {}
        return {
            "request_id": item.get("translation_id") or item.get("translation_unit_id") or item.get("unit_id"),
            "translation_unit_id": item.get("translation_unit_id") or item.get("unit_id"),
            "text": item.get("protected_source_text") or item.get("source_text") or "",
            "protected_text": item.get("protected_source_text") or item.get("source_text") or "",
            "source_lang": profile.get("source_lang") or "auto",
            "target_lang": profile.get("target_lang") or "fr",
            "context": {
                **context,
                "role": item.get("role"),
                "object_type": item.get("object_type"),
                "semantic_kind": item.get("semantic_kind"),
                "translation_profile": profile.get("engine_profile") or {},
            },
            "protected_tokens": item.get("protected") or [],
            "role": item.get("role"),
            "object_type": item.get("object_type"),
            "semantic_kind": item.get("semantic_kind"),
            "constraints": context.get("wysiwyg_constraints") or {},
        }

    def _normalize_batch_item(self, item: dict, output, profile: dict) -> dict:
        if isinstance(output, dict):
            translated_text = output.get("translated_text") or output.get("text") or ""
            raw_output = output.get("raw_output")
            metadata = output.get("metadata") or {}
        else:
            translated_text = str(output or "")
            raw_output = translated_text
            metadata = {}
        return {
            "translation_unit_id": item.get("translation_unit_id") or item.get("unit_id"),
            "unit_id": item.get("unit_id"),
            "translated_text": normalize_spaces(translated_text),
            "raw_output": raw_output,
            "metadata": metadata,
            "profile": profile,
        }

    @staticmethod
    def _needs_retry(source_text: str, translated_text: str) -> bool:
        src = normalize_spaces(source_text)
        out = normalize_spaces(translated_text)
        return not out or out.lower() == src.lower()

    @staticmethod
    def _supported_kwargs(func, kwargs: dict) -> dict:
        signature = inspect.signature(func)
        parameters = signature.parameters
        if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
            return kwargs
        accepted = set(parameters)
        return {key: value for key, value in kwargs.items() if key in accepted}
