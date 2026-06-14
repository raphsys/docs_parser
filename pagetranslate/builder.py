"""PAGETRANSLATE Builder — traduction pilotée par INPUT_DATA PagePrint.

Pipeline métier:

    INPUT_DATA → sélection unités traduisibles → protection → traduction
    → contrôle qualité → projection dans INPUT_DATA traduit

La logique détaillée vit dans des modules dédiés afin que chaque contrat soit
auditable séparément.
"""

from __future__ import annotations

import copy
import time
import uuid

from .context_builder import attach_unit_context, build_translation_profile
from .coalescer import coalesce_translation_units
from .functional_validator import validate_functional_translation
from .projection import project_translations
from .protection import protect_text, restore_text
from .quality import assess_translation_quality, unit_quality
from .schema import PAGETRANSLATE_SCHEMA_VERSION, PRIMARY_TEXT_LEVELS, SOURCE_SCHEMA_VERSION
from .selector import select_translation_units
from .sentence_boundary import annotate_sentence_boundaries
from .terminology import apply_post_translation_glossary, explicit_protected_terms
from .technical_protection import is_technical_role, technical_tokens
from .translation_plan_reader import read_translation_plan
from .translator_bridge import TranslatorBridge
from translation_engines.placeholder_policy import choose_placeholder_style
from translation_engines.translation_memory import TranslationMemory, load_translation_memory


class PageTranslationBuilder:
    """Construit et exécute une passe de traduction sur INPUT_DATA."""

    def __init__(self, translator=None, translation_memory=None):
        self.bridge = TranslatorBridge(translator=translator)
        # Translation memory is opt-in: an explicit instance, or TRANSLATION_MEMORY_PATH.
        # Default stays off so the model path keeps deterministic behaviour.
        if isinstance(translation_memory, TranslationMemory):
            self.translation_memory = translation_memory
        elif translation_memory:
            self.translation_memory = load_translation_memory(translation_memory)
        else:
            self.translation_memory = load_translation_memory(None)
        self.memory_hit_count = 0
        self.model_call_count = 0

    def build(
        self,
        input_data: dict,
        *,
        target_lang: str = "fr",
        source_lang: str | None = None,
        style: str | None = None,
        tone: str | None = None,
        dry_run: bool = False,
        allow_fallback: bool = True,
        batch_size: int = 8,
    ) -> dict:
        if not isinstance(input_data, dict):
            raise TypeError("input_data must be a dict")

        translated_input = copy.deepcopy(input_data)
        translation_profile = build_translation_profile(
            input_data,
            target_lang=target_lang,
            source_lang=source_lang,
            style=style,
            tone=tone,
        )
        views = input_data.get("views") or {}
        plan_field_present = "translation_plan" in views
        plan = views.get("translation_plan") if plan_field_present else None
        translation_plan_input_count = len(plan or [])
        fallback_selector_used = False
        generic_coalescer_used = False
        if plan:
            units = read_translation_plan(input_data)
            selection_mode = "translation_plan"
            selection_warning = None
        elif plan_field_present:
            units = []
            selection_mode = "translation_plan_empty"
            selection_warning = "PAGEPRINT provided an empty translation_plan"
        else:
            if allow_fallback:
                units = select_translation_units(input_data)
                units = annotate_sentence_boundaries(units)
                units = coalesce_translation_units(units)
                fallback_selector_used = True
                generic_coalescer_used = True
                selection_mode = "fallback_selector"
                selection_warning = "PAGEPRINT did not provide translation_plan"
            else:
                units = []
                selection_mode = "fallback_disabled"
                selection_warning = "PAGEPRINT did not provide translation_plan and fallback is disabled"
        units = attach_unit_context(units, input_data, translation_profile)

        translated_units = self._translate_units(
            units,
            translation_profile=translation_profile,
            dry_run=dry_run,
            batch_size=batch_size,
        )
        projections = project_translations(translated_input, translated_units)
        quality = assess_translation_quality(translated_units)
        runtime = self._runtime_status(translated_units, translation_profile, dry_run=dry_run)
        linguistic = self._linguistic_quality_status(quality, translated_units)
        publication = self._publication_readiness_status(quality, translated_units)

        translation_id = f"pagetranslate_{uuid.uuid4().hex[:12]}"
        translated_input["translation_result"] = {
            "schema_version": PAGETRANSLATE_SCHEMA_VERSION,
            "translation_id": translation_id,
            "source_input_id": input_data.get("input_id"),
            "source_schema_version": input_data.get("schema_version"),
            "target_lang": translation_profile["target_lang"],
            "source_lang": translation_profile["source_lang"],
            "unit_count": len(translated_units),
            "translated_count": sum(1 for u in translated_units if u["status"] == "translated"),
            "preserved_count": sum(1 for u in translated_units if u["status"] == "preserved"),
            "dry_run": bool(dry_run),
            "quality": quality,
            "translation_quality": {
                "protected_token_mismatch_count": quality.get("protected_token_mismatch_count", 0),
                "number_mismatch_count": quality.get("number_mismatch_count", 0),
                "terminology_warning_count": quality.get("terminology_warning_count", 0),
                "needs_review_count": quality.get("needs_review_count", 0),
            },
            "pipeline_status": "ok" if not (translation_plan_input_count == 0 and selection_mode == "translation_plan_empty") else "ko",
            "translation_runtime_status": runtime["status"],
            "linguistic_quality_status": linguistic["status"],
            "publication_readiness_status": publication["status"],
            "runtime_validation": runtime,
            "linguistic_quality_validation": linguistic,
            "publication_readiness_validation": publication,
            "projection": projections,
            "selection_mode": selection_mode,
            "warning": selection_warning,
            "translation_plan_input_count": translation_plan_input_count,
            "fallback_selector_used": fallback_selector_used,
            "generic_coalescer_used": generic_coalescer_used,
        }

        result = {
            "schema_version": PAGETRANSLATE_SCHEMA_VERSION,
            "translation_id": translation_id,
            "source_schema_version": input_data.get("schema_version"),
            "source_input_id": input_data.get("input_id"),
            "document": copy.deepcopy(input_data.get("document") or {}),
            "page": copy.deepcopy(input_data.get("page") or {}),
            "translation_profile": translation_profile,
            "translation_units": translated_units,
            "projection": projections,
            "quality": quality,
            "translation_quality": {
                "protected_token_mismatch_count": quality.get("protected_token_mismatch_count", 0),
                "number_mismatch_count": quality.get("number_mismatch_count", 0),
                "terminology_warning_count": quality.get("terminology_warning_count", 0),
                "needs_review_count": quality.get("needs_review_count", 0),
            },
            "pipeline_status": "ok" if not (translation_plan_input_count == 0 and selection_mode == "translation_plan_empty") else "ko",
            "translation_runtime_status": runtime["status"],
            "linguistic_quality_status": linguistic["status"],
            "publication_readiness_status": publication["status"],
            "runtime_validation": runtime,
            "linguistic_quality_validation": linguistic,
            "publication_readiness_validation": publication,
            "translated_input_data": translated_input,
            "debug": {
                "input_is_pageprint": input_data.get("schema_version") == SOURCE_SCHEMA_VERSION,
                "selection_levels": list(PRIMARY_TEXT_LEVELS),
                "selection_mode": selection_mode,
                "translation_plan_input_count": translation_plan_input_count,
                "fallback_selector_used": fallback_selector_used,
                "generic_coalescer_used": generic_coalescer_used,
                "warning": selection_warning,
                "selection_policy": "translation_plan first; selector/coalescer fallback only when plan is missing",
                "fine_token_policy": "word/char excluded from translation units",
            },
        }
        result["functional_validation"] = validate_functional_translation(result)
        return result

    def _translate_units(self, items: list[dict], *, translation_profile: dict, dry_run: bool, batch_size: int) -> list[dict]:
        prepared = [self._prepare_item(item, translation_profile=translation_profile) for item in items]
        self.memory_hit_count = 0
        self.model_call_count = 0
        if dry_run:
            return [
                self._finalize_item(
                    item,
                    item.get("protected_source_text") or item.get("source_text") or "",
                    translation_profile=translation_profile,
                    dry_run=True,
                    raw_output=None,
                    metadata={"latency_ms": 0.0, "input_token_count": 0, "output_token_count": 0, "truncated": False},
                )
                for item in prepared
            ]

        # Translation memory: validated hits skip the engine entirely.
        results: list[dict | None] = [None] * len(prepared)
        engine_items: list[dict] = []
        engine_indices: list[int] = []
        for index, item in enumerate(prepared):
            hit = self._memory_lookup(item, translation_profile)
            if hit is not None:
                self.memory_hit_count += 1
                results[index] = self._finalize_item(
                    item,
                    hit["target"],
                    translation_profile=translation_profile,
                    dry_run=False,
                    raw_output=hit["target"],
                    metadata={"latency_ms": 0.0, "input_token_count": 0, "output_token_count": 0, "truncated": False},
                    memory_hit=True,
                    memory_source=hit.get("memory_source"),
                )
            else:
                engine_items.append(item)
                engine_indices.append(index)

        for finalized, index in zip(self._run_engine(engine_items, translation_profile, batch_size), engine_indices):
            results[index] = finalized
        return [item for item in results if item is not None]

    def _run_engine(self, items: list[dict], translation_profile: dict, batch_size: int) -> list[dict]:
        if not items:
            return []
        self.model_call_count += len(items)
        translator = self.bridge._translator()
        supports_batch = bool(getattr(translator, "supports_batch", False) or hasattr(translator, "translate_batch"))
        translated_items: list[dict] = []
        if supports_batch:
            for start in range(0, len(items), max(1, int(batch_size))):
                batch = items[start:start + max(1, int(batch_size))]
                batch_started = time.perf_counter()
                try:
                    batch_outputs = self.bridge.translate_batch(batch, translation_profile)
                except Exception as exc:
                    translated_items.extend(self._finalize_engine_error(item, exc, translation_profile) for item in batch)
                    continue
                batch_latency_ms = round((time.perf_counter() - batch_started) * 1000.0, 3)
                for item, output in zip(batch, batch_outputs):
                    translated_text = output.get("translated_text") or ""
                    metadata = dict(output.get("metadata") or {})
                    metadata.setdefault("latency_ms", batch_latency_ms / max(1, len(batch)))
                    translated_items.append(
                        self._finalize_item(
                            item,
                            translated_text,
                            translation_profile=translation_profile,
                            dry_run=False,
                            raw_output=output.get("raw_output"),
                            metadata=metadata,
                        )
                    )
            return translated_items
        for item in items:
            started = time.perf_counter()
            try:
                translated_text = self.bridge.translate(item, translation_profile, item.get("protected_source_text") or item.get("source_text") or "")
            except Exception as exc:
                translated_items.append(self._finalize_engine_error(item, exc, translation_profile))
                continue
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            translated_items.append(
                self._finalize_item(
                    item,
                    translated_text,
                    translation_profile=translation_profile,
                    dry_run=False,
                    raw_output=translated_text,
                    metadata={"latency_ms": latency_ms, "input_token_count": 0, "output_token_count": 0, "truncated": False},
                )
            )
        return translated_items

    def _memory_lookup(self, item: dict, translation_profile: dict) -> dict | None:
        memory = self.translation_memory
        if not memory or len(memory) == 0:
            return None
        source_text = item.get("source_text") or ""
        if not source_text.strip():
            return None
        domain = translation_profile.get("domain") or (item.get("context") or {}).get("domain")
        return memory.lookup(
            source_text,
            translation_profile.get("source_lang") or "auto",
            translation_profile.get("target_lang") or "fr",
            domain=domain,
        )

    def _finalize_engine_error(self, item: dict, exc: Exception, translation_profile: dict) -> dict:
        return self._finalize_item(
            item,
            None,
            translation_profile=translation_profile,
            dry_run=False,
            raw_output=None,
            metadata={"latency_ms": 0.0, "input_token_count": 0, "output_token_count": 0, "truncated": False},
            engine_error={"error_type": type(exc).__name__, "message": str(exc)},
        )

    def _prepare_item(self, item: dict, *, translation_profile: dict) -> dict:
        source_text = item["source_text"]
        explicit_tokens = [
            *explicit_protected_terms(translation_profile),
            *(item.get("protected") or []),
        ]
        # Protect technical tokens (None/Conv2D/shapes/SQL/paths) in code/table cells.
        if is_technical_role(item.get("role"), item.get("object_type")):
            explicit_tokens.extend(technical_tokens(source_text))
        engine = self.bridge._translator()
        placeholder_style = choose_placeholder_style(engine_name=getattr(engine, "profile", None))
        protected_text, protections = protect_text(
            source_text,
            explicit_tokens=explicit_tokens,
            placeholder_style=placeholder_style,
            engine_name=getattr(engine, "profile", None),
        )
        item = {
            **item,
            "protected_source_text": protected_text,
            "protections": protections,
        }
        return item

    def _finalize_item(
        self,
        item: dict,
        translated_text: str,
        *,
        translation_profile: dict,
        dry_run: bool,
        raw_output,
        metadata: dict,
        memory_hit: bool = False,
        memory_source: str | None = None,
        engine_error: dict | None = None,
    ) -> dict:
        error = engine_error
        if error:
            translated_text = None
            status = "error"
        else:
            try:
                if dry_run:
                    translated_text = item.get("source_text")
                    status = "dry_run"
                elif memory_hit:
                    # Validated memory output is already final target text.
                    status = "translated" if translated_text and translated_text != item.get("source_text") else "preserved"
                else:
                    restored = restore_text(translated_text, item.get("protections") or [])
                    translated_text = apply_post_translation_glossary(restored, translation_profile)
                    status = "translated" if translated_text and translated_text != item.get("source_text") else "preserved"
            except Exception as exc:
                translated_text = None
                status = "error"
                error = {"error_type": type(exc).__name__, "message": str(exc)}

        item["translation_id"] = f"tr_{uuid.uuid4().hex[:12]}"
        item["translated_text"] = translated_text
        item["status"] = status
        item["raw_engine_output"] = raw_output
        item["engine_trace"] = {
            "engine": getattr(self.bridge._translator(), "profile", None) or type(self.bridge._translator()).__name__,
            "model_name": getattr(self.bridge._translator(), "model_name", None),
            "model_family": getattr(self.bridge._translator(), "model_family", None),
            "protected_source_text": item.get("protected_source_text"),
            "raw_engine_output": raw_output,
            "restored_output": translated_text,
            "post_glossary_output": translated_text,
            "latency_ms": metadata.get("latency_ms"),
            "input_token_count": metadata.get("input_token_count"),
            "output_token_count": metadata.get("output_token_count"),
            "truncated": metadata.get("truncated"),
            "memory_hit": bool(memory_hit),
            "memory_source": memory_source,
            "translation_profile": translation_profile.get("engine_profile") or {},
            "metadata": metadata,
        }
        item["quality"] = unit_quality(item.get("source_text") or "", item["translated_text"] or "", item, translation_profile)
        if status == "preserved" and item["quality"].get("unchanged_problem"):
            item["status"] = "unchanged_suspect"
            item["preserve_reason"] = "identical_output_suspected"
        elif status == "preserved":
            item["preserve_reason"] = "preserved_by_policy_or_translator"
        if error:
            item["error"] = error
            item["quality"]["needs_review"] = True
            item["quality"]["translation_error"] = True
        return item

    def _runtime_status(self, translated_units: list[dict], translation_profile: dict, *, dry_run: bool) -> dict:
        engine = self.bridge._translator()
        has_error = any(unit.get("status") == "error" for unit in translated_units)
        return {
            "status": "ok" if not has_error else "ko",
            "dry_run": bool(dry_run),
            "translator_profile": getattr(engine, "profile", None) or type(engine).__name__,
            "unit_error_count": sum(1 for unit in translated_units if unit.get("status") == "error"),
            "engine_supports_batch": bool(getattr(engine, "supports_batch", False) or hasattr(engine, "translate_batch")),
            "batch_size": getattr(engine, "batch_size", None),
            "memory_hit_count": self.memory_hit_count,
            "model_call_count": self.model_call_count,
        }

    def _linguistic_quality_status(self, quality: dict, translated_units: list[dict]) -> dict:
        needs_review = int(quality.get("needs_review_count") or 0)
        number_mismatch = int(quality.get("number_mismatch_count") or 0)
        protected_mismatch = int(quality.get("protected_token_mismatch_count") or 0)
        terminology_warning = int(quality.get("terminology_warning_count") or 0)
        empty = int(quality.get("empty_count") or 0)
        if protected_mismatch or number_mismatch or empty:
            status = "ko"
        elif needs_review or terminology_warning:
            status = "review"
        else:
            status = "ok"
        return {
            "status": status,
            "needs_review_count": needs_review,
            "protected_token_mismatch_count": protected_mismatch,
            "number_mismatch_count": number_mismatch,
            "terminology_warning_count": terminology_warning,
            "empty_translation_count": empty,
            "translation_unit_count": len(translated_units),
        }

    def _publication_readiness_status(self, quality: dict, translated_units: list[dict]) -> dict:
        overflow = sum(1 for unit in translated_units if (unit.get("quality") or {}).get("wysiwyg_overflow_risk") == "high")
        if overflow:
            status = "ko"
        elif int(quality.get("needs_review_count") or 0):
            status = "review"
        else:
            status = "ok"
        return {
            "status": status,
            "overflow_risk_count": overflow,
            "translation_unit_count": len(translated_units),
        }


def build_page_translation(
    input_data: dict,
    *,
    target_lang: str = "fr",
    source_lang: str | None = None,
    style: str | None = None,
    tone: str | None = None,
    translator=None,
    translation_memory=None,
    dry_run: bool = False,
    allow_fallback: bool = True,
    batch_size: int = 8,
) -> dict:
    return PageTranslationBuilder(translator=translator, translation_memory=translation_memory).build(
        input_data,
        target_lang=target_lang,
        source_lang=source_lang,
        style=style,
        tone=tone,
        dry_run=dry_run,
        allow_fallback=allow_fallback,
        batch_size=batch_size,
    )
