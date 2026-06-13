#!/usr/bin/env python3
"""Run a controlled translation trial on PAGEPRINT INPUT_DATA JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagetranslate import build_page_translation
from tools.run_functional_audit import _pages, audit_pages
from translation_engines import create_translation_engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="PAGEPRINT INPUT_DATA JSON, list of pages, or {'pages': [...]}")
    parser.add_argument("--engine", default=None, help="mock|prefix|rule|ct2|local|external. Defaults to TRANSLATION_ENGINE or mock.")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--inventory", default=None, help="Path to model_inventory.json")
    parser.add_argument("--model", default=None, help="Preferred model name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--fail-on", choices=["functional", "runtime", "quality", "publication", "any"], default="functional", help="Status that drives a non-zero exit code.")
    parser.add_argument("--output", default=None, help="Write trial result JSON to this path.")
    parser.add_argument("--allow-pageprint-ko", action="store_true", help="Run translation even if the preflight audit is KO.")
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    pages = _pages(payload)
    result = run_translation_trial(
        pages,
        engine_name=args.engine,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        allow_pageprint_ko=args.allow_pageprint_ko,
        engine_kwargs={
            "inventory_path": args.inventory,
            "model_name": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
        },
        batch_size=args.batch_size,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if _trial_passes(result, args.fail_on) else 1


def _trial_passes(result: dict, fail_on: str) -> bool:
    def ok(field: str) -> bool:
        return str(result.get(field) or "").lower() in {"ok", ""}

    if fail_on == "functional":
        return result.get("functional_status") == "ok"
    if fail_on == "runtime":
        return result.get("translation_runtime_status") == "ok"
    if fail_on == "quality":
        return str(result.get("linguistic_quality_status") or "").lower() in {"ok", "review"}
    if fail_on == "publication":
        return str(result.get("publication_readiness_status") or "").lower() in {"ok", "review"}
    if fail_on == "any":
        return (
            result.get("functional_status") == "ok"
            and result.get("translation_runtime_status") == "ok"
            and str(result.get("linguistic_quality_status") or "").lower() in {"ok", "review"}
            and str(result.get("publication_readiness_status") or "").lower() in {"ok", "review"}
        )
    return result.get("functional_status") == "ok"


def run_translation_trial(
    pages: list[dict],
    *,
    engine_name: str | None = None,
    target_lang: str = "fr",
    source_lang: str | None = None,
    allow_pageprint_ko: bool = False,
    engine_kwargs: dict | None = None,
    batch_size: int | None = None,
) -> dict:
    preflight = audit_pages(pages, run_pagetranslate=True, dry_run=True)
    engine = create_translation_engine(engine_name, **{k: v for k, v in (engine_kwargs or {}).items() if v is not None})
    engine_label = engine_name or getattr(engine, "profile", None) or "mock"
    if preflight.get("functional_status") != "ok" and not allow_pageprint_ko:
        return {
            "schema_version": "translation_trial.v1",
            "functional_status": "ko",
            "engine": engine_label,
            "pageprint_functional_status": preflight.get("functional_status"),
            "pagetranslate_functional_status": "not_run",
            "errors": ["pageprint_preflight_failed", *(preflight.get("errors") or [])],
            "warnings": preflight.get("warnings") or [],
            "preflight": preflight,
            "unit_count": 0,
            "qa": _empty_qa(),
            "engine_calls": [],
            "page_results": [],
        }

    page_results = []
    errors: list[str] = []
    warnings: list[str] = list(preflight.get("warnings") or [])
    aggregate_qa = _empty_qa()
    engine_calls = []
    total_units = 0
    total_latency_ms = 0.0
    batch_count = 0
    for page_index, page in enumerate(pages):
        page_started = time.perf_counter()
        translation = build_page_translation(
            page,
            translator=engine,
            target_lang=target_lang,
            source_lang=source_lang,
            dry_run=False,
            allow_fallback=False,
            batch_size=batch_size or getattr(engine, "batch_size", 8),
        )
        page_latency_ms = round((time.perf_counter() - page_started) * 1000.0, 3)
        total_latency_ms += page_latency_ms
        functional = translation.get("functional_validation") or {}
        debug = translation.get("debug") or {}
        quality = translation.get("translation_quality") or translation.get("quality") or {}
        runtime = translation.get("runtime_validation") or {}
        linguistic = translation.get("linguistic_quality_validation") or {}
        publication = translation.get("publication_readiness_validation") or {}
        page_errors = functional.get("errors") or []
        if functional.get("functional_status") != "ok":
            errors.extend(f"page_{page_index}:pagetranslate:{error}" for error in page_errors)
        _accumulate_qa(aggregate_qa, quality)
        units = translation.get("translation_units") or []
        if runtime.get("engine_supports_batch"):
            batch_count += max(1, math.ceil(len(units) / max(1, int(getattr(engine, "batch_size", 8)))))
        else:
            batch_count += len(units)
        total_units += len(units)
        for unit in units:
            engine_calls.append(_engine_call_log(page_index, unit))
        page_results.append({
            "page_index": page_index,
            "pageprint_functional_status": "ok",
            "pagetranslate_functional_status": functional.get("functional_status"),
            "selection_mode": debug.get("selection_mode"),
            "fallback_selector_used": debug.get("fallback_selector_used"),
            "generic_coalescer_used": debug.get("generic_coalescer_used"),
            "unit_count": len(units),
            "page_latency_ms": page_latency_ms,
            "batch_count": max(1, math.ceil(len(units) / max(1, int(getattr(engine, "batch_size", 8))))) if runtime.get("engine_supports_batch") else len(units),
            "pipeline_status": translation.get("pipeline_status"),
            "translation_runtime_status": translation.get("translation_runtime_status"),
            "linguistic_quality_status": translation.get("linguistic_quality_status"),
            "publication_readiness_status": translation.get("publication_readiness_status"),
            "runtime_validation": runtime,
            "linguistic_quality_validation": linguistic,
            "publication_readiness_validation": publication,
            "qa": {
                "protected_token_mismatch_count": quality.get("protected_token_mismatch_count", 0),
                "number_mismatch_count": quality.get("number_mismatch_count", 0),
                "terminology_warning_count": quality.get("terminology_warning_count", 0),
                "placeholder_corruption_count": quality.get("placeholder_corruption_count", 0),
                "needs_review_count": quality.get("needs_review_count", 0),
                "unchanged_suspect_count": quality.get("unchanged_suspect_count", 0),
                "empty_translation_count": quality.get("empty_count", 0),
            },
            "errors": page_errors,
        })

    if any(page.get("selection_mode") != "translation_plan" for page in page_results):
        errors.append("non_translation_plan_selection_mode")
    if any(page.get("fallback_selector_used") for page in page_results):
        errors.append("fallback_selector_used")
    if any(page.get("generic_coalescer_used") for page in page_results):
        errors.append("generic_coalescer_used")

    # Runtime status is aggregated from the engine, independent of functional
    # (selection/fallback) errors: a clean pipeline can still have a KO engine.
    runtime_status = "ko" if any(
        str(page.get("translation_runtime_status") or "").lower() == "ko" for page in page_results
    ) else "ok"
    return {
        "schema_version": "translation_trial.v1",
        "functional_status": "ok" if not errors else "ko",
        "engine": engine_label,
        "pageprint_functional_status": preflight.get("functional_status"),
        "pagetranslate_functional_status": "ok" if not errors else "ko",
        "pipeline_status": "ok" if preflight.get("functional_status") == "ok" else "ko",
        "translation_runtime_status": runtime_status,
        "linguistic_quality_status": _aggregate_status(page_results, "linguistic_quality_status"),
        "publication_readiness_status": _aggregate_status(page_results, "publication_readiness_status"),
        "errors": errors,
        "warnings": warnings,
        "preflight": preflight,
        "unit_count": total_units,
        "batch_count": batch_count,
        "avg_latency_ms": round(total_latency_ms / max(1, len(page_results)), 3),
        "units_per_second": round(total_units / max(1e-9, total_latency_ms / 1000.0), 3) if total_latency_ms else None,
        "qa": aggregate_qa,
        "placeholder_corruption_rate": round(
            float(aggregate_qa.get("placeholder_corruption_count") or 0) / max(1, total_units),
            3,
        ),
        "engine_calls": engine_calls,
        "page_results": page_results,
    }


def _empty_qa() -> dict:
    return {
        "protected_token_mismatch_count": 0,
        "number_mismatch_count": 0,
        "terminology_warning_count": 0,
        "placeholder_corruption_count": 0,
        "needs_review_count": 0,
        "unchanged_suspect_count": 0,
        "empty_translation_count": 0,
    }


def _accumulate_qa(target: dict, quality: dict) -> None:
    target["protected_token_mismatch_count"] += int(quality.get("protected_token_mismatch_count") or 0)
    target["number_mismatch_count"] += int(quality.get("number_mismatch_count") or 0)
    target["terminology_warning_count"] += int(quality.get("terminology_warning_count") or 0)
    target["placeholder_corruption_count"] += int(quality.get("placeholder_corruption_count") or 0)
    target["needs_review_count"] += int(quality.get("needs_review_count") or 0)
    target["unchanged_suspect_count"] += int(quality.get("unchanged_suspect_count") or 0)
    target["empty_translation_count"] += int(quality.get("empty_count") or 0)


def _engine_call_log(page_index: int, unit: dict) -> dict:
    trace = unit.get("engine_trace") or {}
    return {
        "page_index": page_index,
        "unit_id": unit.get("unit_id"),
        "translation_unit_id": unit.get("translation_unit_id"),
        "role": unit.get("role"),
        "object_type": unit.get("object_type"),
        "semantic_kind": unit.get("semantic_kind"),
        "source_text": unit.get("source_text"),
        "protected_tokens": unit.get("protected") or [],
        "translated_text": unit.get("translated_text"),
        "status": unit.get("status"),
        "qa": unit.get("quality") or {},
        "engine_trace": trace,
        "protected_source_text": trace.get("protected_source_text"),
        "raw_engine_output": trace.get("raw_engine_output"),
        "restored_output": trace.get("restored_output"),
        "post_glossary_output": trace.get("post_glossary_output"),
        "latency_ms": trace.get("latency_ms"),
        "input_token_count": trace.get("input_token_count"),
        "output_token_count": trace.get("output_token_count"),
        "truncated": trace.get("truncated"),
    }


def _aggregate_status(page_results: list[dict], field: str) -> str:
    values = [str(page.get(field) or "").lower() for page in page_results]
    if not values:
        return "unknown"
    if any(value == "ko" for value in values):
        return "ko"
    if any(value == "review" for value in values):
        return "review"
    if all(value == "ok" for value in values):
        return "ok"
    return values[0] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
