#!/usr/bin/env python3
"""Run strict rev_04 functional audit on PAGEPRINT INPUT_DATA JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pageprint.functional_validator import validate_functional as validate_pageprint_functional
from pagetranslate import build_page_translation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", nargs="?", help="PAGEPRINT INPUT_DATA JSON, list of pages, or {'pages': [...]}")
    parser.add_argument("--run-pagetranslate", action="store_true", help="Run PAGETRANSLATE on each page and audit fallback/debug fields.")
    parser.add_argument("--dry-run", action="store_true", help="Use PAGETRANSLATE dry-run mode.")
    args = parser.parse_args()
    if not args.input_json:
        output = _empty_error("no_input_json_provided")
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 1

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    pages = _pages(payload)
    output = audit_pages(pages, run_pagetranslate=args.run_pagetranslate, dry_run=args.dry_run)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["functional_status"] == "ok" else 1


def audit_pages(pages: list[dict], *, run_pagetranslate: bool = False, dry_run: bool = True) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    metrics = {
        "pages_total": len(pages),
        "pages_with_translation_plan": 0,
        "pages_using_translation_plan": 0,
        "pages_using_fallback": 0,
        "translation_plan_items": 0,
        "role_none_translation_items": 0,
        "word_char_translation_items": 0,
        "reconstruction_units_missing_roles": 0,
        "reconstruction_units_missing_render_target": 0,
        "fallback_selector_usage": 0,
        "generic_coalesced_units": 0,
        "protected_tokens_missing": 0,
        "publisher_mark_sent_to_translation": 0,
        "watermark_sent_to_translation": 0,
        "caption_raw_block_translation": 0,
        "table_pages_without_tables": 0,
        "index_pages_without_index_entries": 0,
    }
    page_reports = []
    for page_index, page in enumerate(pages):
        report = validate_pageprint_functional(page)
        page_report = {
            "page_index": page_index,
            "pageprint_functional_status": report.get("functional_status"),
            "pageprint_errors": report.get("errors") or [],
            "pagetranslate_selection_mode": None,
        }
        if report.get("functional_status") != "ok":
            errors.extend(f"page_{page_index}:{error}" for error in report.get("errors") or [])
        warnings.extend(f"page_{page_index}:{warning}" for warning in report.get("warnings") or [])
        views = page.get("views") or {}
        plan = views.get("translation_plan") or []
        if "translation_plan" in views:
            metrics["pages_with_translation_plan"] += 1
        if plan:
            metrics["pages_using_translation_plan"] += 1
        if (report.get("metrics") or {}).get("fallback_required_after_pageprint"):
            metrics["pages_using_fallback"] += 1
            metrics["fallback_selector_usage"] += 1
        metrics["translation_plan_items"] += len(plan)
        metrics["role_none_translation_items"] += (report.get("metrics") or {}).get("role_none_translation_units", 0)
        metrics["word_char_translation_items"] += (report.get("metrics") or {}).get("word_char_translation_units", 0)
        metrics["reconstruction_units_missing_roles"] += (report.get("metrics") or {}).get("reconstruction_units_missing_roles", 0)
        metrics["publisher_mark_sent_to_translation"] += (report.get("metrics") or {}).get("publisher_mark_sent_to_translation", 0)
        metrics["watermark_sent_to_translation"] += (report.get("metrics") or {}).get("watermark_sent_to_translation", 0)
        metrics["caption_raw_block_translation"] += (report.get("metrics") or {}).get("caption_raw_block_translation", 0)
        metrics["table_pages_without_tables"] += (report.get("metrics") or {}).get("table_pages_without_tables", 0)
        metrics["index_pages_without_index_entries"] += (report.get("metrics") or {}).get("index_pages_without_index_entries", 0)
        metrics["protected_tokens_missing"] += sum(
            1 for item in plan
            if item.get("qa_requirements", {}).get("preserve_protected_tokens") and "protected_tokens" not in item
        )

        if run_pagetranslate:
            translation_result = build_page_translation(page, dry_run=dry_run, allow_fallback=False)
            debug = translation_result.get("debug") or {}
            functional = translation_result.get("functional_validation") or {}
            page_report["pagetranslate_selection_mode"] = debug.get("selection_mode")
            page_report["pagetranslate_functional_status"] = functional.get("functional_status")
            page_report["pagetranslate_errors"] = functional.get("errors") or []
            if debug.get("selection_mode") == "translation_plan":
                metrics["pages_using_translation_plan"] += 0 if plan else 1
            if debug.get("fallback_selector_used"):
                metrics["pages_using_fallback"] += 1
                metrics["fallback_selector_usage"] += 1
                errors.append(f"page_{page_index}:pagetranslate_fallback_selector_used")
            if debug.get("generic_coalescer_used"):
                metrics["generic_coalesced_units"] += 1
                errors.append(f"page_{page_index}:pagetranslate_generic_coalescer_used")
            pt_metrics = functional.get("metrics") or {}
            metrics["reconstruction_units_missing_roles"] += (
                pt_metrics.get("reconstruction_units_missing_roles", 0)
                or pt_metrics.get("reconstruction_unit_role_missing", 0)
            )
            metrics["reconstruction_units_missing_render_target"] += (
                pt_metrics.get("reconstruction_units_missing_render_target", 0)
                or pt_metrics.get("reconstruction_unit_render_target_missing", 0)
            )
            if functional.get("functional_status") != "ok":
                errors.extend(f"page_{page_index}:pagetranslate:{error}" for error in functional.get("errors") or [])
        page_reports.append(page_report)

    if metrics["pages_using_fallback"] > 0:
        errors.append("pages_using_fallback_gt_0")
    if metrics["role_none_translation_items"] > 0:
        errors.append("role_none_translation_items_gt_0")
    if metrics["word_char_translation_items"] > 0:
        errors.append("word_char_translation_items_gt_0")
    if metrics["reconstruction_units_missing_roles"] > 0:
        errors.append("reconstruction_units_missing_roles_gt_0")
    if metrics["reconstruction_units_missing_render_target"] > 0:
        errors.append("reconstruction_units_missing_render_target_gt_0")
    if metrics["protected_tokens_missing"] > 0:
        errors.append("protected_tokens_missing_gt_0")
    if metrics["publisher_mark_sent_to_translation"] > 0:
        errors.append("publisher_mark_sent_to_translation_gt_0")
    if metrics["watermark_sent_to_translation"] > 0:
        errors.append("watermark_sent_to_translation_gt_0")
    if metrics["caption_raw_block_translation"] > 0:
        errors.append("caption_raw_block_translation_gt_0")
    if metrics["table_pages_without_tables"] > 0:
        errors.append("table_pages_without_tables_gt_0")
    if metrics["index_pages_without_index_entries"] > 0:
        errors.append("index_pages_without_index_entries_gt_0")

    return {
        "schema_status": "unknown",
        "functional_status": "ok" if not errors else "ko",
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
        "page_reports": page_reports,
    }


def _pages(payload) -> list[dict]:
    if isinstance(payload, list):
        return [page for page in payload if isinstance(page, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        return [page for page in payload["pages"] if isinstance(page, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _empty_error(error: str) -> dict:
    return {
        "schema_status": "unknown",
        "functional_status": "ko",
        "errors": [error],
        "warnings": [],
        "metrics": {
            "pages_total": 0,
            "pages_with_translation_plan": 0,
            "pages_using_translation_plan": 0,
            "pages_using_fallback": 0,
            "translation_plan_items": 0,
            "role_none_translation_items": 0,
            "word_char_translation_items": 0,
            "reconstruction_units_missing_roles": 0,
            "reconstruction_units_missing_render_target": 0,
            "protected_tokens_missing": 0,
            "publisher_mark_sent_to_translation": 0,
            "watermark_sent_to_translation": 0,
            "caption_raw_block_translation": 0,
            "table_pages_without_tables": 0,
            "index_pages_without_index_entries": 0,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
