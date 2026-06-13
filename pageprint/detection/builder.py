"""PAGE_REGION_DETECT orchestration.

This phase is internal to PAGEPRINT but deliberately isolated. It runs the
legacy hybrid detector when possible, normalizes its output, merges it with
regions already supplied by upstream extractors, and never makes YOLO/ONNX a
hard dependency.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from .schema import PAGE_REGION_DETECT_SCHEMA_VERSION, PROTECTED_SPECIAL_CLASSES


class PageRegionDetectBuilder:
    """Run and normalize special visual region detection for PagePrint."""

    def build(
        self,
        *,
        page_structure: dict,
        page_image: Any = None,
        pdf_page: Any = None,
        sx: float = 1.0,
        sy: float = 1.0,
        run_detector: bool = False,
        force_detect: bool = False,
        normalize_existing: bool = True,
    ) -> tuple[dict, dict]:
        work = copy.deepcopy(page_structure or {})
        existing = [dict(region) for region in work.get("special_regions") or [] if isinstance(region, dict)]
        detected: list[dict] = []
        detector_info: dict = {
            "available": False,
            "reason": "not_run",
        }

        should_run_detector = bool(force_detect or (run_detector and not existing))
        if should_run_detector:
            try:
                from special_region_detector import detect_special_regions

                detected_work, detector_info = detect_special_regions(
                    work,
                    page_image=page_image,
                    pdf_page=pdf_page,
                    sx=sx,
                    sy=sy,
                )
                detected = [
                    dict(region)
                    for region in (detected_work.get("special_regions") or [])
                    if isinstance(region, dict)
                ]
            except Exception as exc:
                detector_info = {
                    "available": False,
                    "reason": f"detector_error:{type(exc).__name__}",
                    "error": str(exc),
                }
        elif existing:
            detector_info = {
                "available": True,
                "reason": "upstream_special_regions_reused",
                "reused_existing_count": len(existing),
            }

        normalized_existing = [
            normalize_detected_region(region, source_default="upstream_special_regions")
            for region in existing
        ] if normalize_existing else existing
        normalized_detected = [normalize_detected_region(region, source_default="page_region_detect") for region in detected]
        merged = _dedupe_regions(normalized_existing + normalized_detected)
        work["special_regions"] = merged

        result = {
            "schema_version": PAGE_REGION_DETECT_SCHEMA_VERSION,
            "changed": len(merged) != len(existing) or merged != normalized_existing,
            "special_region_count": len(merged),
            "special_candidate_region_count": sum(
                1 for region in merged
                if str(region.get("region_type") or "").endswith("_candidate_region")
            ),
            "detectors": {
                "hybrid_special_region_detector": detector_info,
                "onnx_yolo": _onnx_debug(detector_info),
                "pdf_glyph_formula": {
                    "available": bool((detector_info or {}).get("pdf_glyph_candidate_count") is not None),
                    "candidate_count": (detector_info or {}).get("pdf_glyph_candidate_count", 0),
                },
                "block_heuristic": {
                    "available": True,
                    "candidate_count": sum(
                        1 for region in normalized_detected
                        if "block_" in str(region.get("detection_source") or "")
                    ),
                },
            },
            "warnings": [],
        }
        if not os.environ.get("DOCS_PARSER_SPECIAL_REGION_MODEL"):
            result["warnings"].append("onnx_yolo_model_not_configured")
        return work, result


def build_page_region_detect(
    *,
    page_structure: dict,
    page_image: Any = None,
    pdf_page: Any = None,
    sx: float = 1.0,
    sy: float = 1.0,
    run_detector: bool = False,
    force_detect: bool = False,
) -> tuple[dict, dict]:
    return PageRegionDetectBuilder().build(
        page_structure=page_structure,
        page_image=page_image,
        pdf_page=pdf_page,
        sx=sx,
        sy=sy,
        run_detector=run_detector,
        force_detect=force_detect,
    )


def normalize_detected_region(region: dict, *, source_default: str = "page_region_detect") -> dict:
    out = dict(region or {})
    special_class = str(
        out.get("special_class")
        or out.get("object_type")
        or out.get("object_class")
        or out.get("region_type")
        or out.get("type")
        or out.get("kind")
        or "protected_visual"
    ).strip()
    normalized_class = special_class or "protected_visual"
    if normalized_class.lower() in PROTECTED_SPECIAL_CLASSES:
        out["region_type"] = _candidate_region_type(normalized_class)
        out["claim_type"] = _claim_type(normalized_class)
        out["policy_pending"] = True
        out["observation_only"] = True
        out.setdefault("reason", "special_region_detector")
    else:
        out.setdefault("region_type", normalized_class)
    out.setdefault("special_class", normalized_class)
    out.setdefault("object_type", normalized_class)
    out.setdefault("object_class", normalized_class)
    out.setdefault("source", out.get("detection_source") or source_default)
    out.setdefault("detection_source", out.get("source") or source_default)
    return out


def _candidate_region_type(special_class: str) -> str:
    normalized = str(special_class or "").lower()
    if any(key in normalized for key in ("formula", "equation", "math", "chemical", "symbolic")):
        return "formula_candidate_region"
    if any(key in normalized for key in ("code", "algorithm")):
        return "code_candidate_region"
    if "table" in normalized:
        return "table_candidate_region"
    if "diagram" in normalized:
        return "diagram_region"
    return "visual_candidate_region"


def _claim_type(special_class: str) -> str:
    region_type = _candidate_region_type(special_class)
    if region_type == "formula_candidate_region":
        return "formula_candidate"
    if region_type == "code_candidate_region":
        return "code_candidate"
    if region_type == "table_candidate_region":
        return "table_candidate"
    return "visual_candidate"


def _dedupe_regions(regions: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen = set()
    for region in regions:
        bbox = region.get("bbox") or region.get("visual_bbox")
        key = (
            region.get("region_type"),
            region.get("object_type"),
            tuple(round(float(v), 1) for v in bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
        )
        if key in seen:
            continue
        if _overlaps_existing_equivalent(region, deduped):
            continue
        seen.add(key)
        deduped.append(region)
    return deduped


def _overlaps_existing_equivalent(region: dict, existing_regions: list[dict]) -> bool:
    bbox = region.get("bbox") or region.get("visual_bbox")
    for existing in existing_regions:
        if existing.get("region_type") != region.get("region_type"):
            continue
        if existing.get("object_type") != region.get("object_type"):
            continue
        existing_bbox = existing.get("bbox") or existing.get("visual_bbox")
        if _intersection_over_smaller(bbox, existing_bbox) >= 0.92:
            return True
    return False


def _intersection_over_smaller(left: object, right: object) -> float:
    if not (
        isinstance(left, (list, tuple)) and len(left) == 4
        and isinstance(right, (list, tuple)) and len(right) == 4
    ):
        return 0.0
    lx0, ly0, lx1, ly1 = [float(v) for v in left]
    rx0, ry0, rx1, ry1 = [float(v) for v in right]
    inter = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    return inter / max(1.0, min(left_area, right_area))


def _onnx_debug(detector_info: dict) -> dict:
    ai = (detector_info or {}).get("ai")
    if isinstance(ai, dict):
        return ai
    return {
        "available": False,
        "reason": "no_ai_detector_report",
    }
