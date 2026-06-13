"""OCR routing policy.

Native text availability does not mean OCR is useless. This module decides
whether OCR should run full-page, on selected visual regions, or not at all.
"""

from __future__ import annotations


def route_ocr(page_structure: dict, *, native_available: bool, image_available: bool) -> dict:
    layout_type = str(page_structure.get("layout_type") or "").lower()
    page_role = str(page_structure.get("page_role") or "").lower()
    native_text_density = float(page_structure.get("native_text_density") or page_structure.get("text_density") or 0.0)
    regions = list(page_structure.get("regions") or []) + list(page_structure.get("special_regions") or [])
    image_regions = [
        region for region in regions
        if str(region.get("region_type") or region.get("type") or "").lower() in {"image", "image_region", "figure_region", "diagram_region"}
    ]
    probable_text_regions = [
        region for region in image_regions
        if region.get("text_probable") is True
        or str(region.get("region_type") or region.get("type") or "").lower() in {"diagram_region", "figure_region"}
        or _large_region(region, page_structure)
    ]
    if not image_available:
        mode = "none"
    elif not native_available:
        mode = "full_page"
    elif layout_type in {"image_dominant", "annotated_page"} or page_role in {"cover", "figure"}:
        mode = "targeted_regions" if probable_text_regions else "full_page"
    elif native_text_density and native_text_density < 0.08 and image_regions:
        mode = "targeted_regions"
    elif probable_text_regions:
        mode = "targeted_regions"
    else:
        mode = "none"
    targets = probable_text_regions if mode == "targeted_regions" else []
    return {
        "schema_version": "ocr_routing.v2",
        "mode": mode,
        "native_available": bool(native_available),
        "image_available": bool(image_available),
        "native_text_density": native_text_density,
        "target_regions": targets,
        "ocr_claims": [
            {
                "source": "ocr_targeted_region",
                "region_id": region.get("id") or region.get("region_id"),
                "bbox": region.get("bbox"),
                "confidence": region.get("text_probability") or region.get("confidence"),
                "reason": "visual_region_text_probable",
            }
            for region in targets
        ],
        "decision_policy": "ocr_outputs_are_claims_for_evidence_resolver",
        "reason": _reason(mode, native_available, layout_type, page_role, bool(targets)),
    }


def _reason(mode: str, native_available: bool, layout_type: str, page_role: str, has_image_regions: bool) -> str:
    if mode == "none":
        return "no_image_available" if not native_available else "native_text_sufficient_no_image_regions"
    if mode == "full_page":
        return "native_text_absent_or_image_dominant"
    if has_image_regions:
        return "targeted_visual_text_possible"
    return f"context:{page_role or layout_type}"


def _large_region(region: dict, page_structure: dict) -> bool:
    bbox = region.get("bbox")
    dimensions = page_structure.get("dimensions") or {}
    page_area = float(dimensions.get("width") or 0) * float(dimensions.get("height") or 0)
    if not page_area or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return area / page_area >= 0.12
