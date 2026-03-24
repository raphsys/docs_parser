import os
import re

import fitz

from visual_compare import compare_reconstruction


def _normalize_spaces(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _iter_translated_units(pages):
    for page in pages or []:
        if not isinstance(page, dict):
            continue
        if page.get("schema_version") == "layout.v2" and str(page.get("page_role", "")).strip().lower() == "toc":
            rows = ((page.get("toc") or {}).get("toc_rows") or [])
            for row in rows:
                label = _normalize_spaces(row.get("translated_label") or row.get("label") or "")
                if label:
                    yield {
                        "page_id": page.get("page", 0),
                        "role": "toc_label",
                        "strategy": "layout_constrained",
                        "source_text": _normalize_spaces(row.get("label") or ""),
                        "translated_text": label,
                    }
            continue
        for block in page.get("blocks", []) or []:
            role = block.get("role") or "body"
            for line in block.get("lines", []) or []:
                for phrase in line.get("phrases", []) or []:
                    yield {
                        "page_id": page.get("page", 0),
                        "role": phrase.get("role") or role,
                        "strategy": phrase.get("translation_strategy") or block.get("translation_strategy") or "semantic_reflow",
                        "unit_type": phrase.get("unit_type") or line.get("unit_type") or block.get("unit_type") or "",
                        "page_family": page.get("page_family") or ((page.get("layout") or {}).get("page_family")) or "",
                        "translatable": bool(phrase.get("translatable", block.get("translatable", True))),
                        "render_policy": phrase.get("render_policy") or line.get("render_policy") or block.get("render_policy") or "",
                        "render_mode": phrase.get("render_mode") or line.get("render_mode") or block.get("render_mode") or "",
                        "source_text": _normalize_spaces(
                            phrase.get("texte_original") or phrase.get("raw_text") or phrase.get("text") or ""
                        ),
                        "translated_text": _normalize_spaces(
                            phrase.get("translated_text") or phrase.get("texte") or phrase.get("text") or ""
                        ),
                    }


def _english_leak_count(pages, target_lang="fr"):
    if (target_lang or "").strip().lower() not in {"fr", "french"}:
        return {"unit_count": 0, "flagged_units": 0, "flagged_samples": []}
    rx = re.compile(
        r"\b(the|and|with|for|from|this|that|what|where|why|building|using|overfitting)\b",
        re.IGNORECASE,
    )
    flagged = []
    total = 0
    for unit in _iter_translated_units(pages):
        if not unit.get("translatable", True):
            continue
        if (unit.get("strategy") or "").strip().lower() == "exact_preserve":
            continue
        if _normalize_spaces(unit.get("unit_type") or "").lower() in {"reference_link", "citation", "code_visible"}:
            continue
        if _normalize_spaces(unit.get("render_policy") or "").lower() == "background_only":
            continue
        if _normalize_spaces(unit.get("render_mode") or "").lower() == "background_only":
            continue
        text = unit["translated_text"]
        if not text:
            continue
        total += 1
        hits = len(rx.findall(text))
        if hits >= 1:
            flagged.append(
                {
                    "page_id": unit["page_id"],
                    "role": unit["role"],
                    "strategy": unit["strategy"],
                    "source_text": unit["source_text"],
                    "translated_text": text,
                }
            )
    return {
        "unit_count": total,
        "flagged_units": len(flagged),
        "flagged_samples": flagged[:25],
    }


def _overlap_ratio(a, b):
    r1 = fitz.Rect(a)
    r2 = fitz.Rect(b)
    inter = (r1 & r2).get_area()
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, min(r1.get_area(), r2.get_area()))


def _is_decorative_raster(rect, page_area):
    if not isinstance(rect, fitz.Rect):
        rect = fitz.Rect(rect)
    w = max(0.0, rect.width)
    h = max(0.0, rect.height)
    if w <= 0 or h <= 0:
        return True
    aspect = max(w / max(1e-9, h), h / max(1e-9, w))
    area_ratio = rect.get_area() / max(1e-9, page_area)
    # Ignore tiny chart glyph rasters such as axis ticks or small markers.
    if area_ratio < 0.0004 and max(w, h) < 18.0:
        return True
    if min(w, h) < 10.0 and aspect >= 8.0:
        return True
    if min(w, h) < 14.0 and aspect >= 14.0:
        return True
    if area_ratio < 0.0008 and aspect >= 10.0:
        return True
    return False


def _evaluate_layout_pdf(pdf_path, overlap_threshold=0.25, text_image_threshold=0.10, fullpage_area_ratio=0.95):
    doc = fitz.open(pdf_path)
    total_words = 0
    total_off_page = 0
    total_word_overlap = 0
    total_text_img_coll = 0

    for page in doc:
        words = page.get_text("words")
        wrects = [fitz.Rect(w[:4]) for w in words if str(w[4]).strip()]
        total_words += len(wrects)
        for r in wrects:
            if r.x0 < page.rect.x0 or r.y0 < page.rect.y0 or r.x1 > page.rect.x1 or r.y1 > page.rect.y1:
                total_off_page += 1
        for i in range(len(wrects)):
            for j in range(i + 1, len(wrects)):
                if _overlap_ratio(wrects[i], wrects[j]) > overlap_threshold:
                    total_word_overlap += 1

        d = page.get_text("dict")
        txt = [fitz.Rect(b["bbox"]) for b in d["blocks"] if b.get("type", 0) == 0 and "bbox" in b]
        imgs = [fitz.Rect(b["bbox"]) for b in d["blocks"] if b.get("type", 0) == 1 and "bbox" in b]
        page_area = max(1e-9, page.rect.get_area())
        imgs = [
            im for im in imgs
            if (im.get_area() / page_area) < fullpage_area_ratio and not _is_decorative_raster(im, page_area)
        ]
        for t in txt:
            if any(_overlap_ratio(t, im) > text_image_threshold for im in imgs):
                total_text_img_coll += 1
    doc.close()

    return {
        "total_words": total_words,
        "off_page_words": total_off_page,
        "word_overlaps": total_word_overlap,
        "text_img_collisions": total_text_img_coll,
    }


def publication_qa(source_pages, translated_pages, pdf_path, coverage_report=None, target_lang="fr", original_image_paths=None):
    coverage_report = coverage_report or {"summary": {}}
    coverage_summary = coverage_report.get("summary", {})
    rendered_text_report = (coverage_report.get("rendered_text_report") if isinstance(coverage_report, dict) else None) or {"summary": {}}
    rendered_summary = rendered_text_report.get("summary", {})
    english_leak = _english_leak_count(translated_pages, target_lang=target_lang)
    layout_metrics = _evaluate_layout_pdf(pdf_path)
    visual_min_score = float(os.getenv("PUBLICATION_QA_VISUAL_MIN_SCORE", "0.80"))
    max_word_overlaps = int(os.getenv("PUBLICATION_QA_MAX_WORD_OVERLAPS", "1"))
    max_text_img_collisions = int(os.getenv("PUBLICATION_QA_MAX_TEXT_IMAGE_COLLISIONS", "0"))
    max_coverage_warnings = int(os.getenv("PUBLICATION_QA_MAX_COVERAGE_WARNINGS", "0"))
    max_rendered_text_warnings = int(os.getenv("PUBLICATION_QA_MAX_RENDERED_TEXT_WARNINGS", "0"))
    max_english_leaks = int(os.getenv("PUBLICATION_QA_MAX_ENGLISH_LEAKS", "0"))

    visual_compare = None
    if original_image_paths and any(p for p in original_image_paths):
        try:
            visual_compare = compare_reconstruction(original_image_paths, pdf_path, dpi=150)
        except Exception as exc:
            visual_compare = {"error": str(exc)}

    overall_visual = None
    if isinstance(visual_compare, dict):
        overall_visual = ((visual_compare.get("aggregate") or {}).get("overall"))

    decisions = []
    if coverage_summary.get("missing_units", 0) > 0:
        decisions.append("missing_coverage_units")
    if coverage_summary.get("warning_units", 0) > max_coverage_warnings:
        decisions.append("high_coverage_warning_count")
    if rendered_summary.get("rendered_missing_units", 0) > 0:
        decisions.append("missing_rendered_text_units")
    if rendered_summary.get("rendered_warning_units", 0) > max_rendered_text_warnings:
        decisions.append("high_rendered_text_warning_count")
    if english_leak.get("flagged_units", 0) > max_english_leaks:
        decisions.append("english_leak_detected")
    if layout_metrics.get("word_overlaps", 0) > max_word_overlaps:
        decisions.append("word_overlap_detected")
    if layout_metrics.get("text_img_collisions", 0) > max_text_img_collisions:
        decisions.append("text_image_collision_detected")
    if overall_visual is not None and overall_visual < visual_min_score:
        decisions.append("visual_similarity_below_target")

    publication_ready = len(decisions) == 0
    return {
        "publication_ready": publication_ready,
        "blocking_reasons": decisions,
        "scores": {
            "content_coverage_score": coverage_summary.get("coverage_score", 0.0),
            "rendered_text_coverage_score": rendered_summary.get("rendered_coverage_score", 0.0),
            "english_leak_score": round(
                max(
                    0.0,
                    1.0 - (english_leak.get("flagged_units", 0) / max(1, english_leak.get("unit_count", 1))),
                ),
                4,
            ),
            "layout_fidelity_score": round(
                max(
                    0.0,
                    1.0
                    - (
                        (layout_metrics.get("word_overlaps", 0) * 0.15)
                        + (layout_metrics.get("text_img_collisions", 0) * 0.1)
                        + (layout_metrics.get("off_page_words", 0) * 0.05)
                    )
                    / max(1, layout_metrics.get("total_words", 1)),
                ),
                4,
            ),
            "visual_similarity_score": overall_visual,
        },
        "layout_metrics": layout_metrics,
        "english_leak": english_leak,
        "coverage_summary": coverage_summary,
        "rendered_text_summary": rendered_summary,
        "visual_compare": visual_compare,
    }
