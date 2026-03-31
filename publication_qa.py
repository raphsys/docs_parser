import os
import re

import fitz
from PIL import Image, ImageFilter

from visual_compare import compare_reconstruction


PIXEL_TO_POINT = 72.0 / 150.0
VISUAL_ANNOTATION_ROLES = {"diagram_label", "diagram_text_label"}
VISUAL_ANNOTATION_STRUCTURAL_ROLES = {
    "diagram_label",
    "chart_axis_label",
    "chart_tick_label",
    "chart_legend_label",
    "chart_series_label",
}
VISUAL_ANNOTATION_BAND_ROLES = {"annotation_band", "legend_band", "axis_band"}
VISUAL_ANNOTATION_GROUP_MODES = {
    "annotation_group",
    "chart_legend_group",
    "chart_axis_group",
    "chart_series_group",
}
LOCKED_EQUATION_ROLES = {"equation_inline", "equation_block"}


def _norm_lower(value):
    return (str(value or "")).strip().lower()


def _descriptor_value(entity, key, default=""):
    if not isinstance(entity, dict):
        return default
    if key in entity and entity.get(key) not in {None, ""}:
        return entity.get(key)
    descriptor = entity.get("descriptor")
    if isinstance(descriptor, dict):
        return descriptor.get(key, default)
    return default


def _entity_bbox_rect(entity):
    if not isinstance(entity, dict):
        return None
    bbox = entity.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return fitz.Rect([float(v) * PIXEL_TO_POINT for v in bbox])
    except Exception:
        return None


def _page_refs(page, fallback_index=0):
    if not isinstance(page, dict):
        return {
            "page_index": max(0, int(fallback_index or 0)),
            "page_number": max(1, int(fallback_index or 0) + 1),
        }
    raw_page_number = page.get("page")
    raw_page_index = page.get("page_index")
    descriptor = (page.get("layout_descriptor") or ((page.get("layout") or {}).get("layout_descriptor")) or {})
    descriptor_page_index = descriptor.get("page_id") if isinstance(descriptor, dict) else None
    descriptor_page_number = descriptor.get("page_number") if isinstance(descriptor, dict) else None

    try:
        page_index = int(raw_page_index)
    except Exception:
        page_index = None
    if page_index is None:
        page_index = max(0, int(fallback_index or 0))
    if page_index is None:
        try:
            page_index = int(descriptor_page_index)
        except Exception:
            page_index = None

    try:
        page_number = int(raw_page_number)
    except Exception:
        page_number = None
    if page_number is None:
        page_number = max(1, int(fallback_index or 0) + 1)
    if page_number is None:
        try:
            page_number = int(descriptor_page_number)
        except Exception:
            page_number = None
    if page_number is None:
        page_number = max(1, int(page_index) + 1)

    return {
        "page_index": max(0, int(page_index)),
        "page_number": max(1, int(page_number)),
    }


def _page_layout_descriptor_maps(page):
    descriptor = (page or {}).get("layout_descriptor")
    if not isinstance(descriptor, dict):
        descriptor = ((page or {}).get("layout") or {}).get("layout_descriptor")
    if not isinstance(descriptor, dict):
        return {}, {}, {}, {}
    element_map = {
        str(el.get("id")): el
        for el in (descriptor.get("elements") or [])
        if isinstance(el, dict) and el.get("id")
    }
    region_map = {
        str(rg.get("id")): rg
        for rg in (descriptor.get("regions") or [])
        if isinstance(rg, dict) and rg.get("id")
    }
    visual_text_model = descriptor.get("visual_text_model") or {}
    visual_object_map = {}
    for obj in visual_text_model.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        source_element_id = str(obj.get("source_element_id") or "")
        if source_element_id:
            visual_object_map[source_element_id] = obj
    visual_group_map = {
        str(gr.get("id")): gr
        for gr in (visual_text_model.get("groups") or [])
        if isinstance(gr, dict) and gr.get("id")
    }
    return element_map, region_map, visual_object_map, visual_group_map


def _visual_annotation_layout_class(entity, inherited=None):
    inherited = inherited or {}
    role = _norm_lower(entity.get("role") or inherited.get("role"))
    structural_role = _norm_lower(
        entity.get("descriptor_structural_role")
        or _descriptor_value(entity, "structural_role")
        or inherited.get("descriptor_structural_role")
    )
    band_role = _norm_lower(
        entity.get("descriptor_band_role")
        or _descriptor_value(entity, "band_role")
        or inherited.get("descriptor_band_role")
    )
    group_render_mode = _norm_lower(
        entity.get("descriptor_group_render_mode")
        or _descriptor_value(entity, "group_render_mode")
        or inherited.get("descriptor_group_render_mode")
    )
    attachment_target_id = _norm_lower(
        entity.get("descriptor_attachment_target_id")
        or _descriptor_value(entity, "attachment_target_id")
        or inherited.get("descriptor_attachment_target_id")
    )
    visual_text = entity.get("descriptor_visual_text")
    if not isinstance(visual_text, dict):
        visual_text = _descriptor_value(entity, "visual_text", {}) or inherited.get("descriptor_visual_text") or {}
    text_embedding_mode = _norm_lower((visual_text or {}).get("text_embedding_mode"))
    render_mode = _norm_lower(entity.get("render_mode") or inherited.get("render_mode"))
    if render_mode == "background_only":
        return ""
    is_visual_label = bool(
        role in VISUAL_ANNOTATION_ROLES
        or structural_role in VISUAL_ANNOTATION_STRUCTURAL_ROLES
    )
    if text_embedding_mode == "embedded_in_visual":
        return "visual_annotation_text"
    if is_visual_label and (
        band_role in VISUAL_ANNOTATION_BAND_ROLES
        or group_render_mode in VISUAL_ANNOTATION_GROUP_MODES
        or attachment_target_id in {"illustration_main", "chart_main"}
    ):
        return "visual_annotation_text"
    if role in {"title", "header", "figure_caption"} and (
        band_role in VISUAL_ANNOTATION_BAND_ROLES
        or group_render_mode in VISUAL_ANNOTATION_GROUP_MODES
    ):
        return "visual_annotation_text"
    return ""


def _collect_visual_annotation_regions(pages):
    regions_by_page = {}
    for page_idx, page in enumerate(pages or []):
        if not isinstance(page, dict):
            continue
        page_key = _page_refs(page, fallback_index=page_idx)["page_index"]
        page_regions = regions_by_page.setdefault(page_key, [])
        element_map, region_map, visual_object_map, visual_group_map = _page_layout_descriptor_maps(page)
        for block in page.get("blocks", []) or []:
            block_id = str(block.get("id") or "")
            descriptor_block = element_map.get(block_id) if block_id else None
            descriptor_region = None
            if isinstance(descriptor_block, dict):
                descriptor_region = region_map.get(str(descriptor_block.get("page_region_id") or ""))
            descriptor_visual_object = visual_object_map.get(block_id) if block_id else None
            descriptor_visual_group = None
            if isinstance(descriptor_visual_object, dict):
                descriptor_visual_group = visual_group_map.get(str(descriptor_visual_object.get("group_id") or ""))
            block_inherited = {
                "role": block.get("role"),
                "descriptor_structural_role": block.get("descriptor_structural_role") or _descriptor_value(block, "structural_role") or ((descriptor_block or {}).get("structural_role")),
                "descriptor_band_role": block.get("descriptor_band_role") or _descriptor_value(block, "band_role") or ((descriptor_block or {}).get("band_role")),
                "descriptor_group_render_mode": block.get("descriptor_group_render_mode") or _descriptor_value(block, "group_render_mode") or ((descriptor_block or {}).get("group_render_mode")),
                "descriptor_attachment_target_id": block.get("descriptor_attachment_target_id") or _descriptor_value(block, "attachment_target_id") or ((descriptor_block or {}).get("attachment_target_id")),
                "descriptor_visual_text": block.get("descriptor_visual_text") or _descriptor_value(block, "visual_text", {}) or ((descriptor_block or {}).get("visual_text")) or {},
                "render_mode": block.get("render_mode"),
            }
            if not block_inherited.get("descriptor_group_render_mode") and isinstance(descriptor_visual_group, dict):
                block_inherited["descriptor_group_render_mode"] = _norm_lower((descriptor_visual_group or {}).get("group_render_mode"))
            if not block_inherited.get("descriptor_band_role") and isinstance(descriptor_region, dict):
                block_inherited["descriptor_band_role"] = _norm_lower((descriptor_block or {}).get("band_role") or (descriptor_region or {}).get("type"))
            added = False
            for line in block.get("lines", []) or []:
                line_inherited = dict(block_inherited)
                line_inherited["role"] = line.get("role") or block_inherited.get("role")
                line_inherited["render_mode"] = line.get("render_mode") or block_inherited.get("render_mode")
                line_added = False
                for phrase in line.get("phrases", []) or []:
                    if _visual_annotation_layout_class(phrase, line_inherited) != "visual_annotation_text":
                        continue
                    rect = _entity_bbox_rect(phrase) or _entity_bbox_rect(line) or _entity_bbox_rect(block)
                    if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                        continue
                    page_regions.append(
                        {
                            "rect": rect,
                            "role": _norm_lower(phrase.get("role") or line_inherited.get("role")),
                            "layout_class": "visual_annotation_text",
                        }
                    )
                    line_added = True
                    added = True
                if line_added:
                    continue
                if _visual_annotation_layout_class(line, line_inherited) == "visual_annotation_text":
                    rect = _entity_bbox_rect(line) or _entity_bbox_rect(block)
                    if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                        page_regions.append(
                            {
                                "rect": rect,
                                "role": _norm_lower(line.get("role") or line_inherited.get("role")),
                                "layout_class": "visual_annotation_text",
                            }
                        )
                        added = True
            if added:
                continue
            if _visual_annotation_layout_class(block, block_inherited) == "visual_annotation_text":
                rect = _entity_bbox_rect(block)
                if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                    page_regions.append(
                        {
                            "rect": rect,
                            "role": _norm_lower(block.get("role")),
                            "layout_class": "visual_annotation_text",
                        }
                    )
    return regions_by_page


def _collect_locked_equation_regions(pages):
    regions_by_page = {}
    for page_idx, page in enumerate(pages or []):
        if not isinstance(page, dict):
            continue
        page_key = _page_refs(page, fallback_index=page_idx)["page_index"]
        page_regions = regions_by_page.setdefault(page_key, [])
        for block in page.get("blocks", []) or []:
            role = _norm_lower(block.get("role"))
            render_mode = _norm_lower(block.get("render_mode"))
            if role not in LOCKED_EQUATION_ROLES or render_mode != "background_only":
                continue
            rect = _entity_bbox_rect(block)
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            page_regions.append(
                {
                    "rect": rect,
                    "role": role,
                    "layout_class": "locked_equation_overlay",
                }
            )
    return regions_by_page


def _rect_center_inside(rect, zone):
    r = fitz.Rect(rect)
    z = fitz.Rect(zone)
    cx = (r.x0 + r.x1) * 0.5
    cy = (r.y0 + r.y1) * 0.5
    return z.x0 <= cx <= z.x1 and z.y0 <= cy <= z.y1


def _rect_matches_visual_annotation(rect, regions, min_overlap=0.35):
    if not regions:
        return False
    target = fitz.Rect(rect)
    for region in regions:
        zone = region.get("rect") if isinstance(region, dict) else region
        if not isinstance(zone, fitz.Rect):
            zone = fitz.Rect(zone)
        if _overlap_ratio(target, zone) >= min_overlap or _rect_center_inside(target, zone):
            return True
    return False


def _blurred_similarity(original_crop, reconstructed_crop):
    if original_crop.size != reconstructed_crop.size:
        reconstructed_crop = reconstructed_crop.resize(original_crop.size, Image.Resampling.BILINEAR)
    orig = original_crop.convert("RGB").filter(ImageFilter.GaussianBlur(radius=2.2)).resize((24, 24), Image.Resampling.BILINEAR)
    recon = reconstructed_crop.convert("RGB").filter(ImageFilter.GaussianBlur(radius=2.2)).resize((24, 24), Image.Resampling.BILINEAR)
    orig_bytes = orig.tobytes()
    recon_bytes = recon.tobytes()
    if not orig_bytes or not recon_bytes:
        return None
    diff = 0.0
    triplet_count = min(len(orig_bytes), len(recon_bytes)) // 3
    for idx in range(0, triplet_count * 3, 3):
        diff += (
            abs(orig_bytes[idx] - recon_bytes[idx])
            + abs(orig_bytes[idx + 1] - recon_bytes[idx + 1])
            + abs(orig_bytes[idx + 2] - recon_bytes[idx + 2])
        ) / 3.0
    mad = diff / max(1, triplet_count)
    return max(0.0, 1.0 - (mad / 255.0))


def _normalize_spaces(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _iter_translated_units(pages):
    for page_idx, page in enumerate(pages or []):
        if not isinstance(page, dict):
            continue
        refs = _page_refs(page, fallback_index=page_idx)
        if page.get("schema_version") == "layout.v2" and str(page.get("page_role", "")).strip().lower() == "toc":
            rows = ((page.get("toc") or {}).get("toc_rows") or [])
            for row in rows:
                label = _normalize_spaces(row.get("translated_label") or row.get("label") or "")
                if label:
                    yield {
                        "page_id": refs["page_number"],
                        "page_number": refs["page_number"],
                        "page_index": refs["page_index"],
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
                    layout_class = _visual_annotation_layout_class(
                        phrase,
                        {
                            "role": phrase.get("role") or line.get("role") or role,
                            "descriptor_structural_role": phrase.get("descriptor_structural_role") or line.get("descriptor_structural_role") or block.get("descriptor_structural_role") or _descriptor_value(block, "structural_role"),
                            "descriptor_band_role": phrase.get("descriptor_band_role") or line.get("descriptor_band_role") or block.get("descriptor_band_role") or _descriptor_value(block, "band_role"),
                            "descriptor_group_render_mode": phrase.get("descriptor_group_render_mode") or line.get("descriptor_group_render_mode") or block.get("descriptor_group_render_mode") or _descriptor_value(block, "group_render_mode"),
                            "descriptor_attachment_target_id": phrase.get("descriptor_attachment_target_id") or line.get("descriptor_attachment_target_id") or block.get("descriptor_attachment_target_id") or _descriptor_value(block, "attachment_target_id"),
                            "descriptor_visual_text": phrase.get("descriptor_visual_text") or line.get("descriptor_visual_text") or block.get("descriptor_visual_text") or _descriptor_value(block, "visual_text", {}),
                            "render_mode": phrase.get("render_mode") or line.get("render_mode") or block.get("render_mode"),
                        },
                    )
                    yield {
                        "page_id": refs["page_number"],
                        "page_number": refs["page_number"],
                        "page_index": refs["page_index"],
                        "role": phrase.get("role") or role,
                        "strategy": phrase.get("translation_strategy") or block.get("translation_strategy") or "semantic_reflow",
                        "unit_type": phrase.get("unit_type") or line.get("unit_type") or block.get("unit_type") or "",
                        "page_family": page.get("page_family") or ((page.get("layout") or {}).get("page_family")) or "",
                        "translatable": bool(phrase.get("translatable", block.get("translatable", True))),
                        "render_policy": phrase.get("render_policy") or line.get("render_policy") or block.get("render_policy") or "",
                        "render_mode": phrase.get("render_mode") or line.get("render_mode") or block.get("render_mode") or "",
                        "layout_class": layout_class,
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
                    "page_number": unit.get("page_number", unit["page_id"]),
                    "page_index": unit.get("page_index", 0),
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


def _evaluate_visual_annotation_fidelity(original_image_paths, translated_pages, pdf_path, dpi=150):
    regions_by_page = _collect_visual_annotation_regions(translated_pages)
    if not regions_by_page:
        return {
            "region_count": 0,
            "pages_evaluated": 0,
            "background_similarity_score": None,
            "per_page": [],
        }
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return {
            "region_count": 0,
            "pages_evaluated": 0,
            "background_similarity_score": None,
            "per_page": [],
        }
    page_scores = []
    total_regions = 0
    try:
        scale = float(dpi) / 72.0
        for page_idx, page in enumerate(doc):
            original_path = original_image_paths[page_idx] if page_idx < len(original_image_paths) else ""
            if not original_path or not os.path.exists(original_path):
                continue
            page_regions = regions_by_page.get(page_idx) or regions_by_page.get(page_idx + 1) or []
            if not page_regions:
                continue
            with Image.open(original_path).convert("RGB") as original_im:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                reconstructed_im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                crop_scores = []
                for region in page_regions:
                    rect = fitz.Rect(region.get("rect"))
                    pad_pt = max(4.0, min(16.0, rect.height * 0.35))
                    padded = fitz.Rect(rect.x0 - pad_pt, rect.y0 - pad_pt, rect.x1 + pad_pt, rect.y1 + pad_pt) & page.rect
                    if padded.get_area() <= 0:
                        continue
                    x0, y0, x1, y1 = [int(round(v * scale)) for v in (padded.x0, padded.y0, padded.x1, padded.y1)]
                    x0 = max(0, min(original_im.width, x0))
                    x1 = max(0, min(original_im.width, x1))
                    y0 = max(0, min(original_im.height, y0))
                    y1 = max(0, min(original_im.height, y1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    orig_crop = original_im.crop((x0, y0, x1, y1))
                    recon_crop = reconstructed_im.crop((x0, y0, x1, y1))
                    score = _blurred_similarity(orig_crop, recon_crop)
                    if score is None:
                        continue
                    crop_scores.append(score)
                    total_regions += 1
                if crop_scores:
                    page_scores.append(
                        {
                            "page": page_idx + 1,
                            "region_count": len(crop_scores),
                            "background_similarity_score": round(sum(crop_scores) / len(crop_scores), 4),
                        }
                    )
    finally:
        doc.close()
    aggregate = None
    if page_scores:
        aggregate = round(sum(entry["background_similarity_score"] for entry in page_scores) / len(page_scores), 4)
    return {
        "region_count": total_regions,
        "pages_evaluated": len(page_scores),
        "background_similarity_score": aggregate,
        "per_page": page_scores,
    }


def _evaluate_layout_pdf(
    pdf_path,
    translated_pages=None,
    overlap_threshold=0.25,
    text_image_threshold=0.10,
    fullpage_area_ratio=0.95,
):
    doc = fitz.open(pdf_path)
    total_words = 0
    total_off_page = 0
    total_word_overlap = 0
    total_text_img_coll = 0
    ignored_visual_word_overlaps = 0
    ignored_visual_text_img_coll = 0
    ignored_equation_text_img_coll = 0
    visual_annotation_word_count = 0
    visual_annotation_block_count = 0
    visual_regions_by_page = _collect_visual_annotation_regions(translated_pages)
    equation_regions_by_page = _collect_locked_equation_regions(translated_pages)

    for page_idx, page in enumerate(doc):
        page_regions = visual_regions_by_page.get(page_idx) or visual_regions_by_page.get(page_idx + 1) or []
        equation_regions = equation_regions_by_page.get(page_idx) or equation_regions_by_page.get(page_idx + 1) or []
        words = page.get_text("words")
        word_entries = []
        for w in words:
            if not str(w[4]).strip():
                continue
            rect = fitz.Rect(w[:4])
            is_visual_annotation = _rect_matches_visual_annotation(rect, page_regions, min_overlap=0.45)
            if is_visual_annotation:
                visual_annotation_word_count += 1
            word_entries.append({"rect": rect, "is_visual_annotation": is_visual_annotation})
        wrects = [entry["rect"] for entry in word_entries]
        total_words += len(wrects)
        for r in wrects:
            if r.x0 < page.rect.x0 or r.y0 < page.rect.y0 or r.x1 > page.rect.x1 or r.y1 > page.rect.y1:
                total_off_page += 1
        for i in range(len(word_entries)):
            for j in range(i + 1, len(word_entries)):
                if _overlap_ratio(word_entries[i]["rect"], word_entries[j]["rect"]) > overlap_threshold:
                    if word_entries[i]["is_visual_annotation"] and word_entries[j]["is_visual_annotation"]:
                        ignored_visual_word_overlaps += 1
                        continue
                    total_word_overlap += 1

        d = page.get_text("dict")
        txt = []
        for b in d["blocks"]:
            if b.get("type", 0) != 0 or "bbox" not in b:
                continue
            rect = fitz.Rect(b["bbox"])
            is_visual_annotation = _rect_matches_visual_annotation(rect, page_regions, min_overlap=0.30)
            is_locked_equation = _rect_matches_visual_annotation(rect, equation_regions, min_overlap=0.30)
            if is_visual_annotation:
                visual_annotation_block_count += 1
            txt.append(
                {
                    "rect": rect,
                    "is_visual_annotation": is_visual_annotation,
                    "is_locked_equation": is_locked_equation,
                }
            )
        imgs = []
        for b in d["blocks"]:
            if b.get("type", 0) != 1 or "bbox" not in b:
                continue
            rect = fitz.Rect(b["bbox"])
            imgs.append(
                {
                    "rect": rect,
                    "is_locked_equation": _rect_matches_visual_annotation(rect, equation_regions, min_overlap=0.30),
                }
            )
        page_area = max(1e-9, page.rect.get_area())
        imgs = [
            im for im in imgs
            if (im["rect"].get_area() / page_area) < fullpage_area_ratio and not _is_decorative_raster(im["rect"], page_area)
        ]
        for t in txt:
            overlapping_imgs = [im for im in imgs if _overlap_ratio(t["rect"], im["rect"]) > text_image_threshold]
            if overlapping_imgs:
                if t["is_visual_annotation"]:
                    ignored_visual_text_img_coll += 1
                    continue
                if t["is_locked_equation"] or all(im["is_locked_equation"] for im in overlapping_imgs):
                    ignored_equation_text_img_coll += 1
                    continue
                total_text_img_coll += 1
    doc.close()

    return {
        "total_words": total_words,
        "off_page_words": total_off_page,
        "word_overlaps": total_word_overlap,
        "text_img_collisions": total_text_img_coll,
        "ignored_visual_annotation_word_overlaps": ignored_visual_word_overlaps,
        "ignored_visual_annotation_text_img_collisions": ignored_visual_text_img_coll,
        "ignored_locked_equation_text_img_collisions": ignored_equation_text_img_coll,
        "visual_annotation_word_count": visual_annotation_word_count,
        "visual_annotation_block_count": visual_annotation_block_count,
        "visual_annotation_region_count": sum(len(v) for v in visual_regions_by_page.values()),
        "locked_equation_region_count": sum(len(v) for v in equation_regions_by_page.values()),
    }


def publication_qa(source_pages, translated_pages, pdf_path, coverage_report=None, target_lang="fr", original_image_paths=None):
    coverage_report = coverage_report or {"summary": {}}
    coverage_summary = coverage_report.get("summary", {})
    rendered_text_report = (coverage_report.get("rendered_text_report") if isinstance(coverage_report, dict) else None) or {"summary": {}}
    rendered_summary = rendered_text_report.get("summary", {})
    english_leak = _english_leak_count(translated_pages, target_lang=target_lang)
    layout_metrics = _evaluate_layout_pdf(pdf_path, translated_pages=translated_pages)
    visual_min_score = float(os.getenv("PUBLICATION_QA_VISUAL_MIN_SCORE", "0.80"))
    visual_annotation_min_score = float(os.getenv("PUBLICATION_QA_VISUAL_ANNOTATION_MIN_SCORE", "0.0"))
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
    visual_annotation_regions = _evaluate_visual_annotation_fidelity(
        original_image_paths or [],
        translated_pages,
        pdf_path,
        dpi=150,
    )
    visual_annotation_score = visual_annotation_regions.get("background_similarity_score")

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
    if (
        visual_annotation_min_score > 0.0
        and visual_annotation_score is not None
        and visual_annotation_score < visual_annotation_min_score
    ):
        decisions.append("visual_annotation_fidelity_below_target")

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
        "visual_annotation_regions": visual_annotation_regions,
        "visual_compare": visual_compare,
    }
