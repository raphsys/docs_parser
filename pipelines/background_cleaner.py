"""Clean (text-removed) background generation for the canonical pipeline.

The legacy engine (ocr_server.process_page) inpainted the source text out of the
page to produce a clean master background, then PAGERECONSTRUCT painted the
translated text onto it. The canonical orchestrator skipped this step and handed
PAGERECONSTRUCT the raw source image, so every page reconstructed in
``source_background`` mode with ``source_text_leak_risk = high`` — never
publication-ready by construction.

This module restores that step: it inpaints the bounding boxes of the
*translatable* text units (the ones PAGERECONSTRUCT will repaint) while leaving
formula/code/preserved units and figures untouched (they keep their pixels).
"""

from __future__ import annotations

from pipelines.background_cover import build_deterministic_text_cover_background

import os

try:
    import cv2
    from text_removal_strategy import TextRemovalStrategy
    _STRATEGY = TextRemovalStrategy()
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    _STRATEGY = None


def _scale(input_data: dict) -> tuple[float, float]:
    """px/pt. La géométrie canonique est input_data['page']['geometry']
    (et NON page_intelligence.page_geometry — clé inexistante qui renvoyait 1.0
    et plaçait les zones d'inpaint à la mauvaise échelle → texte source non effacé).
    Repli : render_width_px / width."""
    g = ((input_data or {}).get("page") or {}).get("geometry") or {}
    sx = float(g.get("scale_x_px_per_pt") or 0) or 0.0
    sy = float(g.get("scale_y_px_per_pt") or 0) or 0.0
    if not sx and g.get("render_width_px") and g.get("width"):
        sx = float(g["render_width_px"]) / float(g["width"])
    if not sy and g.get("render_height_px") and g.get("height"):
        sy = float(g["render_height_px"]) / float(g["height"])
    return (sx or 1.0), (sy or 1.0)


def _unit_is_translatable(unit: dict) -> bool:
    policy = unit.get("policy") or {}
    if policy.get("translatable") is True or unit.get("translatable") is True:
        if policy.get("skip_translation") or (unit.get("constraints") or {}).get("skip_translation"):
            return False
        if policy.get("render_policy") == "background_only":
            return False
        if policy.get("translation_strategy") in {"exact_preserve", "keep_original", "background_only"}:
            return False
        return True
    content = unit.get("content") or {}
    # After PAGETRANSLATE, translated text on a source unit is a hard signal that
    # the corresponding source glyphs must be removed from the clean background.
    return bool(content.get("translated_text") or unit.get("translated_text"))


def _bbox_from_unit(unit: dict):
    return (unit.get("geometry") or {}).get("bbox") or unit.get("bbox")


def _pad_px(box: list[int], *, pad: int = 3) -> list[int]:
    return [max(0, int(box[0]) - pad), max(0, int(box[1]) - pad), int(box[2]) + pad, int(box[3]) + pad]


def _merge_px_boxes(boxes: list[list[int]], gap: int = 2) -> list[list[int]]:
    """Merge small boxes on the same visual row so Telea receives stable masks.

    Phrase-level masks are too tight and leave glyph edges.  We merge adjacent
    boxes with similar y ranges while keeping separate paragraphs/figures apart.
    """
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    boxes.sort(key=lambda b: (b[1], b[0]))
    merged: list[list[int]] = []
    for box in boxes:
        placed = False
        for cur in merged:
            y_overlap = max(0, min(cur[3], box[3]) - max(cur[1], box[1]))
            min_h = max(1, min(cur[3] - cur[1], box[3] - box[1]))
            close_x = box[0] <= cur[2] + max(gap, min_h * 2)
            if y_overlap / min_h >= 0.45 and close_x:
                cur[0] = min(cur[0], box[0]); cur[1] = min(cur[1], box[1])
                cur[2] = max(cur[2], box[2]); cur[3] = max(cur[3], box[3])
                placed = True
                break
        if not placed:
            merged.append(list(box))
    return merged


def _text_regions_px(input_data: dict, sx: float, sy: float, protected_boxes_pt: list | None = None) -> list[list[int]]:
    """Pixel bboxes du texte source à effacer.

    Source de vérité: unités traduisibles + segments réellement traduits.  On ne
    dépend pas seulement de ``views.translation_units`` ni seulement des phrases:
    les mauvaises classifications/segmentations partielles laissaient des
    paragraphes entiers dans le clean background.
    """
    units = input_data.get("units") or []
    protected_boxes_pt = protected_boxes_pt or []
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    candidate_boxes: list[list[int]] = []

    def add_bbox(b) -> None:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return
        bpt = [float(x) for x in b]
        if _overlaps_protected(bpt, protected_boxes_pt, 0.70):
            return
        x0, y0, x1, y1 = (bpt[0] * sx, bpt[1] * sy, bpt[2] * sx, bpt[3] * sy)
        if x1 > x0 and y1 > y0:
            candidate_boxes.append(_pad_px([int(x0), int(y0), int(x1), int(y1)]))

    # 1) Actual translation segments have priority: if the translator/reconstructor
    # will repaint it, the old glyphs must go.
    for seg in ((input_data.get("semantic_system") or {}).get("translation_segments") or []):
        if not isinstance(seg, dict):
            continue
        if seg.get("translation_mode") not in {None, "translate"}:
            continue
        add_bbox(seg.get("bbox"))
        for sid in seg.get("source_unit_ids") or []:
            unit = by_id.get(sid)
            if unit:
                add_bbox(_bbox_from_unit(unit))

    # 2) Then all translatable line boxes.  Line boxes remove glyphs more reliably
    # than phrase boxes and still preserve most page texture.
    for level in ("line", "phrase", "block"):
        for u in units or []:
            if not isinstance(u, dict) or u.get("level") != level:
                continue
            if not _unit_is_translatable(u):
                continue
            add_bbox(_bbox_from_unit(u))

    return _merge_px_boxes(candidate_boxes)


# Vrais visuels raster: l'inpaint Telea les ABÎMERAIT -> toujours protégés.
_STRONG_PROTECT = ("image", "figure", "chart", "photo", "formula", "code",
                   "diagram", "logo", "watermark")
# Zones « boîte » ambiguës: un encadré ombré / une table peut CONTENIR du texte
# traduisible. On ne les protège que si elles n'ont PAS de texte à remplacer
# (sinon il faut inpainter le texte — Telea préserve l'ombre/le filet).
_AMBIGUOUS_PROTECT = ("drawing", "non_text", "table", "panel", "box")


def _translatable_line_boxes_pt(units: list[dict]) -> list:
    out = []
    for u in units or []:
        if isinstance(u, dict) and u.get("level") == "line" and (u.get("policy") or {}).get("translatable"):
            b = (u.get("geometry") or {}).get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                out.append([float(x) for x in b])
    return out


def _contains_translatable_text(region_bbox, text_boxes, min_ratio: float = 0.40) -> bool:
    for t in text_boxes:
        area = max(1e-6, (t[2] - t[0]) * (t[3] - t[1]))
        ix = max(0.0, min(t[2], region_bbox[2]) - max(t[0], region_bbox[0])) * \
             max(0.0, min(t[3], region_bbox[3]) - max(t[1], region_bbox[1]))
        if ix / area >= min_ratio:
            return True
    return False


def _protected_boxes_pt(input_data: dict) -> list:
    """Bbox (pt) des vrais visuels à préserver de l'inpaint. Un panneau ombré qui
    contient du texte traduisible N'EST PAS protégé (on inpainte le texte, Telea
    garde l'ombre) — sinon l'ancien texte fuit sous la traduction."""
    text_boxes = _translatable_line_boxes_pt(input_data.get("units") or [])
    out = []
    for r in input_data.get("regions") or []:
        rt = str(r.get("region_type") or r.get("object_type") or "").lower()
        b = r.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        bb = [float(x) for x in b]
        if any(k in rt for k in _STRONG_PROTECT):
            out.append(bb)
        elif any(k in rt for k in _AMBIGUOUS_PROTECT) and not _contains_translatable_text(bb, text_boxes):
            out.append(bb)
    return out


def _overlaps_protected(b, protected_pt, min_ratio: float) -> bool:
    area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    for p in protected_pt:
        ix = max(0.0, min(b[2], p[2]) - max(b[0], p[0])) * max(0.0, min(b[3], p[3]) - max(b[1], p[1]))
        if ix / area >= min_ratio:
            return True
    return False


def build_clean_background(input_data: dict, *, out_path: str) -> str | None:
    # First choice: deterministic line-cover background. It is safer than
    # Telea/inpainting for dense book pages because it cannot leave old
    # glyphs under translated text. It also marks the background as verified.
    try:
        cover = build_deterministic_text_cover_background(input_data, out_path=out_path)
        if cover:
            return cover
    except Exception:
        pass
    """Inpaint translatable text out of the source image. Returns the clean
    background path, or None if unavailable (caller falls back to source)."""
    if _STRATEGY is None or cv2 is None:
        return None
    assets = input_data.get("assets") or {}
    src = assets.get("source_image_path")
    if not (src and os.path.isfile(src)):
        return None
    sx, sy = _scale(input_data)
    regions = _text_regions_px(input_data, sx, sy, _protected_boxes_pt(input_data))
    if not regions:
        return None
    try:
        from PIL import Image
        img = Image.open(src).convert("RGB")
        clean_bgr, _mask, _debug = _STRATEGY.remove(img, regions, mode="default")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, clean_bgr)
        return out_path if os.path.isfile(out_path) else None
    except Exception:
        return None
