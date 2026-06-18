"""Selective background builder.

Contract:
    cleanbg/background is not an empty page. It is the visual substrate.

It may keep non-text visual material:
    - photos/images;
    - diagram shapes/arrows/lines/boxes;
    - non-text chart graphics;
    - decorative visual objects.

It must remove:
    - every text glyph, translatable or not;
    - page numbers / headers / footers / captions / labels;
    - formulas/equations/math zones;
    - code zones;
    - text-like special zones.

All removed content is later restored through TextOp, exact text preservation, or
PreservationOp. Non-text visuals may stay in cleanbg.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

try:
    from source_ownership import build_source_ownership
except Exception:  # pragma: no cover
    build_source_ownership = None


BACKGROUND_ONLY_STATES = {"background_only"}
TEXT_LEVELS = {"block", "line", "phrase", "span", "word", "char"}

TEXTUAL_MARKERS = {
    "text", "line", "phrase", "span", "word", "char",
    "caption", "label", "title", "subtitle", "header", "footer",
    "page_number", "page_reference", "toc", "index",
    "preserved_text_exact", "preserve_text_exactly",
}

SPECIAL_TEXTLIKE_MARKERS = {
    "formula", "equation", "math", "symbolic", "chemical_formula",
    "code", "algorithm",
}

NON_TEXT_VISUAL_MARKERS = {
    "image", "photo", "figure", "diagram", "chart", "graph", "plot",
    "logo", "watermark", "stamp", "seal", "visual",
}


def _valid_box(b) -> bool:
    return isinstance(b, (list, tuple)) and len(b) == 4 and float(b[2]) > float(b[0]) and float(b[3]) > float(b[1])


def _norm_box(b) -> list[float] | None:
    if not _valid_box(b):
        return None
    return [float(x) for x in b]


def _bbox_from(obj: dict | None):
    if not isinstance(obj, dict):
        return None
    for key in ("patch_bbox", "coverage_bbox", "layout_bbox", "bbox", "anchor_bbox"):
        nb = _norm_box(obj.get(key))
        if nb:
            return nb
    rt = obj.get("render_target") or {}
    for key in ("patch_bbox", "coverage_bbox", "layout_bbox", "bbox", "anchor_bbox"):
        nb = _norm_box(rt.get(key))
        if nb:
            return nb
    g = obj.get("geometry") or {}
    nb = _norm_box(g.get("bbox"))
    if nb:
        return nb
    return None


def _text_of(obj: dict | None) -> str:
    if not isinstance(obj, dict):
        return ""
    c = obj.get("content") or {}
    return " ".join(str(
        c.get("text")
        or obj.get("text")
        or obj.get("source_text")
        or obj.get("translated_text")
        or ""
    ).split())


def _raw_signature(obj: dict | None) -> str:
    if not isinstance(obj, dict):
        return ""
    vals = []
    for k in (
        "state", "level", "unit_level", "type", "role", "reason",
        "region_type", "object_type", "object_class", "semantic_kind",
        "claim_type", "source", "preservation_mode", "translation_strategy",
        "render_policy",
    ):
        vals.append(str(obj.get(k) or ""))
    pol = obj.get("policy") or {}
    if isinstance(pol, dict):
        for k in ("preserve_reason", "non_translatable_reason", "translation_strategy", "render_policy"):
            vals.append(str(pol.get(k) or ""))
    c = obj.get("content") or {}
    if isinstance(c, dict):
        vals.append(str(c.get("text") or ""))
    vals.append(str(obj.get("text") or ""))
    return " ".join(vals).lower()


def _has_any(raw: str, markers: set[str]) -> bool:
    return any(m in raw for m in markers)


def _is_text_unit(obj: dict | None) -> bool:
    if not isinstance(obj, dict):
        return False
    level = str(obj.get("level") or obj.get("unit_level") or obj.get("type") or "").lower()
    if level in TEXT_LEVELS:
        return bool(_text_of(obj)) or level in {"word", "char"}
    raw = _raw_signature(obj)
    return bool(_text_of(obj)) and _has_any(raw, TEXTUAL_MARKERS)


def _is_formula_or_code(obj: dict | None) -> bool:
    return _has_any(_raw_signature(obj), SPECIAL_TEXTLIKE_MARKERS)


def _is_textlike_content(obj: dict | None) -> bool:
    raw = _raw_signature(obj)
    if _is_text_unit(obj):
        return True
    if _has_any(raw, SPECIAL_TEXTLIKE_MARKERS):
        return True
    if _has_any(raw, TEXTUAL_MARKERS):
        return True
    return False


def _page_scale(input_data: dict) -> tuple[float, float]:
    g = ((input_data or {}).get("page") or {}).get("geometry") or {}
    sx = float(g.get("scale_x_px_per_pt") or 0) or 0.0
    sy = float(g.get("scale_y_px_per_pt") or 0) or 0.0
    if not sx and g.get("render_width_px") and g.get("width"):
        sx = float(g["render_width_px"]) / max(1e-6, float(g["width"]))
    if not sy and g.get("render_height_px") and g.get("height"):
        sy = float(g["render_height_px"]) / max(1e-6, float(g["height"]))
    return (sx or 1.0), (sy or 1.0)


def _pt_to_px(b: list[float], sx: float, sy: float, w: int, h: int, pad_x: int = 6, pad_y: int = 3) -> list[int]:
    x0 = max(0, int(round(b[0] * sx)) - pad_x)
    y0 = max(0, int(round(b[1] * sy)) - pad_y)
    x1 = min(w, int(round(b[2] * sx)) + pad_x)
    y1 = min(h, int(round(b[3] * sy)) + pad_y)
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return [x0, y0, x1, y1]


def _area(b: list[float]) -> float:
    if not _valid_box(b):
        return 0.0
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _inter_area(a: list[float], b: list[float]) -> float:
    if not (_valid_box(a) and _valid_box(b)):
        return 0.0
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return ix * iy


def _overlap_ratio(a: list[float], b: list[float]) -> float:
    return _inter_area(a, b) / max(1e-6, _area(a))


def _iter_ownership(input_data: dict) -> dict[str, dict]:
    views = input_data.get("views") or {}
    existing = views.get("source_ownership")
    if isinstance(existing, dict) and existing:
        return existing
    if build_source_ownership is None:
        return {}
    try:
        return build_source_ownership(input_data)
    except Exception:
        return {}


def _collect_boxes_from_ownership(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    for entry in _iter_ownership(input_data).values():
        if not isinstance(entry, dict):
            continue
        state = str(entry.get("state") or "").lower()
        if state in BACKGROUND_ONLY_STATES:
            continue
        if not _is_textlike_content(entry):
            # preserved_visual=image/diagram/photo etc. stays in background
            # unless its descendants carry actual text bboxes collected elsewhere.
            continue
        nb = _norm_box(entry.get("bbox"))
        if nb:
            boxes.append(nb)
    return boxes


def _collect_boxes_from_all_text_units(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    for u in input_data.get("units") or []:
        if not isinstance(u, dict):
            continue
        if not _is_text_unit(u) and not _is_formula_or_code(u):
            continue
        nb = _bbox_from(u)
        if nb:
            boxes.append(nb)
    return boxes


def _collect_boxes_from_regions(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    for r in input_data.get("regions") or []:
        if not isinstance(r, dict):
            continue
        raw = _raw_signature(r)
        # Remove entire math/code/formula regions. Do NOT remove entire image/
        # figure/diagram regions; their text descendants are removed separately.
        remove_whole_region = _has_any(raw, SPECIAL_TEXTLIKE_MARKERS)
        remove_text_region = _has_any(raw, TEXTUAL_MARKERS) and not _has_any(raw, NON_TEXT_VISUAL_MARKERS)
        if not (remove_whole_region or remove_text_region):
            continue
        nb = _bbox_from(r)
        if nb:
            boxes.append(nb)
    return boxes


def _collect_boxes_from_preservation_plan(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    views = input_data.get("views") or {}
    plans = []
    plans.extend(input_data.get("preservation_plan") or [])
    plans.extend(views.get("preservation_plan") or [])
    for p in plans:
        if not isinstance(p, dict):
            continue
        if not _is_textlike_content(p):
            continue
        nb = _bbox_from(p)
        if nb:
            boxes.append(nb)
    return boxes


def _collect_boxes_from_projection(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    tr = input_data.get("translation_result") or {}
    for item in tr.get("projection") or []:
        if not isinstance(item, dict):
            continue
        if not _is_textlike_content(item):
            continue
        b = _bbox_from(item.get("reconstruction_unit") or item) or _bbox_from(item)
        if b:
            boxes.append(b)
    return boxes


def _collect_boxes_from_reconstruction_view(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    views = input_data.get("views") or {}
    for item in views.get("reconstruction_units") or input_data.get("reconstruction_units") or []:
        if not isinstance(item, dict):
            continue
        if not _is_textlike_content(item):
            continue
        b = _bbox_from(item)
        if b:
            boxes.append(b)
    return boxes


def _merge_close_row_boxes(boxes: list[list[float]], y_tol: float = 3.0, gap: float = 8.0) -> list[list[float]]:
    boxes = [b for b in boxes if _valid_box(b)]
    boxes.sort(key=lambda b: (b[1], b[0], b[2] - b[0]))
    out: list[list[float]] = []
    for b in boxes:
        placed = False
        for cur in out:
            y_overlap = max(0.0, min(cur[3], b[3]) - max(cur[1], b[1]))
            min_h = max(1e-6, min(cur[3] - cur[1], b[3] - b[1]))
            same_row = y_overlap / min_h >= 0.45 or abs(cur[1] - b[1]) <= y_tol
            close_x = b[0] <= cur[2] + max(gap, min_h * 1.5)
            if same_row and close_x:
                cur[0] = min(cur[0], b[0])
                cur[1] = min(cur[1], b[1])
                cur[2] = max(cur[2], b[2])
                cur[3] = max(cur[3], b[3])
                placed = True
                break
        if not placed:
            out.append(list(b))
    return out


def _dedupe_boxes(boxes: list[list[float]]) -> list[list[float]]:
    boxes = [b for b in boxes if _valid_box(b)]
    boxes.sort(key=lambda b: (b[1], b[0], -_area(b)))
    out: list[list[float]] = []
    for b in boxes:
        if any(_overlap_ratio(b, o) > 0.90 for o in out):
            continue
        out.append(list(b))
    return _merge_close_row_boxes(out)


def collect_background_purity_boxes(input_data: dict) -> list[list[float]]:
    """Return bboxes to erase from cleanbg.

    Despite the historical function name, this is selective:
    erase all text and text-like special content, keep non-text visual content.
    """
    boxes: list[list[float]] = []
    boxes.extend(_collect_boxes_from_ownership(input_data))
    boxes.extend(_collect_boxes_from_all_text_units(input_data))
    boxes.extend(_collect_boxes_from_regions(input_data))
    boxes.extend(_collect_boxes_from_preservation_plan(input_data))
    boxes.extend(_collect_boxes_from_projection(input_data))
    boxes.extend(_collect_boxes_from_reconstruction_view(input_data))
    return _dedupe_boxes(boxes)


def collect_text_cover_boxes(input_data: dict) -> list[list[float]]:
    return collect_background_purity_boxes(input_data)


def _global_background_color(img) -> tuple[int, int, int]:
    w, h = img.size
    pts = []
    margin = max(8, min(w, h) // 40)
    sample_rects = [
        (0, 0, margin, margin),
        (max(0, w - margin), 0, w, margin),
        (0, max(0, h - margin), margin, h),
        (max(0, w - margin), max(0, h - margin), w, h),
    ]
    px = img.load()
    for x0, y0, x1, y1 in sample_rects:
        for y in range(y0, y1, max(1, margin // 8)):
            for x in range(x0, x1, max(1, margin // 8)):
                r, g, b = px[x, y]
                if (r + g + b) / 3 >= 120:
                    pts.append((r, g, b))
    if not pts:
        return (255, 255, 255)
    return tuple(int(median([p[i] for p in pts])) for i in range(3))


def _local_fill_color(img, rect: list[int], fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    w, h = img.size
    x0, y0, x1, y1 = rect
    pad = max(6, min(24, (y1 - y0) * 3))
    ox0, oy0 = max(0, x0 - pad), max(0, y0 - pad)
    ox1, oy1 = min(w, x1 + pad), min(h, y1 + pad)
    px = img.load()
    pts = []

    def add(x: int, y: int):
        if 0 <= x < w and 0 <= y < h and not (x0 <= x <= x1 and y0 <= y <= y1):
            r, g, b = px[x, y]
            lum = (r + g + b) / 3
            if lum >= 145:
                pts.append((r, g, b))

    step = max(1, pad // 4)
    for x in range(ox0, ox1, step):
        add(x, oy0)
        add(x, max(0, oy1 - 1))
    for y in range(oy0, oy1, step):
        add(ox0, y)
        add(max(0, ox1 - 1), y)

    if len(pts) < 8:
        return fallback
    return tuple(int(median([p[i] for p in pts])) for i in range(3))


def build_deterministic_text_cover_background(input_data: dict, *, out_path: str) -> str | None:
    if Image is None:
        return None

    assets = input_data.get("assets") or {}
    src = assets.get("source_image_path")
    if not src or not Path(src).is_file():
        return None

    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    sx, sy = _page_scale(input_data)
    w, h = img.size
    boxes_pt = collect_background_purity_boxes(input_data)
    if not boxes_pt:
        return None

    fallback_color = _global_background_color(img)
    for b in boxes_pt:
        rect = _pt_to_px(b, sx, sy, w, h)
        color = _local_fill_color(img, rect, fallback_color)
        draw.rectangle(rect, fill=color)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)

    vl = input_data.setdefault("visual_layers", {})
    assets = input_data.setdefault("assets", {})
    vl["clean_background_path"] = str(out)
    vl["background_preview_path"] = str(out)
    vl["clean_background_verified"] = True
    vl["text_removed"] = True
    vl["source_content_removed"] = False
    vl["special_content_removed"] = True
    vl["background_strategy"] = "deterministic_background_text_special_purity_v1_1"
    vl["background_purity_expected_cover_count"] = len(boxes_pt)
    vl["background_purity_contract"] = {
        "mode": "visual_substrate_keep_non_text_visuals",
        "no_source_text": True,
        "no_preserved_text_exact": True,
        "no_formula_code_math": True,
        "keep_non_text_visuals": True,
        "cover_box_count": len(boxes_pt),
    }
    assets["background_clean_path"] = str(out)
    assets["background_preview_path"] = str(out)
    assets["background_clean_verified"] = True
    assets["text_removed"] = True
    assets["source_content_removed"] = False
    assets["special_content_removed"] = True
    return str(out)
