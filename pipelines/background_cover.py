"""Deterministic text-cover background builder.

Every source text line that will be redrawn is covered in the background.
The fill color is estimated locally around the bbox.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None


def _bbox_from(obj: dict | None):
    if not isinstance(obj, dict):
        return None
    for key in ("patch_bbox", "coverage_bbox", "layout_bbox", "bbox"):
        b = obj.get(key)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return [float(x) for x in b]
    rt = obj.get("render_target") or {}
    for key in ("patch_bbox", "coverage_bbox", "layout_bbox", "bbox"):
        b = rt.get(key)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return [float(x) for x in b]
    g = obj.get("geometry") or {}
    b = g.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(x) for x in b]
    return None


def _text_of(obj: dict | None) -> str:
    if not isinstance(obj, dict):
        return ""
    c = obj.get("content") or {}
    return " ".join(str(c.get("text") or obj.get("text") or obj.get("source_text") or obj.get("translated_text") or "").split())


def _page_scale(input_data: dict) -> tuple[float, float]:
    g = ((input_data or {}).get("page") or {}).get("geometry") or {}
    sx = float(g.get("scale_x_px_per_pt") or 0) or 0.0
    sy = float(g.get("scale_y_px_per_pt") or 0) or 0.0
    if not sx and g.get("render_width_px") and g.get("width"):
        sx = float(g["render_width_px"]) / max(1e-6, float(g["width"]))
    if not sy and g.get("render_height_px") and g.get("height"):
        sy = float(g["render_height_px"]) / max(1e-6, float(g["height"]))
    return (sx or 1.0), (sy or 1.0)


def _pt_to_px(b: list[float], sx: float, sy: float, w: int, h: int, pad_x: int = 4, pad_y: int = 2) -> list[int]:
    x0 = max(0, int(round(b[0] * sx)) - pad_x)
    y0 = max(0, int(round(b[1] * sy)) - pad_y)
    x1 = min(w, int(round(b[2] * sx)) + pad_x)
    y1 = min(h, int(round(b[3] * sy)) + pad_y)
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return [x0, y0, x1, y1]


def _overlap_ratio(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _dedupe_boxes(boxes: list[list[float]]) -> list[list[float]]:
    boxes = [b for b in boxes if isinstance(b, list) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]]
    boxes.sort(key=lambda b: (b[1], b[0], b[2] - b[0]))
    out: list[list[float]] = []
    for b in boxes:
        if any(_overlap_ratio(b, o) > 0.88 and _overlap_ratio(o, b) > 0.55 for o in out):
            continue
        out.append(b)
    return out


def _collect_boxes_from_projection(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    tr = input_data.get("translation_result") or {}
    for item in tr.get("projection") or []:
        b = _bbox_from(item.get("reconstruction_unit") or item) or _bbox_from(item)
        if b:
            boxes.append(b)
    return boxes


def _collect_boxes_from_reconstruction_view(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    views = input_data.get("views") or {}
    for item in views.get("reconstruction_units") or input_data.get("reconstruction_units") or []:
        if not _text_of(item):
            continue
        b = _bbox_from(item)
        if b:
            boxes.append(b)
    return boxes


def _collect_visible_line_boxes(input_data: dict) -> list[list[float]]:
    boxes: list[list[float]] = []
    for u in input_data.get("units") or []:
        if not isinstance(u, dict) or u.get("level") != "line":
            continue
        if not _text_of(u):
            continue
        b = _bbox_from(u)
        if b:
            boxes.append(b)
    return boxes


def collect_text_cover_boxes(input_data: dict) -> list[list[float]]:
    boxes = []
    boxes.extend(_collect_boxes_from_projection(input_data))
    boxes.extend(_collect_boxes_from_reconstruction_view(input_data))
    boxes.extend(_collect_visible_line_boxes(input_data))
    return _dedupe_boxes(boxes)


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
    pad = max(4, min(16, (y1 - y0) * 2))
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

    step = max(1, pad // 3)
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
    boxes_pt = collect_text_cover_boxes(input_data)
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

    input_data.setdefault("visual_layers", {})["clean_background_path"] = str(out)
    input_data.setdefault("visual_layers", {})["background_preview_path"] = str(out)
    input_data.setdefault("visual_layers", {})["clean_background_verified"] = True
    input_data.setdefault("visual_layers", {})["text_removed"] = True
    input_data.setdefault("visual_layers", {})["background_strategy"] = "deterministic_text_cover_v1"
    input_data.setdefault("assets", {})["background_clean_path"] = str(out)
    input_data.setdefault("assets", {})["background_preview_path"] = str(out)
    input_data.setdefault("assets", {})["background_clean_verified"] = True
    input_data.setdefault("assets", {})["text_removed"] = True
    return str(out)
