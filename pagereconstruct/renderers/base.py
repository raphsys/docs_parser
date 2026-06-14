"""Shared text layout/measure + base renderer.

A renderer first MEASURES (returns a RenderResult with real per-line geometry),
then paints from that measure. This lets the QA detect overlaps/overflow on the
actual rendered geometry, not just the plan (directive PR-Lot 3).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..ops import RenderResult

_FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
_FALLBACK = _FONT_DIR + "DejaVuSans.ttf"
_MEASURE_DRAW = ImageDraw.Draw(Image.new("RGB", (4, 4)))


def font_path(style: dict) -> str:
    flags = (style or {}).get("flags") or {}
    b, i = bool(flags.get("bold")), bool(flags.get("italic"))
    if flags.get("monospace"):
        base, sx = "DejaVuSansMono", {(0, 0): "", (1, 0): "-Bold", (0, 1): "-Oblique", (1, 1): "-BoldOblique"}
    elif flags.get("serif", True):
        base, sx = "DejaVuSerif", {(0, 0): "", (1, 0): "-Bold", (0, 1): "-Italic", (1, 1): "-BoldItalic"}
    else:
        base, sx = "DejaVuSans", {(0, 0): "", (1, 0): "-Bold", (0, 1): "-Oblique", (1, 1): "-BoldOblique"}
    p = f"{_FONT_DIR}{base}{sx[(b, i)]}.ttf"
    return p if Path(p).is_file() else _FALLBACK


def hex_rgb(value, default=(20, 20, 20)):
    s = str(value or "").lstrip("#")
    if len(s) == 6:
        try:
            return tuple(int(s[k:k + 2], 16) for k in (0, 2, 4))
        except ValueError:
            pass
    return default


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for w in str(text).split():
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def layout_text(text, px, style, *, align="left", min_ratio=0.86, allow_lines=None, expand_width_to=None):
    """Compute font + per-line boxes for `text` inside px. Pure (no painting)."""
    draw = _MEASURE_DRAW
    x0, y0, x1, y1 = px
    w, h = max(4, x1 - x0), max(4, y1 - y0)
    if expand_width_to:
        w = max(w, expand_width_to - x0)
    fpath = font_path(style)
    size_px = (style.get("font_size_pt") or 10.0) * style.get("_scale_y", 1.0)
    min_px = max(6, int(size_px * min_ratio))
    size = max(min_px, int(round(size_px)))
    font = ImageFont.truetype(fpath, size)
    lines = _wrap(draw, text, font, w)
    lh = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + max(1, int(size * 0.18))
    while size > min_px and (lh * len(lines) > h or (allow_lines and len(lines) > allow_lines)):
        size -= 1
        font = ImageFont.truetype(fpath, size)
        lines = _wrap(draw, text, font, w)
        lh = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + max(1, int(size * 0.18))
    boxes = []
    y = y0
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        x = x0 + (max(0, (w - tw) / 2) if align == "center" else (max(0, w - tw) if align == "right" else 0))
        boxes.append([x, y, x + tw, y + lh])
        y += lh
    overflow = lh * len(lines) > h + lh
    return {"font": font, "fpath": fpath, "size": size, "lines": lines, "line_boxes": boxes,
            "line_h": lh, "overflow": overflow, "align": align}


class BaseRenderer:
    renderer_name = "base"

    def text_for(self, unit) -> str:
        return (unit.get("translated_text") or "").strip()

    def layout_opts(self, style, page_w_px=None) -> dict:
        return {"align": (style.get("alignment") if style.get("alignment") in {"left", "center", "right"} else "left"),
                "min_ratio": 0.86}

    def _bbox(self, unit):
        return unit.get("layout_bbox") or unit.get("coverage_bbox") or unit.get("bbox")

    def measure(self, unit, sx, sy, page_w_px=None) -> RenderResult:
        style = dict(unit.get("style") or {})
        style["_scale_y"] = sy
        bbox = self._bbox(unit)
        rr = RenderResult(unit_id=unit.get("id"), renderer=self.renderer_name, planned_bbox=bbox)
        text = self.text_for(unit)
        if not text or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            rr.status = "review"
            rr.findings.append({"type": "missing_text_or_bbox"})
            return rr
        px = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
        lay = layout_text(text, px, style, **self.layout_opts(style, page_w_px))
        rr.line_boxes = lay["line_boxes"]
        rr.actual_line_boxes = lay["line_boxes"]
        rr.line_count = len(lay["lines"])
        rr.font_used = lay["fpath"]
        rr.font_size_used = lay["size"]
        rr.overflow = lay["overflow"]
        if lay["line_boxes"]:
            xs0 = min(b[0] for b in lay["line_boxes"]); ys0 = min(b[1] for b in lay["line_boxes"])
            xs1 = max(b[2] for b in lay["line_boxes"]); ys1 = max(b[3] for b in lay["line_boxes"])
            rr.actual_text_bbox = [xs0, ys0, xs1, ys1]
        if lay["overflow"]:
            rr.status = "review"
            rr.findings.append({"type": "overflow_unresolved", "unit_id": unit.get("id")})
        rr._lay = lay  # internal handoff to render
        return rr

    def render(self, draw, unit, sx, sy, page_w_px=None) -> list:
        rr = self.measure(unit, sx, sy, page_w_px)
        lay = getattr(rr, "_lay", None)
        if lay is None:
            return rr.findings
        style = dict(unit.get("style") or {})
        color = hex_rgb(style.get("color"))
        font = lay["font"]
        for ln, box in zip(lay["lines"], lay["line_boxes"]):
            draw.text((box[0], box[1]), ln, fill=color, font=font)
        return rr.findings
