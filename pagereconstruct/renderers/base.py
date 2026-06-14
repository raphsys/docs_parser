"""Shared text helpers + base renderer (raster). Renderers draw the translated
text of a unit according to its role contract and return findings."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageFont

_FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
_FALLBACK = _FONT_DIR + "DejaVuSans.ttf"


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


def wrap(draw, text, font, max_w):
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


def fit(draw, text, box_w, box_h, fpath, size_px, min_px, *, allow_lines=None):
    size = max(min_px, int(round(size_px)))
    while size >= min_px:
        font = ImageFont.truetype(fpath, size)
        lines = wrap(draw, text, font, box_w)
        lh = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + max(1, int(size * 0.18))
        line_ok = allow_lines is None or len(lines) <= allow_lines
        if (lh * len(lines) <= box_h and line_ok) or size == min_px:
            return font, lines, lh
        size -= 1
    return font, lines, lh


def draw_block(draw, text, px, style, *, align="left", min_ratio=0.86, allow_lines=None,
               expand_width_to=None) -> list:
    """Draw wrapped text inside px bbox; return findings (overflow/clip)."""
    findings = []
    x0, y0, x1, y1 = px
    w, h = max(4, x1 - x0), max(4, y1 - y0)
    if expand_width_to:
        w = max(w, expand_width_to - x0)
    fpath = font_path(style)
    size_px = (style.get("font_size_pt") or 10.0) * style.get("_scale_y", 1.0)
    min_px = max(6, int(size_px * min_ratio))
    font, lines, lh = fit(draw, text, w, h, fpath, size_px, min_px, allow_lines=allow_lines)
    if lh * len(lines) > h + lh:
        findings.append({"type": "overflow_unresolved"})
    color = hex_rgb(style.get("color"))
    y = y0
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        x = x0 + (max(0, (w - tw) / 2) if align == "center" else (max(0, w - tw) if align == "right" else 0))
        draw.text((x, y), ln, fill=color, font=font)
        y += lh
    return findings


class BaseRenderer:
    renderer_name = "base"

    def render(self, draw, unit, sx, sy, page_w_px=None) -> list:
        style = dict(unit.get("style") or {})
        style["_scale_y"] = sy
        bbox = unit.get("layout_bbox") or unit.get("coverage_bbox") or unit.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return [{"type": "missing_layout_bbox"}]
        px = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
        return self.draw(draw, unit, px, style, page_w_px)

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        return draw_block(draw, unit.get("translated_text") or "", px, style)
