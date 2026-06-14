"""Raster debug backend (PIL) — now style-aware.

Executes the PageRenderPlan: declared patches erase the source text zones
(respecting protected regions), then translated text is drawn with the RESOLVED
source style (serif/sans, bold/italic, source pt size, colour, line height,
alignment). It stays a raster debug backend; the vector PDF backend is a later
pass. The point here is typographic fidelity, not pixel perfection.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
_FALLBACK = _FONT_DIR + "DejaVuSans.ttf"


def _font_path(style: dict) -> str:
    flags = style.get("flags") or {}
    b, i = bool(flags.get("bold")), bool(flags.get("italic"))
    if flags.get("monospace"):
        base, sx = "DejaVuSansMono", {(0, 0): "", (1, 0): "-Bold", (0, 1): "-Oblique", (1, 1): "-BoldOblique"}
    elif flags.get("serif", True):
        base, sx = "DejaVuSerif", {(0, 0): "", (1, 0): "-Bold", (0, 1): "-Italic", (1, 1): "-BoldItalic"}
    else:
        base, sx = "DejaVuSans", {(0, 0): "", (1, 0): "-Bold", (0, 1): "-Oblique", (1, 1): "-BoldOblique"}
    p = f"{_FONT_DIR}{base}{sx[(b, i)]}.ttf"
    return p if Path(p).is_file() else _FALLBACK


def _hex_rgb(value, default=(20, 20, 20)):
    s = str(value or "").lstrip("#")
    if len(s) == 6:
        try:
            return tuple(int(s[k:k + 2], 16) for k in (0, 2, 4))
        except ValueError:
            pass
    return default


def _sample_bg_color(img, b, pad: int = 3):
    """Median colour of the ring just outside the bbox = local background."""
    W, H = img.size
    x0, y0, x1, y1 = (int(v) for v in b)
    pts = []
    for rx0, ry0, rx1, ry1 in ((x0, y0 - pad, x1, y0), (x0, y1, x1, y1 + pad),
                               (x0 - pad, y0, x0, y1), (x1, y0, x1 + pad, y1)):
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(W, rx1), min(H, ry1)
        if rx1 > rx0 and ry1 > ry0:
            pts.extend(img.crop((rx0, ry0, rx1, ry1)).getdata())
    if not pts:
        return (255, 255, 255)
    m = len(pts) // 2
    return (sorted(p[0] for p in pts)[m], sorted(p[1] for p in pts)[m], sorted(p[2] for p in pts)[m])


def _scale(page: dict):
    w_pt, h_pt = page.get("width_pt"), page.get("height_pt")
    rw, rh = page.get("render_width_px"), page.get("render_height_px")
    if w_pt and h_pt and rw and rh:
        return rw / w_pt, rh / h_pt
    return 1.0, 1.0


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


def _fit(draw, text, box_w, box_h, font_path, size_px, min_px):
    size = max(min_px, int(round(size_px)))
    while size >= min_px:
        font = ImageFont.truetype(font_path, size)
        lines = _wrap(draw, text, font, box_w)
        lh = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + max(1, int(size * 0.18))
        if lh * len(lines) <= box_h or size == min_px:
            return font, lines, lh
        size -= 1
    return font, lines, lh


def render_reconstructed_page(plan: dict, source_image_path: str) -> Image.Image | None:
    if not source_image_path or not Path(source_image_path).is_file():
        return None
    img = Image.open(source_image_path).convert("RGB")
    page = plan.get("page") or {}
    sx, sy = _scale(page)
    draw = ImageDraw.Draw(img)
    layers = plan.get("layers") or {}

    # 1. Execute declared patches (erase source text), skipping protected zones.
    patch_by_unit = {}
    for p in layers.get("patches") or []:
        patch_by_unit[p.get("unit_id")] = p
        if p.get("protected_overlap_ratio", 0) > 0.5:
            continue  # do not erase a mostly-protected zone
        b = p.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        px = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        # Sampled background colour (never a fixed white on a coloured page).
        color = _hex_rgb(p["background_color"]) if p.get("background_color") else _sample_bg_color(img, px)
        draw.rectangle(px, fill=color)

    # 2. Draw translated text with the resolved style.
    for t in layers.get("translated_text") or []:
        text = (t.get("translated_text") or "").strip()
        bbox = t.get("layout_bbox") or t.get("coverage_bbox") or t.get("bbox")
        if not text or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        style = t.get("style") or {}
        x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
        w, h = max(4, x1 - x0), max(4, y1 - y0)
        font_path = _font_path(style)
        size_px = (style.get("font_size_pt") or 10.0) * sy
        min_px = max(6, int(size_px * 0.86))
        font, lines, lh = _fit(draw, text, w, h, font_path, size_px, min_px)
        color = _hex_rgb(style.get("color"))
        align = style.get("alignment") or "left"
        y = y0
        for ln in lines:
            tw = draw.textlength(ln, font=font)
            if align == "center":
                x = x0 + max(0, (w - tw) / 2)
            elif align == "right":
                x = x0 + max(0, w - tw)
            else:
                x = x0
            draw.text((x, y), ln, fill=color, font=font)
            y += lh
    return img


def reconstruct_to_png(plan: dict, source_image_path: str, out_path: str) -> bool:
    img = render_reconstructed_page(plan, source_image_path)
    if img is None:
        return False
    img.save(out_path)
    return True
