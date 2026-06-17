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


def _sample_bg_color(img, b, pad: int = 6):
    """Robust local background colour around a patch bbox.

    The old median ring sampler was easily contaminated by neighbouring glyphs,
    leaving grey/black ghosts in clean backgrounds.  Prefer light/low-ink pixels
    in a wider ring; fall back to plain white on book-like pages.
    """
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

    def lum(p):
        return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]

    # Ignore glyph ink and antialiased dark pixels whenever possible.
    light = [p for p in pts if lum(p) >= 180 and max(p) - min(p) <= 80]
    sample = light if len(light) >= max(12, len(pts) * 0.20) else pts
    if sample is pts and sum(1 for p in pts if lum(p) >= 235) >= len(pts) * 0.25:
        return (255, 255, 255)
    m = len(sample) // 2
    return (sorted(p[0] for p in sample)[m], sorted(p[1] for p in sample)[m], sorted(p[2] for p in sample)[m])




def _crop_has_visible_ink(crop) -> bool:
    try:
        data = list(crop.getdata())
    except Exception:
        return False
    if not data:
        return False
    # White or very light background means no visual source text to copy.
    return any(sum(px[:3]) < 720 for px in data)


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

    # 2. Draw translated text via the role-specific renderer (dispatcher).
    from .renderer_dispatcher import dispatch
    for t in layers.get("translated_text") or []:
        if not (t.get("translated_text") or "").strip() and t.get("renderer") not in {"code", "formula"}:
            continue
        renderer = dispatch(t.get("renderer"), t.get("role"))
        renderer.render(draw, t, sx, sy, page_w_px=img.width)
    return img


def render_ops_to_png(plan: dict, source_image_path: str | None) -> Image.Image | None:
    """Pure executor (PNG) of frozen RenderOps. No dispatch/measure: ops carry
    resolved lines/size/font. Mirrors backends/pdf_vector.execute_ops."""
    ops = plan.get("render_ops") or []
    page = plan.get("page") or {}
    sx, sy = _scale(page)
    base_path = next((o.get("path") for o in ops if o.get("op_type") == "background" and o.get("path")), None) or source_image_path
    if not (base_path and Path(base_path).is_file()):
        return None
    img = Image.open(base_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    src_img = None
    if source_image_path and Path(source_image_path).is_file():
        src_img = Image.open(source_image_path).convert("RGB")
    for op in ops:
        t = op.get("op_type")
        if t == "patch":
            b = op.get("bbox")
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            px = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
            color = _hex_rgb(op["color"]) if op.get("color") else _sample_bg_color(img, px)
            draw.rectangle(px, fill=color)
        elif t == "preservation":
            b = op.get("bbox")
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            px = tuple(int(v) for v in (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
            method = op.get("method")
            # The best preservation is visual copy from the source crop.  Earlier
            # versions executed only copy_source_region and silently ignored
            # draw_text_exact/keep_pixels, which made numeric rows, page numbers
            # and exact labels disappear from the final PNG.
            if method in {"copy_source_region", "keep_pixels", "draw_text_exact"} and src_img is not None:
                if px[2] > px[0] and px[3] > px[1]:
                    crop = src_img.crop(px)
                    if method != "draw_text_exact" or _crop_has_visible_ink(crop):
                        img.paste(crop, (px[0], px[1]))
                        continue
            if method == "draw_text_exact" and op.get("text"):
                size = max(6, int(round(max(1.0, (b[3] - b[1])) * sy * 0.9)))
                try:
                    font = ImageFont.truetype(_FALLBACK, size)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((b[0] * sx, b[1] * sy), str(op.get("text")), fill=(20, 20, 20), font=font)
        elif t == "text":
            size = max(6, int(round(float(op.get("size_pt") or 10.0) * sy)))
            try:
                font = ImageFont.truetype(op.get("font_path") or _FALLBACK, size)
            except Exception:
                font = ImageFont.truetype(_FALLBACK, size)
            color = tuple(op.get("color") or (20, 20, 20))
            for ln in op.get("lines") or []:
                draw.text((ln["x"] * sx, ln["y_top"] * sy), ln["text"], fill=color, font=font)
    return img


def reconstruct_to_png(plan: dict, source_image_path: str, out_path: str) -> bool:
    # Contract-driven path: execute frozen RenderOps (zero dispatch in backend).
    if plan.get("render_ops"):
        img = render_ops_to_png(plan, source_image_path)
    else:
        img = render_reconstructed_page(plan, source_image_path)
    if img is None:
        return False
    img.save(out_path)
    return True
