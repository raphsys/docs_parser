"""Vector PDF backend (PyMuPDF) — V1.

Executes a PageRenderPlan into a real vector PDF: background image, patch
rectangles, then translated text via insert_textbox using the resolved style
(font class -> base-14 font, size pt, colour, alignment). Import-guarded: if
PyMuPDF (fitz) is unavailable the backend reports unavailable instead of raising.
"""

from __future__ import annotations

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:  # pragma: no cover
    _HAS_FITZ = False

_ALIGN = {"left": 0, "center": 1, "right": 2, "justify": 3}


def _fontname(style: dict) -> str:
    flags = (style or {}).get("flags") or {}
    b, i = bool(flags.get("bold")), bool(flags.get("italic"))
    if flags.get("monospace"):
        return {(0, 0): "cour", (1, 0): "cobo", (0, 1): "coit", (1, 1): "cobi"}[(b, i)]
    if flags.get("serif", True):
        return {(0, 0): "tiro", (1, 0): "tibo", (0, 1): "tiit", (1, 1): "tibi"}[(b, i)]
    return {(0, 0): "helv", (1, 0): "hebo", (0, 1): "heit", (1, 1): "hebi"}[(b, i)]


def _rgb(value):
    s = str(value or "#000000").lstrip("#")
    if len(s) == 6:
        try:
            return tuple(int(s[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
        except ValueError:
            pass
    return (0, 0, 0)


def is_available() -> bool:
    return _HAS_FITZ


def render(plan: dict, output_path: str) -> dict:
    if not _HAS_FITZ:
        return {"backend": "pdf_vector", "ok": False, "error": "pymupdf_unavailable", "output_path": None}
    page = plan.get("page") or {}
    w = float(page.get("width_pt") or 595.0)
    h = float(page.get("height_pt") or 842.0)
    findings = []
    doc = fitz.open()
    pg = doc.new_page(width=w, height=h)

    layers = plan.get("layers") or {}
    bg = (layers.get("background") or plan.get("background") or [{}])[0]
    if bg.get("path"):
        try:
            pg.insert_image(pg.rect, filename=bg["path"])
        except Exception:
            findings.append({"type": "background_insert_failed"})

    for p in layers.get("patches") or []:
        b = p.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        if p.get("protected_overlap_ratio", 0) > 0.5:
            continue
        color = _rgb(p.get("background_color")) if p.get("background_color") else (1, 1, 1)
        pg.draw_rect(fitz.Rect(*b), color=None, fill=color)

    # Same layout engine as the raster backend: dispatch -> measure (in pt space)
    # -> insert each measured line, so PNG and PDF stay consistent.
    from ..renderer_dispatcher import dispatch
    for t in layers.get("translated_text") or []:
        renderer = dispatch(t.get("renderer"), t.get("role"))
        rr = renderer.measure(t, 1.0, 1.0, page_w_px=w)
        lay = getattr(rr, "_lay", None)
        findings.extend(rr.findings)
        if lay is None or not lay.get("lines"):
            continue
        style = t.get("style") or {}
        size = float(lay.get("size") or style.get("font_size_pt") or 10.0)
        fname, color = _fontname(style), _rgb(style.get("color"))
        for ln, box in zip(lay["lines"], lay["line_boxes"]):
            try:
                pg.insert_text((box[0], box[1] + size), ln, fontname=fname, fontsize=size, color=color)
            except Exception as exc:
                findings.append({"type": "text_render_failed", "unit_id": t.get("id"), "message": str(exc)})

    doc.save(output_path)
    doc.close()
    return {"backend": "pdf_vector", "ok": True, "output_path": output_path, "findings": findings}
