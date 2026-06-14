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

    bg = (plan.get("background") or [{}])[0]
    if bg.get("path"):
        try:
            pg.insert_image(pg.rect, filename=bg["path"])
        except Exception:
            findings.append({"type": "background_insert_failed"})

    layers = plan.get("layers") or {}
    for p in layers.get("patches") or []:
        b = p.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        if p.get("protected_overlap_ratio", 0) > 0.5:
            continue
        color = _rgb(p.get("background_color")) if p.get("background_color") else (1, 1, 1)
        pg.draw_rect(fitz.Rect(*b), color=None, fill=color)

    for t in layers.get("translated_text") or []:
        text = (t.get("translated_text") or "").strip()
        b = t.get("layout_bbox") or t.get("coverage_bbox") or t.get("bbox")
        if not text or not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        style = t.get("style") or {}
        size = float(style.get("font_size_pt") or 10.0)
        rc = doc  # keep ref
        try:
            leftover = pg.insert_textbox(
                fitz.Rect(*b), text, fontname=_fontname(style), fontsize=size,
                color=_rgb(style.get("color")), align=_ALIGN.get(style.get("alignment", "left"), 0),
            )
            if leftover < 0:
                findings.append({"type": "overflow_unresolved", "unit_id": t.get("id"), "severity": "review"})
        except Exception as exc:
            findings.append({"type": "text_render_failed", "unit_id": t.get("id"), "message": str(exc)})

    doc.save(output_path)
    doc.close()
    return {"backend": "pdf_vector", "ok": True, "output_path": output_path, "findings": findings}
