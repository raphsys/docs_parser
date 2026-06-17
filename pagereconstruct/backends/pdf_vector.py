"""Vector PDF backend (PyMuPDF) — V1.

Executes a PageRenderPlan into a real vector PDF: background image, patch
rectangles, then translated text via insert_textbox using the resolved style
(font class -> base-14 font, size pt, colour, alignment). Import-guarded: if
PyMuPDF (fitz) is unavailable the backend reports unavailable instead of raising.
"""

from __future__ import annotations

import os

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


def _fontname_from_path(font_path: str) -> str:
    p = (font_path or "").lower()
    b, i = "bold" in p, ("italic" in p or "oblique" in p)
    if "mono" in p:
        return {(0, 0): "cour", (1, 0): "cobo", (0, 1): "coit", (1, 1): "cobi"}[(b, i)]
    if "sans" in p:
        return {(0, 0): "helv", (1, 0): "hebo", (0, 1): "heit", (1, 1): "hebi"}[(b, i)]
    return {(0, 0): "tiro", (1, 0): "tibo", (0, 1): "tiit", (1, 1): "tibi"}[(b, i)]


def execute_ops(ops: list, output_path: str, *, page_size_pt, source_image_path: str | None = None) -> dict:
    """Pure executor: paints RenderOps to a vector PDF. No dispatch, no measure,
    no decision — every position/size/font is already resolved in the ops."""
    if not _HAS_FITZ:
        return {"backend": "pdf_vector", "ok": False, "error": "pymupdf_unavailable", "output_path": None}
    w, h = float(page_size_pt[0] or 595.0), float(page_size_pt[1] or 842.0)
    findings = []
    doc = fitz.open()
    pg = doc.new_page(width=w, height=h)
    for op in ops or []:
        if op.get("op_type") == "background":
            from ..render_ops import assert_publication_background_allowed
            assert_publication_background_allowed(op, source_image_path=source_image_path)
        t = op.get("op_type")
        if t == "background" and op.get("path"):
            try:
                pg.insert_image(pg.rect, filename=op["path"])
            except Exception:
                findings.append({"type": "background_insert_failed"})
        elif t == "patch":
            b = op.get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                col = _rgb(op.get("color")) if op.get("color") else (1, 1, 1)
                pg.draw_rect(fitz.Rect(*b), color=None, fill=col)
        elif t == "preservation":
            b = op.get("bbox")
            if op.get("method") in {"copy_source_region", "keep_pixels"} and source_image_path and isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    pg.insert_image(fitz.Rect(*b), filename=source_image_path, clip=fitz.Rect(*b))
                except Exception:
                    findings.append({"type": "preservation_copy_failed"})
            elif op.get("method") == "draw_text_exact" and op.get("text") and isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    pg.insert_text((b[0], b[3]), op["text"], fontname="tiro", fontsize=max(6.0, (b[3] - b[1]) * 0.8))
                except Exception:
                    findings.append({"type": "preservation_text_failed"})
            # keep_pixels: déjà présent dans le fond propre -> no-op
        elif t == "text":
            size = float(op.get("size_pt") or 10.0)
            color = tuple(c / 255.0 for c in (op.get("color") or [0, 0, 0]))
            fpath = op.get("font_path") or ""
            # Use the REAL TTF (same DejaVu as the raster backend) so PDF == PNG;
            # fall back to a base-14 font and audit the substitution.
            use_fontfile = bool(fpath) and os.path.isfile(fpath)
            fname = _fontname_from_path(fpath)
            if not use_fontfile:
                findings.append({"type": "font_substitution", "unit_id": op.get("unit_id"), "to": fname})
            for ln in op.get("lines") or []:
                try:
                    if use_fontfile:
                        pg.insert_text((ln["x"], ln["y_top"] + size), ln["text"],
                                       fontfile=fpath, fontname="djv", fontsize=size, color=color)
                    else:
                        pg.insert_text((ln["x"], ln["y_top"] + size), ln["text"],
                                       fontname=fname, fontsize=size, color=color)
                except Exception as exc:
                    findings.append({"type": "text_render_failed", "unit_id": op.get("unit_id"), "message": str(exc)})
    doc.save(output_path)
    doc.close()
    return {"backend": "pdf_vector", "ok": True, "output_path": output_path, "findings": findings, "mode": "ops"}


def render(plan: dict, output_path: str) -> dict:
    """Ops-only: le backend EXÉCUTE les RenderOps gelées, ne dispatche jamais de
    renderer ni ne mesure (toutes les décisions sont dans les ops)."""
    if not _HAS_FITZ:
        return {"backend": "pdf_vector", "ok": False, "error": "pymupdf_unavailable", "output_path": None}
    page = plan.get("page") or {}
    w = float(page.get("width_pt") or 595.0)
    h = float(page.get("height_pt") or 842.0)
    ops = plan.get("render_ops")
    if not ops:
        return {"backend": "pdf_vector", "ok": False, "error": "no_render_ops", "output_path": None}
    src = ((plan.get("final_contract") or {}).get("background") or {}).get("source_image_path")
    return execute_ops(ops, output_path, page_size_pt=(w, h), source_image_path=src)
