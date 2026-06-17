"""Raster debug backend (PNG). NOT the final WYSIWYG output — used for overlays,
contact sheets and visual debug. The vector output is backends/pdf_vector.py."""

from __future__ import annotations

from ..render_backend import reconstruct_to_png, render_reconstructed_page

__all__ = ["reconstruct_to_png", "render_reconstructed_page", "render"]


def render(plan: dict, output_path: str, *, source_image_path: str | None = None) -> dict:
    for op in plan.get("render_ops") or []:
        if op.get("op_type") == "background":
            from ..render_ops import assert_publication_background_allowed
            src_path = source_image_path or ((plan.get("final_contract") or {}).get("background") or {}).get("source_image_path")
            assert_publication_background_allowed(op, source_image_path=src_path)
    src = source_image_path or ((plan.get("background") or [{}])[0].get("path"))
    ok = reconstruct_to_png(plan, src, output_path) if src else False
    return {"backend": "raster_debug", "output_path": output_path if ok else None, "ok": ok}
