"""Reconstruction backends: raster_debug (PNG, debug) and pdf_vector (final)."""

from __future__ import annotations

from . import pdf_vector, raster_debug


def select_backend(name: str = "pdf_vector"):
    if name in {"pdf_vector", "pdf", "vector"} and pdf_vector.is_available():
        return pdf_vector
    return raster_debug


__all__ = ["raster_debug", "pdf_vector", "select_backend"]
