"""Normalise translated_input_data into the four plans PAGERECONSTRUCT merges.

Source of truth (directive §5, §20):
  - translated text   = views.reconstruction_units (PAGETRANSLATE)
  - reconstruction    = views.reconstruction_plan  (PAGEPRINT)
  - preservation      = views.preservation_plan    (PAGEPRINT)
  - exclusion         = views.exclusion_plan        (PAGEPRINT)
  - geometry/style    = units[]
  - backgrounds       = visual_layers / assets
"""

from __future__ import annotations

from .errors import PageReconstructInputError


class PageReconstructInputAdapter:
    def normalize(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise TypeError("translated_input_data must be a dict")
        views = data.get("views") or {}
        reconstruction_units = views.get("reconstruction_units")
        if reconstruction_units is None:
            raise PageReconstructInputError(
                "translated_input_data.views.reconstruction_units is missing"
            )
        return {
            "schema_version": data.get("schema_version"),
            "page": data.get("page") or {},
            "page_intelligence": data.get("page_intelligence") or {},
            "document": data.get("document") or {},
            "assets": data.get("assets") or {},
            "visual_layers": data.get("visual_layers") or {},
            "units": data.get("units") or [],
            "regions": data.get("regions") or [],
            "style_system": data.get("style_system") or {},
            "translated_units": reconstruction_units or [],
            "reconstruction_plan": views.get("reconstruction_plan") or [],
            "preservation_plan": views.get("preservation_plan") or [],
            "exclusion_plan": views.get("exclusion_plan") or [],
            "quality": data.get("quality") or {},
            "translation_result": data.get("translation_result") or {},
        }
