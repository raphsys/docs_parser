"""Decide the reconstruction background and the source-text-leak risk (Lot 6).

The PageRenderPlan must carry its background explicitly so the backend does not
receive the source image as a hidden background. Priority:
  clean_background > source_background (with mandatory patches) > blank_degraded.
"""

from __future__ import annotations


def resolve_background(normalized: dict) -> dict:
    assets = normalized.get("assets") or {}
    visual_layers = normalized.get("visual_layers") or {}
    has_translated_text = bool(normalized.get("translated_units"))

    clean = (visual_layers.get("clean_background_path")
             or assets.get("background_clean_path")
             or assets.get("background_path"))
    source = assets.get("source_image_path")

    if clean:
        return {"mode": "clean_background", "path": clean,
                "source_text_leak_risk": "low", "findings": []}
    if source:
        risk = "high" if has_translated_text else "none"
        findings = ([{"type": "source_text_leak_risk", "level": "high",
                      "message": "source background still contains original text; patches required"}]
                    if has_translated_text else [])
        return {"mode": "source_background", "path": source,
                "source_text_leak_risk": risk, "findings": findings}
    return {"mode": "blank_degraded", "path": None, "source_text_leak_risk": "high",
            "findings": [{"type": "missing_background", "severity": "review"}]}
