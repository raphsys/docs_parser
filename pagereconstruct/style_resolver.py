"""Resolve a usable typographic style for every translated text unit.

Directive §9: no translated_text unit must reach the backend with style = {}.
Fallback chain (most → least reliable):
  1. reconstruction_unit.style
  2. reconstruction_plan_item.style / style_source_unit_id
  3. reconstruction_unit.style_source_unit_id
  4. source units (dominant span style)
  5. style_system dominant body style
  6. controlled default (serif 10pt) — flagged low confidence
"""

from __future__ import annotations

from .font_resolver_bridge import apply_font_class
from .font_size_sanitizer import sanitize as sanitize_font_size

_IGNORED = {"flags", "font_size_unit", "font_size_px"}

DEFAULT_STYLE = {
    "font_family": None, "font_size_pt": 10.0, "color": "#000000", "fill_color": "#000000",
    "background_color": None,
    "flags": {"bold": False, "italic": False, "serif": True, "monospace": False},
    "line_height_pt": 12.0, "alignment": "left",
    "style_source": "default", "style_source_unit_id": None, "confidence": 0.2,
}


def _is_real(style: dict) -> bool:
    return any(v is not None for k, v in (style or {}).items() if k not in _IGNORED)


def _dominant_style(unit: dict, unit_index: dict) -> dict:
    style = (unit.get("visual") or {}).get("style") or {}
    if _is_real(style):
        return style
    for cid in unit.get("children_ids") or []:
        child = unit_index.get(cid)
        if child:
            cs = _dominant_style(child, unit_index)
            if _is_real(cs):
                return cs
    return style


def _finalize(style: dict, *, source: str, source_id, confidence: float) -> dict:
    flags = dict(style.get("flags") or {})
    size = style.get("font_size_pt") or DEFAULT_STYLE["font_size_pt"]
    return {
        "font_family": style.get("font_family"),
        "font_size_pt": float(size),
        "color": style.get("color") or "#000000",
        "fill_color": style.get("fill_color") or style.get("color") or "#000000",
        "background_color": style.get("background_color"),
        "flags": {
            "bold": bool(flags.get("bold")),
            "italic": bool(flags.get("italic")),
            "serif": bool(flags.get("serif", True)),
            "monospace": bool(flags.get("monospace")),
        },
        "line_height_pt": float(style.get("line_height_pt") or size * 1.2),
        "alignment": style.get("alignment") or "left",
        "style_source": source,
        "style_source_unit_id": source_id,
        "confidence": confidence,
    }


def resolve_style(reconstruction_unit: dict, recon_plan_item: dict | None, unit_index: dict,
                  style_system: dict | None = None, *, role: str | None = None, line_bbox=None) -> dict:
    """Resolve style, then normalise font class and repair absurd font sizes."""
    style = _resolve_base(reconstruction_unit, recon_plan_item, unit_index, style_system)
    findings = []
    style["font_size_pt_raw"] = style.get("font_size_pt")
    apply_font_class(style)  # serif/sans/mono from font_family
    resolved_size, size_findings = sanitize_font_size(style.get("font_size_pt"), line_bbox, role)
    style["font_size_pt"] = resolved_size
    style["font_size_pt_resolved"] = resolved_size
    findings.extend(size_findings)
    style.setdefault("findings", []).extend(findings)
    return style


def _resolve_base(reconstruction_unit: dict, recon_plan_item: dict | None, unit_index: dict,
                  style_system: dict | None = None) -> dict:
    ru = reconstruction_unit or {}
    plan = recon_plan_item or {}

    if _is_real(ru.get("style") or {}):
        return _finalize(ru["style"], source="reconstruction_unit", source_id=ru.get("style_source_unit_id"), confidence=0.9)

    if _is_real(plan.get("style") or {}):
        return _finalize(plan["style"], source="reconstruction_plan", source_id=plan.get("style_source_unit_id"), confidence=0.85)

    for sid in (plan.get("style_source_unit_id"), ru.get("style_source_unit_id")):
        u = unit_index.get(sid) if sid else None
        if u:
            s = _dominant_style(u, unit_index)
            if _is_real(s):
                return _finalize(s, source="style_source_unit_id", source_id=sid, confidence=0.8)

    for sid in ru.get("source_unit_ids") or []:
        u = unit_index.get(sid)
        if u:
            s = _dominant_style(u, unit_index)
            if _is_real(s):
                return _finalize(s, source="source_unit", source_id=sid, confidence=0.75)

    dom_id = (style_system or {}).get("dominant_body_style_id")
    dom = ((style_system or {}).get("global_styles") or {}).get(dom_id) if dom_id else None
    if _is_real(dom or {}):
        return _finalize(dom, source="style_system", source_id=dom_id, confidence=0.5)

    return dict(DEFAULT_STYLE)
