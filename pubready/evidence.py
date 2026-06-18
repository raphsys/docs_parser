"""Accès aux données d'ORIGINE pageprint pour comparer le reconstruit à la source
(style, texte, bbox par unité source)."""

from __future__ import annotations


def index_units(normalized: dict) -> dict:
    return {u.get("unit_id"): u for u in (normalized.get("units") or []) if isinstance(u, dict) and u.get("unit_id")}


def source_style(unit: dict) -> dict:
    """Style source pageprint normalisé (font/size/color/flags/alignment)."""
    s = (unit.get("visual") or {}).get("style") or {}
    flags = s.get("flags") or {}
    return {
        "font_family": s.get("font") or s.get("font_family"),
        "font_size_pt": s.get("font_size_pt") or s.get("size"),
        "color": s.get("color") or "#000000",
        "bold": bool(flags.get("bold") or s.get("bold")),
        "italic": bool(flags.get("italic") or s.get("italic")),
        "mono": bool(flags.get("monospace")),
        "serif": flags.get("serif"),
        "alignment": s.get("alignment") or (unit.get("understanding") or {}).get("alignment"),
    }


def source_text(unit: dict) -> str:
    return str((unit.get("content") or {}).get("text") or "").strip()


def source_bbox(unit: dict):
    return (unit.get("geometry") or {}).get("bbox")


def font_class_of(family: str | None, mono: bool = False, serif: bool | None = None) -> str:
    f = str(family or "").lower()
    if mono or any(k in f for k in ("mono", "courier", "consol", "code")):
        return "mono"
    if any(k in f for k in ("sans", "arial", "helvet", "calibri", "verdana", "dejavusans", "franklin", "gothic")):
        return "sans"
    if serif is False:
        return "sans"
    return "serif"
