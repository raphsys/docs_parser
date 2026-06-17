"""SpecialZonePreserver — classe les objets non traduisibles en 4 niveaux de
préservation (block / inline / page / background) et expose les zones à protéger.

Réutilise les protected_regions + la PreservationContract existants ; ne réinvente
pas la détection (cf. legacy registry immutable_overlays / FormulaItem → ADAPT).
"""

from __future__ import annotations

from dataclasses import dataclass, field

_INLINE_REASONS = ("formula_expression", "inline")
_PAGE_REASONS = ("page_number", "page_reference", "logo", "watermark", "publisher_mark",
                 "toc_page_reference", "toc_section_number", "caption_label", "caption_number")
_BG_REASONS = ("image", "figure", "drawing", "chart", "diagram", "background")
_BLOCK_REASONS = ("formula", "code", "equation", "table_grid", "table")

_CRITICAL = ("formula", "code", "image", "figure", "table_grid", "diagram", "logo", "equation")


@dataclass
class SpecialZone:
    zone_id: str
    bbox: list
    reason: str
    level: str            # inline | block | page | background
    critical: bool = False

    def to_dict(self):
        return {"zone_id": self.zone_id, "bbox": self.bbox, "reason": self.reason,
                "level": self.level, "critical": self.critical}


def _level_for(reason: str, bbox, page_h: float | None) -> str:
    r = reason.lower()
    if any(k in r for k in _INLINE_REASONS):
        return "inline"
    if any(k in r for k in _PAGE_REASONS):
        return "page"
    if any(k in r for k in _BG_REASONS):
        return "background"
    if any(k in r for k in _BLOCK_REASONS):
        return "block"
    return "block"


def classify_zones(plan: dict) -> list[SpecialZone]:
    """Classe les régions protégées + objets préservés en zones de 4 niveaux."""
    page_h = (plan.get("page") or {}).get("height_pt")
    zones: list[SpecialZone] = []
    seen = set()
    n = 0
    for r in plan.get("protected_regions") or []:
        b = r.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        reason = str(r.get("reason") or "protected")
        key = (round(b[0]), round(b[1]), round(b[2]), round(b[3]), reason)
        if key in seen:
            continue
        seen.add(key); n += 1
        zones.append(SpecialZone(zone_id=r.get("id") or f"zone_{n:04d}", bbox=[float(x) for x in b],
                                 reason=reason, level=_level_for(reason, b, page_h),
                                 critical=any(k in reason.lower() for k in _CRITICAL)))
    return zones
