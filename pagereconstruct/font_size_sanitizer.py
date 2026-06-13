"""Repair absurd extracted font sizes (directive Lot 5/A3).

The demo showed a median font_size_pt ~4.78 for book body text that is visually
~9-11pt: the extracted size is often a glyph metric, not the real font size.
When the size is far below the source line height, rebuild it from geometry.
"""

from __future__ import annotations

_BOUNDS = {
    "body_paragraph": (7.0, 14.0), "paragraph": (7.0, 14.0), "list_item": (7.0, 14.0),
    "author_bio": (7.0, 14.0), "index_subentry": (7.0, 14.0),
    "title": (9.0, 32.0), "section_heading": (9.0, 32.0), "subsection_heading": (9.0, 28.0),
    "subtitle": (9.0, 28.0),
    "table_body_cell": (5.5, 12.0), "table_header_cell": (5.5, 12.0), "table_numeric_cell": (5.5, 12.0),
    "diagram_label": (4.5, 12.0), "toc_entry_title": (6.0, 14.0),
}
_DEFAULT_BOUNDS = (5.0, 32.0)


def _line_height(line_bbox) -> float | None:
    if isinstance(line_bbox, (list, tuple)) and len(line_bbox) == 4:
        h = float(line_bbox[3]) - float(line_bbox[1])
        return h if h > 0 else None
    return None


def sanitize(font_size_pt, line_bbox, role: str | None = None) -> tuple[float, list]:
    """Return (resolved_size_pt, findings)."""
    findings = []
    lo, hi = _BOUNDS.get(str(role or ""), _DEFAULT_BOUNDS)
    line_h = _line_height(line_bbox)
    raw = float(font_size_pt) if font_size_pt else None

    # Only repair clearly absurd sizes (< 6pt) so legitimate sizes are untouched.
    if raw is None:
        resolved = (line_h * 0.85) if line_h else lo
        findings.append({"type": "font_size_inferred_from_line_geometry"})
    elif raw < 6.0:
        resolved = (line_h * 0.85) if line_h else max(lo, 8.5)
        findings.append({"type": "font_size_repaired_from_line_geometry",
                         "raw": round(raw, 2), "resolved": round(resolved, 2)})
    else:
        resolved = raw

    clamped = max(lo, min(hi, resolved))
    if abs(clamped - resolved) > 0.01:
        findings.append({"type": "font_size_clamped", "to": round(clamped, 2)})
    return round(clamped, 2), findings
