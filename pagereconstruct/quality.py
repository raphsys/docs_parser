"""Reconstruction quality metrics derived from a PageRenderPlan (directive Lot 10)."""

from __future__ import annotations


def _findings(plan: dict) -> list:
    out = list(plan.get("findings") or [])
    for t in (plan.get("layers") or {}).get("translated_text") or []:
        out.extend((t.get("style") or {}).get("findings") or [])
    return out


def assess(plan: dict) -> dict:
    tt = (plan.get("layers") or {}).get("translated_text") or []
    findings = _findings(plan)

    def count(*types):
        return sum(1 for f in findings if f.get("type") in types)

    bg = (plan.get("background") or [{}])[0]
    styled = sum(1 for t in tt if (t.get("style") or {}).get("font_size_pt"))
    return {
        "text_units": len(tt),
        "styled_units": styled,
        "unresolved_style": count("unresolved_style"),
        "font_size_repaired": count("font_size_repaired_from_line_geometry", "font_size_inferred_from_line_geometry"),
        "layout_repaired": count("layout_bbox_repaired_from_coverage"),
        "overflow": count("overflow_unresolved", "table_cell_overflow"),
        "patch_protected_overlap": count("patch_protected_overlap"),
        "unknown_renderer": count("unknown_role_review"),
        "background_mode": bg.get("mode"),
        "source_text_leak_risk": bg.get("source_text_leak_risk"),
        "missing_background": bg.get("mode") == "blank_degraded",
    }
