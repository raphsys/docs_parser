"""Audit PAGEPRINT : pageprint fournit-il assez pour reconstruire la page ?"""

from __future__ import annotations

from ..schema import StageAuditResult, Finding, OK, REVIEW, KO


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    res = StageAuditResult(stage_name="pageprint")
    n = normalized or {}
    page = n.get("page") or {}
    geom = page.get("geometry") or (n.get("page_intelligence") or {}).get("page_geometry") or {}
    units = n.get("units") or []
    regions = n.get("regions") or []
    assets = n.get("assets") or {}

    checks = {
        "page_size": bool(geom.get("width") or geom.get("width_pt")),
        "units": len(units) > 0,
        "assets_source_image": bool(assets.get("source_image_path")),
        "regions": True,  # peut être vide légitimement
    }
    # hiérarchie présente
    levels = {u.get("level") for u in units if isinstance(u, dict)}
    checks["hierarchy"] = bool(levels & {"block", "line", "phrase"})
    # bboxes valides
    bad_bbox = sum(1 for u in units if isinstance(u, dict)
                   and (u.get("geometry") or {}).get("bbox")
                   and len((u.get("geometry") or {}).get("bbox")) != 4)
    checks["valid_bbox"] = bad_bbox == 0

    if not checks["page_size"]:
        res.hard_blockers.append("missing_page_size")
    if not checks["units"]:
        res.hard_blockers.append("missing_units")
    if not checks["valid_bbox"]:
        res.hard_blockers.append("invalid_bbox")
    if not checks["hierarchy"]:
        res.findings.append(Finding(type="missing_hierarchy", severity=REVIEW))

    res.score = round(sum(1 for v in checks.values() if v) / len(checks), 3)
    res.status = KO if res.hard_blockers else (OK if res.score >= 0.95 else REVIEW)
    return res
