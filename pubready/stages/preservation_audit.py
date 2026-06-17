"""Audit préservation : chaque zone spéciale critique (formule/code/image/table/
logo) est protégée et n'est PAS écrasée par du texte traduit ni un patch."""

from __future__ import annotations

from ..schema import StageAuditResult, ElementAudit, DimensionScore, Finding, OK, REVIEW, KO


def _ov_a(box, region):
    ix = max(0.0, min(box[2], region[2]) - max(box[0], region[0])) * max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    return ix / max(1e-6, (box[2]-box[0])*(box[3]-box[1]))


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    from pagereconstruct.composition.special_zone_preserver import classify_zones
    res = StageAuditResult(stage_name="preservation")
    zones = classify_zones(plan)
    if not zones:
        res.score = 1.0
        return res
    layers = plan.get("layers") or {}
    text_boxes = [t.get("layout_bbox") or t.get("bbox") for t in layers.get("translated_text") or []]
    text_boxes = [b for b in text_boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    patch_boxes = [p.get("bbox") for p in layers.get("patches") or [] if p.get("bbox")]

    auds = []
    for z in zones:
        # un texte traduit ou un patch qui recouvre ≥10% de la zone = atteinte.
        txt_ov = max((_ov_a(z.bbox, b) for b in text_boxes), default=0.0)
        patch_ov = max((_ov_a(z.bbox, p) for p in patch_boxes), default=0.0)
        hit = max(txt_ov, patch_ov)
        score = 1.0 if hit <= 0.10 else max(0.0, 1.0 - hit)
        status = OK if hit <= 0.10 else (KO if (z.critical and hit > 0.10) else REVIEW)
        dim = DimensionScore("preserved_intact", round(score, 3), z.reason, round(hit, 3), status, 1.0,
                             None if status == OK else "special_zone_overlap")
        el = ElementAudit(element_id=z.zone_id, level="block", role=z.reason, dimensions=[dim])
        el.combine(); auds.append(el)
        if status == KO:
            res.hard_blockers.append("special_zone_overlap")
            res.findings.append(Finding(type="special_zone_overlap", severity=KO, element_id=z.zone_id,
                                        detail={"reason": z.reason, "overlap": round(hit, 3)}))
    res.elements = auds
    res.score = round(sum(a.score for a in auds) / len(auds), 3)
    res.status = KO if res.hard_blockers else (OK if res.score >= 0.99 else REVIEW)
    return res
