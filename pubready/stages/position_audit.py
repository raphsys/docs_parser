"""Audit position GRANULAIRE par bloc: la bbox de rendu doit rester ancrée sur
la zone source (pageprint), sans dérive ni collision."""

from __future__ import annotations

from ..schema import StageAuditResult, ElementAudit, DimensionScore, Finding, OK, REVIEW, KO
from ..evidence import index_units, source_bbox


def _union(boxes):
    boxes = [b for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _iou(a, b):
    if not (a and b):
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix
    return ix / ua if ua > 0 else 0.0


def _centroid_drift(a, b):
    ca = ((a[0]+a[2])/2, (a[1]+a[3])/2); cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    return ((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2) ** 0.5


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    units = index_units(normalized)
    res = StageAuditResult(stage_name="block_layout")
    blocks = (plan.get("layers") or {}).get("translated_text") or []
    if not blocks:
        res.score = 1.0
        return res
    auds = []
    for b in blocks:
        rec = b.get("layout_bbox") or b.get("bbox")
        src = _union([source_bbox(units[s]) for s in (b.get("source_unit_ids") or []) if s in units]) or b.get("bbox")
        if not (isinstance(rec, (list, tuple)) and len(rec) == 4 and src):
            dims = [DimensionScore("position", 0.85, None, None, REVIEW, 1.0, "missing_geometry")]
        else:
            iou = _iou(rec, src)
            drift = _centroid_drift(rec, src)
            h = max(1.0, src[3]-src[1])
            # position OK si bien ancré (IoU élevé) ou dérive < hauteur de ligne.
            pos = max(iou, 1.0 - min(1.0, drift / (h*3)))
            dims = [DimensionScore("position", round(pos, 3), [round(x) for x in src], [round(x) for x in rec],
                                   OK if pos >= 0.6 else REVIEW if pos >= 0.3 else KO, 1.0,
                                   None if pos >= 0.6 else "position_drift")]
        el = ElementAudit(element_id=b.get("id"), level="block", role=b.get("role") or "",
                          source_text=b.get("source_text") or "", dimensions=dims)
        el.combine(); auds.append(el)
    res.elements = auds
    # collisions bloc/bloc + bloc/protégé (qualité de placement collective).
    boxes = [(b.get("id"), b.get("layout_bbox") or b.get("bbox")) for b in blocks]
    boxes = [(i, x) for i, x in boxes if isinstance(x, (list, tuple)) and len(x) == 4]
    bb_overlap = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            bb_overlap = max(bb_overlap, _ov_min(boxes[i][1], boxes[j][1]))
    prot = [r.get("bbox") for r in plan.get("protected_regions") or [] if r.get("bbox")]
    bp_overlap = max((_ov_a(b, p) for _, b in boxes for p in prot), default=0.0)
    if bb_overlap > 0.10:
        res.hard_blockers.append("block_text_overlap_critical")
        res.findings.append(Finding(type="block_block_overlap", severity=KO, detail={"ratio": round(bb_overlap, 3)}))
    if bp_overlap > 0.10:
        res.hard_blockers.append("block_protected_overlap")
        res.findings.append(Finding(type="block_protected_overlap", severity=KO, detail={"ratio": round(bp_overlap, 3)}))

    elem_score = sum(a.score for a in auds) / len(auds)
    coll_score = max(0.0, 1.0 - bb_overlap - bp_overlap)
    res.score = round(min(elem_score, coll_score), 3)
    res.status = KO if (res.hard_blockers or any(a.status == KO for a in auds)) else (REVIEW if res.score < 0.95 else OK)
    return res


def _ov_min(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    da = (a[2]-a[0])*(a[3]-a[1]); db = (b[2]-b[0])*(b[3]-b[1])
    return ix / max(1e-6, min(da, db))


def _ov_a(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
