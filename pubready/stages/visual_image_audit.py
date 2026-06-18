"""VisualImageAudit — regarde RÉELLEMENT l'image finale (cv2).

- zones protégées : doivent rester ~identiques (préservées) → sinon objet détruit.
- zones de texte source remplacé : doivent avoir CHANGÉ → sinon ancien texte visible (leak).
- diff heatmap + crops en échec exportés.
Obligatoire en mode publication (sinon visual_image_qa_missing côté gates).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from ..schema import StageAuditResult, Finding, EvidenceItem, OK, REVIEW, KO


@dataclass
class VisualImageAuditResult:
    image_qa_executed: bool = False
    old_text_visible: bool = False
    double_text_rendering: bool = False
    visual_overlap: bool = False
    duplicate_page_number: bool = False
    caption_anchor_error: bool = False
    excessive_ink_density_regions: list = field(default_factory=list)
    failed_crops: list = field(default_factory=list)
    score: float = 0.0
    blockers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def result_to_stage(result: VisualImageAuditResult | dict) -> StageAuditResult:
    r = result if isinstance(result, VisualImageAuditResult) else VisualImageAuditResult(**dict(result or {}))
    res = StageAuditResult(stage_name="visual_image", score=float(r.score if r.image_qa_executed else 0.0))
    if not r.image_qa_executed:
        res.hard_blockers.append("visual_image_qa_missing")
    if r.old_text_visible:
        res.hard_blockers.append("source_text_leak")
        res.findings.append(Finding(type="old_text_visible", severity=KO))
        res.score = min(res.score, 0.50)
    if r.double_text_rendering:
        res.hard_blockers.append("double_text_rendering")
        res.findings.append(Finding(type="double_text_rendering", severity=KO))
        res.score = min(res.score, 0.50)
    if r.visual_overlap:
        res.hard_blockers.append("text_text_overlap_critical")
    if r.duplicate_page_number:
        res.hard_blockers.append("duplicate_page_number")
    if r.caption_anchor_error:
        res.hard_blockers.append("caption_anchor_error")
    if r.excessive_ink_density_regions:
        res.findings.append(Finding(type="excessive_ink_density", severity=KO,
                                    detail={"regions": r.excessive_ink_density_regions}))
        res.score = min(res.score, 0.50)
    for b in r.blockers or []:
        if b not in res.hard_blockers:
            res.hard_blockers.append(b)
    res.hard_blockers = sorted(set(res.hard_blockers))
    res.status = KO if res.hard_blockers else (OK if res.score >= 0.95 else REVIEW)
    return res


def _scale(page: dict):
    wpt, hpt = page.get("width_pt"), page.get("height_pt")
    rw, rh = page.get("render_width_px"), page.get("render_height_px")
    if wpt and hpt and rw and rh:
        return rw / wpt, rh / hpt
    g = page or {}
    return g.get("scale_x_px_per_pt", 1.0) or 1.0, g.get("scale_y_px_per_pt", 1.0) or 1.0


def audit_page(plan: dict, normalized: dict, *, source_image_path: str,
               reconstructed_image_path: str, out_dir: str | None = None) -> StageAuditResult:
    if plan.get("visual_image_audit"):
        return result_to_stage(plan.get("visual_image_audit"))
    res = StageAuditResult(stage_name="visual_image")
    try:
        import cv2
        import numpy as np
    except Exception:
        res.score, res.status = 0.0, KO
        res.hard_blockers.append("visual_image_qa_missing")
        res.findings.append(Finding(type="cv2_unavailable", severity=REVIEW))
        return res
    if not (source_image_path and reconstructed_image_path
            and os.path.isfile(source_image_path) and os.path.isfile(reconstructed_image_path)):
        res.hard_blockers.append("final_render_missing"); res.status = KO; res.score = 0.0
        return res

    src = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
    rec = cv2.imread(reconstructed_image_path, cv2.IMREAD_GRAYSCALE)
    if src is None or rec is None:
        res.hard_blockers.append("final_render_missing"); res.status = KO; res.score = 0.0
        return res
    if rec.shape != src.shape:
        rec = cv2.resize(rec, (src.shape[1], src.shape[0]))
    H, W = src.shape[:2]
    diff = cv2.absdiff(src, rec)
    page = plan.get("page") or {}
    sx, sy = _scale(page)

    def crop_diff(bb):
        x0, y0, x1, y1 = (int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy))
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return float(diff[y0:y1, x0:x1].mean())

    # 1. zones protégées : diff faible attendue (préservé). diff élevée = détruit.
    prot_scores = []
    for r in plan.get("protected_regions") or []:
        reason = str(r.get("reason") or "").lower()
        if reason in {
            "page_number", "page_reference", "toc_page_reference",
            "toc_section_number", "caption_label", "caption_number",
            "preserved_text_exact", "preserve_text_exactly",
        }:
            continue
        b = r.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        d = crop_diff(b)
        if d is None:
            continue
        sim = max(0.0, 1.0 - d / 80.0)        # diff>80/255 = altération forte
        prot_scores.append(sim)
        if sim < 0.5:
            res.findings.append(Finding(type="object_destroyed", severity=KO,
                                        detail={"reason": r.get("reason"), "diff": round(d, 1)}))
            if "destroyed_non_text_object" not in res.hard_blockers:
                res.hard_blockers.append("destroyed_non_text_object")

    # 2. zones de texte source remplacé : diff élevée attendue. diff faible = leak.
    leak_scores = []
    for t in (plan.get("layers") or {}).get("translated_text") or []:
        if (t.get("role") or "") in {"formula_expression", "code_line", "code_block"}:
            continue
        bb = t.get("coverage_bbox") or t.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        d = crop_diff(bb)
        if d is None:
            continue
        changed = min(1.0, d / 12.0)          # diff<12/255 ≈ inchangé = ancien texte visible
        leak_scores.append(changed)
        if changed < 0.3 and (t.get("source_text") or "").strip():
            res.findings.append(Finding(type="old_text_visible", severity=KO, element_id=t.get("id"),
                                        detail={"diff": round(d, 1)}))
            if "source_text_leak" not in res.hard_blockers:
                res.hard_blockers.append("source_text_leak")

    prot_score = sum(prot_scores)/len(prot_scores) if prot_scores else 1.0
    leak_score = sum(leak_scores)/len(leak_scores) if leak_scores else 1.0
    res.score = round(min(prot_score, leak_score), 3)
    res.status = KO if res.hard_blockers else (OK if res.score >= 0.95 else REVIEW)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        try:
            hm = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(out_dir, "diff_heatmap.png"), hm)
            res.evidence.append(EvidenceItem(kind="image", path=os.path.join(out_dir, "diff_heatmap.png")))
        except Exception:
            pass
    return res
