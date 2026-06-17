"""Évaluateur de page : lance les audits granulaires (typo / traduction /
position) + réutilise la QA visuelle existante (overlap / fuite / non-texte),
combine selon les gates → PagePublicationReadyReport explicable."""

from __future__ import annotations

from .schema import PagePublicationReadyReport, StageAuditResult, Finding, OK, REVIEW, KO
from .gates import page_decision, STAGE_THRESHOLDS
from .stages import (typography_audit, translation_audit, position_audit,
                     pageprint_audit, contract_audit, background_audit, intrablock_audit,
                     preservation_audit, render_ops_audit, visual_image_audit)

# Poids de combinaison du score page (explicite).
_WEIGHTS = {
    "pageprint": 0.05, "contract": 0.06, "pagetranslate": 0.17, "typography": 0.15,
    "block_layout": 0.11, "intrablock": 0.11, "preservation": 0.09, "background": 0.11,
    "text_removal": 0.05, "render_ops": 0.05, "visual_image": 0.10,
}


def _visual_proxy(plan: dict) -> StageAuditResult:
    """visual_image PROXY (collision) quand les images réelles ne sont pas fournies."""
    from pagereconstruct.visual_qa import assess as vqa_assess
    v = vqa_assess(plan)
    s = v.get("scores") or {}
    vi_score = min(float(s.get("overlap", 1.0)), float(s.get("non_text_presence", 1.0)))
    vi = StageAuditResult(stage_name="visual_image", score=round(vi_score, 3))
    for hb in v.get("hard_blockers", []):
        if hb in {"patch_protected_overlap", "collision_ko"}:
            vi.hard_blockers.append("text_protected_overlap_critical")
    vi.status = KO if vi.hard_blockers else (OK if vi_score >= STAGE_THRESHOLDS["visual_image"] else REVIEW)
    return vi


def evaluate_page(plan: dict, normalized: dict, *, page_id: str = "", page_index: int = 0,
                  mode: str = "publication", source_image_path: str | None = None,
                  reconstructed_image_path: str | None = None, out_dir: str | None = None) -> PagePublicationReadyReport:
    stages = [
        pageprint_audit.audit_page(plan, normalized),      # pageprint
        translation_audit.audit_page(plan, normalized),    # pagetranslate
        contract_audit.audit_page(plan, normalized),       # contract
        background_audit.audit_page(plan, normalized, mode=mode),  # background granulaire
        background_audit.audit_text_removal_stage(plan),   # text removal ledger
        typography_audit.audit_page(plan, normalized),     # typography
        position_audit.audit_page(plan, normalized),       # block_layout
        intrablock_audit.audit_page(plan, normalized),     # intrablock
        preservation_audit.audit_page(plan, normalized),   # preservation
        render_ops_audit.audit_page(plan, normalized),     # render_ops
    ]
    # visual_image : RÉEL si images fournies, sinon proxy collision.
    if plan.get("visual_image_audit"):
        stages.append(visual_image_audit.result_to_stage(plan.get("visual_image_audit")))
    elif source_image_path and reconstructed_image_path:
        stages.append(visual_image_audit.audit_page(
            plan, normalized, source_image_path=source_image_path,
            reconstructed_image_path=reconstructed_image_path, out_dir=out_dir))
    else:
        vp = _visual_proxy(plan)
        if mode == "publication":
            # En publication, l'audit image RÉEL est obligatoire (proxy insuffisant).
            vp.hard_blockers.append("visual_image_qa_missing")
            vp.status = KO
        stages.append(vp)

    status, ready, blockers = page_decision(stages, mode=mode)
    # score combiné pondéré (explicable), borné par les blockers.
    by = {st.stage_name: st.score for st in stages}
    num = sum(_WEIGHTS[k] * by.get(k, 1.0) for k in _WEIGHTS)
    den = sum(_WEIGHTS.values())
    score = num / den
    if blockers or status == KO:
        score = min(score, 0.50 if "source_text_leak" in blockers or "double_text_rendering" in blockers else 0.60)
    elif status == REVIEW:
        score = min(score, 0.90)

    rep = PagePublicationReadyReport(
        page_id=page_id or (plan.get("page") or {}).get("page_role") or "page",
        page_index=page_index, status=status, publication_ready=ready,
        publication_ready_score=round(score, 3),
        stage_scores=by, hard_blockers=blockers, stages=stages,
    )
    for st in stages:
        rep.findings.extend(st.findings)
        rep.correction_suggestions.extend(st.suggestions)
    return rep
