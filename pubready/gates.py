"""Gates publication-ready : seuils stricts par étape + hard blockers + règles
de décision page/document. Tout est explicite et testable."""

from __future__ import annotations

# Seuils minimaux par étape (mode publication).
STAGE_THRESHOLDS = {
    "pageprint": 0.95,
    "pagetranslate": 0.98,
    "contract": 0.98,
    "background": 0.98,
    "text_removal": 0.98,
    "block_layout": 0.95,
    "intrablock": 0.95,
    "preservation": 0.99,
    "typography": 0.95,
    "render_ops": 1.0,
    "visual_image": 0.95,
}
TEXT_PRESENCE_REQUIRED = 1.0
NON_TEXT_PRESENCE_REQUIRED = 0.99
DOCUMENT_REQUIRED = 0.95
PAGE_MIN = 0.90

# Hard blockers (présence => page ko).
PAGE_HARD_BLOCKERS = {
    "missing_translatable_text", "translation_truncated", "protected_token_changed",
    "missing_clean_background", "source_text_leak", "destroyed_non_text_object",
    "source_background_forbidden", "clean_background_not_verified",
    "missing_text_removal_entry", "source_text_visible_under_translation",
    "source_unit_both_preserved_and_translated",
    "missing_preservation_op", "text_text_overlap_critical", "text_protected_overlap_critical",
    "patch_protected_overlap", "unresolved_style", "backend_hidden_source_background",
    "pdf_png_ops_divergence", "visual_image_qa_missing", "final_render_missing",
    "textop_missing_composition_id", "textop_from_raw_reconstruction_unit",
    "double_text_rendering", "duplicate_page_number", "caption_anchor_error",
    "reading_order_changed",
}


def stage_status(stage_name: str, score: float, hard_blockers: list) -> str:
    if hard_blockers:
        return "ko"
    thr = STAGE_THRESHOLDS.get(stage_name, 0.95)
    return "ok" if score >= thr else "review"


def page_decision(stage_results, *, mode: str = "publication") -> tuple[str, bool, list]:
    """Retourne (status, publication_ready, hard_blockers) pour une page."""
    blockers = []
    for st in stage_results:
        for hb in st.hard_blockers:
            if hb in PAGE_HARD_BLOCKERS:
                blockers.append(hb)
    any_ko = any(st.status == "ko" for st in stage_results)
    any_review = any(st.status == "review" for st in stage_results)

    # mode publication: audit image obligatoire.
    if mode == "publication":
        stages = {st.stage_name for st in stage_results}
        if "visual_image" not in stages:
            blockers.append("visual_image_qa_missing")

    status = "ko" if (blockers or any_ko) else ("review" if any_review else "ok")
    # publication-ready: tous les gates + aucun blocker.
    ready = (status == "ok" and not blockers
             and all(st.score >= STAGE_THRESHOLDS.get(st.stage_name, 0.95) for st in stage_results))
    return status, ready, sorted(set(blockers))


def document_decision(page_reports) -> tuple[str, bool, list]:
    """(status, publication_ready, blocking_pages). 1 page ko/texte manquant/leak
    /objet détruit → doc bloqué. Sinon ok si tout passe."""
    blocking = []
    for p in page_reports:
        if p.status == "ko" or p.hard_blockers or p.publication_ready_score < PAGE_MIN:
            blocking.append(p.page_id)
    status = "ko" if any(p.status == "ko" for p in page_reports) else (
        "review" if any(p.status == "review" for p in page_reports) else "ok")
    ready = (not blocking and all(p.publication_ready for p in page_reports))
    return status, ready, blocking
