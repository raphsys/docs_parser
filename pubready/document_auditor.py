"""Consolidation document : score ≠ moyenne simple. Une page ko / texte manquant
/ leak / objet détruit bloque le document."""

from __future__ import annotations

from .schema import DocumentPublicationReadyReport
from .gates import document_decision


def evaluate_document(page_reports, *, document_id: str = "document") -> DocumentPublicationReadyReport:
    status, ready, blocking = document_decision(page_reports)
    scores = [p.publication_ready_score for p in page_reports] or [0.0]
    base = sum(scores) / len(scores)
    review_pen = 0.03 * sum(1 for p in page_reports if p.status == "review")
    crit_pen = 0.10 * sum(1 for p in page_reports if p.hard_blockers)
    doc_score = max(0.0, base - review_pen - crit_pen)
    worst = sorted(page_reports, key=lambda p: p.publication_ready_score)[:3]
    return DocumentPublicationReadyReport(
        document_id=document_id, page_count=len(page_reports),
        status=status, publication_ready=ready,
        publication_ready_score=round(doc_score, 3),
        pages=list(page_reports), blocking_pages=list(blocking),
        worst_pages=[p.page_id for p in worst],
    )
