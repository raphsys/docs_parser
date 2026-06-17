"""Schémas de score publication-ready — GRANULAIRE et EXPLICABLE.

Hiérarchie d'audit (demande utilisateur):
    page → blocs → phrases → dimensions (typographie/traduction/position…)
chaque dimension comparée à l'ORIGINE pageprint, individuellement, puis combinée.

Un score n'est jamais un nombre nu : il porte attendu / observé / findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# statut commun
OK, REVIEW, KO = "ok", "review", "ko"


@dataclass
class DimensionScore:
    """Score d'UNE dimension (ex: typo.font_size, translation, position) sur un
    élément, en regard de l'origine pageprint."""
    name: str                       # font_family|font_size|color|bold|italic|alignment|indent|translation|position|presence
    score: float                    # 0..1
    expected: object = None         # valeur source (pageprint)
    observed: object = None         # valeur reconstruite
    status: str = OK                # ok | review | ko
    weight: float = 1.0
    finding: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ElementAudit:
    """Audit d'un élément (bloc ou phrase) sur toutes ses dimensions."""
    element_id: str
    level: str                      # block | phrase
    role: str = ""
    source_text: str = ""
    translated_text: str = ""
    dimensions: list = field(default_factory=list)   # [DimensionScore]
    children: list = field(default_factory=list)      # [ElementAudit] (phrases d'un bloc)
    score: float = 1.0
    status: str = OK

    def combine(self) -> float:
        dims = self.dimensions or []
        if dims:
            w = sum(d.weight for d in dims) or 1.0
            self.score = sum(d.score * d.weight for d in dims) / w
            self.status = (KO if any(d.status == KO for d in dims)
                           else REVIEW if any(d.status == REVIEW for d in dims) else OK)
        if self.children:
            cs = [c.combine() for c in self.children]
            self.score = min(self.score, sum(cs) / len(cs))
            if any(c.status == KO for c in self.children):
                self.status = KO
            elif self.status != KO and any(c.status == REVIEW for c in self.children):
                self.status = REVIEW
        return self.score

    def to_dict(self):
        return {"element_id": self.element_id, "level": self.level, "role": self.role,
                "source_text": self.source_text[:80], "translated_text": self.translated_text[:80],
                "score": round(self.score, 3), "status": self.status,
                "dimensions": [d.to_dict() for d in self.dimensions],
                "children": [c.to_dict() for c in self.children]}


@dataclass
class Finding:
    type: str
    severity: str = REVIEW          # ok | review | ko
    element_id: str | None = None
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class EvidenceItem:
    kind: str                       # crop | metric | image
    path: str | None = None
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class CorrectionSuggestion:
    action: str                     # move_block|shrink_block|reflow|force_code_preserve|mark_review...
    element_id: str | None = None
    params: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class StageAuditResult:
    stage_name: str
    score: float = 1.0
    status: str = OK
    hard_blockers: list = field(default_factory=list)    # [str]
    findings: list = field(default_factory=list)          # [Finding]
    evidence: list = field(default_factory=list)          # [EvidenceItem]
    suggestions: list = field(default_factory=list)       # [CorrectionSuggestion]
    elements: list = field(default_factory=list)          # [ElementAudit] (granulaire)

    def to_dict(self):
        return {"stage_name": self.stage_name, "score": round(self.score, 3), "status": self.status,
                "hard_blockers": list(self.hard_blockers),
                "findings": [f.to_dict() if isinstance(f, Finding) else f for f in self.findings],
                "suggestions": [s.to_dict() if isinstance(s, CorrectionSuggestion) else s for s in self.suggestions],
                "elements": [e.to_dict() if isinstance(e, ElementAudit) else e for e in self.elements]}


@dataclass
class PagePublicationReadyReport:
    page_id: str
    page_index: int = 0
    status: str = OK
    publication_ready: bool = False
    publication_ready_score: float = 0.0
    stage_scores: dict = field(default_factory=dict)     # {stage: score}
    hard_blockers: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    correction_suggestions: list = field(default_factory=list)
    stages: list = field(default_factory=list)            # [StageAuditResult]

    def to_dict(self):
        return {"page_id": self.page_id, "page_index": self.page_index, "status": self.status,
                "publication_ready": self.publication_ready,
                "publication_ready_score": round(self.publication_ready_score, 3),
                "stage_scores": {k: round(v, 3) for k, v in self.stage_scores.items()},
                "hard_blockers": list(self.hard_blockers),
                "findings": [f.to_dict() if isinstance(f, Finding) else f for f in self.findings],
                "correction_suggestions": [s.to_dict() if isinstance(s, CorrectionSuggestion) else s for s in self.correction_suggestions],
                "stages": [s.to_dict() if isinstance(s, StageAuditResult) else s for s in self.stages]}


@dataclass
class DocumentPublicationReadyReport:
    document_id: str
    page_count: int = 0
    status: str = OK
    publication_ready: bool = False
    publication_ready_score: float = 0.0
    pages: list = field(default_factory=list)             # [PagePublicationReadyReport]
    worst_pages: list = field(default_factory=list)
    blocking_pages: list = field(default_factory=list)
    global_findings: list = field(default_factory=list)

    def to_dict(self):
        return {"document_id": self.document_id, "page_count": self.page_count, "status": self.status,
                "publication_ready": self.publication_ready,
                "publication_ready_score": round(self.publication_ready_score, 3),
                "worst_pages": list(self.worst_pages), "blocking_pages": list(self.blocking_pages),
                "global_findings": [f.to_dict() if isinstance(f, Finding) else f for f in self.global_findings],
                "pages": [p.to_dict() if isinstance(p, PagePublicationReadyReport) else p for p in self.pages]}
