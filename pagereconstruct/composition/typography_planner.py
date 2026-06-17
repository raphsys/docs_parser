"""TypographyPlanner — la typographie doit être PRÊTE avant le rendu.

Produit un TypographyPlan (échelle de page + plan par bloc + em estimé +
confiance) en réutilisant le moteur existant ocr_typography_engine (cap/x-height
sur image). Ne réinvente pas l'estimation em (legacy registry → ADAPT).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlockStylePlan:
    block_id: str
    role: str
    font_class: str = "serif"
    font_size_pt_source_metric: float | None = None
    font_size_pt_em_estimated: float | None = None
    font_size_pt_rendered: float | None = None
    line_height_target: float | None = None
    style_confidence: float = 0.0

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


@dataclass
class TypographyPlan:
    style_ladder: dict = field(default_factory=dict)      # role -> em size
    block_style_plans: list = field(default_factory=list)  # [BlockStylePlan]
    confidence: float = 0.0
    findings: list = field(default_factory=list)

    def to_dict(self):
        return {"style_ladder": self.style_ladder, "confidence": round(self.confidence, 3),
                "block_style_plans": [b.to_dict() for b in self.block_style_plans],
                "findings": list(self.findings)}


def plan_typography(contract, *, page_image_path: str | None = None) -> TypographyPlan:
    from ..ocr_typography_engine import enhance_contract_typography
    img = page_image_path or getattr(getattr(contract, "background", None), "source_image_path", None)
    res = enhance_contract_typography(contract, page_image_path=img)
    plans = []
    ladder = {}
    for r in res.resolved:
        plans.append(BlockStylePlan(
            block_id=r.block_id, role=r.role, font_class=r.font_class,
            font_size_pt_em_estimated=r.font_size_pt_em, line_height_target=r.line_height_pt,
            style_confidence=r.confidence))
        # échelle de page : taille em par rôle (médiane implicite via dernier vu).
        ladder.setdefault(r.role, round(r.font_size_pt_em, 2))
    tp = TypographyPlan(style_ladder=ladder, block_style_plans=plans,
                        confidence=res.page_score, findings=[f for f in res.findings])
    return tp
