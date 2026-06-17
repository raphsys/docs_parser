"""Template global pour pages de livre avec figure et caption."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TemplateMatch:
    matched: bool
    confidence: float = 0.0
    findings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class BookFigurePageTemplate:
    def match(self, contract) -> TemplateMatch:
        data = contract.to_dict() if hasattr(contract, "to_dict") else dict(contract or {})
        roles = {b.get("role") for b in data.get("blocks", [])}
        objects = data.get("objects") or []
        has_figure = any((o.get("object_type") or o.get("region_type") or "").lower() in {"figure", "image"} for o in objects)
        has_caption = "figure_caption" in roles or "figure_caption_text" in roles
        matched = bool(has_figure and has_caption)
        return TemplateMatch(matched=matched, confidence=0.85 if matched else 0.0)

    def apply(self, contract):
        if hasattr(contract, "findings"):
            contract.findings.append({"type": "book_figure_page_template_applied"})
        elif isinstance(contract, dict):
            contract.setdefault("findings", []).append({"type": "book_figure_page_template_applied"})
        return contract
