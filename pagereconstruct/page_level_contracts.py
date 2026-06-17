"""Contrats dédiés aux objets page-level.

Ces objets ne passent pas par une heuristique de paragraphe ordinaire: leur
politique de traduction, de préservation et d'ancrage est explicite.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class PageHeaderContract:
    header_id: str
    running_title_text: str
    page_number_text: str
    title_bbox: list
    page_number_bbox: list
    style_title: dict = field(default_factory=dict)
    style_page_number: dict = field(default_factory=dict)
    translation_policy: str = "translate_running_title"
    preservation_policy: str = "preserve_page_number"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageNumberContract:
    page_number: str
    bbox: list
    placement: str = "header"
    translate: bool = False
    duplicate_allowed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SectionHeadingContract:
    section_id: str
    number_text: str
    title_text: str
    number_bbox: list
    title_bbox: list
    combined_bbox: list
    number_style: dict = field(default_factory=dict)
    title_style: dict = field(default_factory=dict)
    gap: float = 0.0
    baseline: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FigureCaptionContract:
    caption_id: str
    figure_id: str
    caption_number: str
    caption_text: str
    number_bbox: list
    text_bbox: list
    combined_bbox: list
    anchor: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InlineLinkContract:
    source_text: str
    translated_text: str
    url: str
    style: dict
    run_policy: str = "inline"

    def to_dict(self) -> dict:
        return asdict(self)


def _blue(style: dict) -> bool:
    c = str((style or {}).get("color") or "").lower()
    return c in {"#0000ff", "#0645ad", "blue"} or c.endswith("ff")


def audit_page_level_contracts(
    *,
    page_numbers: list[PageNumberContract] | None = None,
    figure_captions: list[FigureCaptionContract] | None = None,
    inline_links: list[InlineLinkContract] | None = None,
) -> dict:
    blockers: list[str] = []
    numbers = page_numbers or []
    if any(n.translate for n in numbers):
        blockers.append("page_number_translated")
    seen: dict[str, int] = {}
    for n in numbers:
        seen[n.page_number] = seen.get(n.page_number, 0) + 1
    if any(count > 1 for count in seen.values()) and not all(n.duplicate_allowed for n in numbers):
        blockers.append("duplicate_page_number")

    for c in figure_captions or []:
        if not c.figure_id or not c.anchor:
            blockers.append("figure_caption_unanchored")

    for link in inline_links or []:
        if link.run_policy != "inline" or not _blue(link.style) or link.source_text != link.url:
            blockers.append("inline_url_not_inline_blue_run")

    blockers = sorted(set(blockers))
    return {"status": "ko" if blockers else "ok", "hard_blockers": blockers}
