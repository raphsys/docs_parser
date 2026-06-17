"""StyleContract — typographie figée d'une unité (jamais inventée par le rendu).

Reprend `BlockSemanticProfile` (dominant font) + style attributes legacy. La
provenance (`source`) distingue extracted / inferred / repaired pour auditer les
réparations (directive: aucune réparation silencieuse).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class StyleContract:
    font_family: str | None = None
    font_class: str = "serif"          # serif | sans | mono
    font_size_pt: float | None = None
    bold: bool = False
    italic: bool = False
    color: str = "#000000"
    line_height: float | None = None
    alignment: str = "left"            # left | center | right | justify
    indent_pt: float = 0.0
    fontfile: str | None = None        # police réelle si dispo (legacy FontResolver)
    source: str = "unknown"            # extracted | inferred | repaired | unknown
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def reliable(self) -> bool:
        return self.source in {"extracted", "inferred"} and bool(self.font_size_pt)

    @classmethod
    def from_resolved_style(cls, style: dict | None) -> "StyleContract":
        s = style or {}
        flags = s.get("flags") or {}
        return cls(
            font_family=s.get("font") or s.get("font_family"),
            font_class=str(s.get("font_class") or _infer_class(s.get("font") or "")),
            font_size_pt=_num(s.get("font_size_pt") or s.get("size")),
            bold=bool(flags.get("bold") or s.get("bold")),
            italic=bool(flags.get("italic") or s.get("italic")),
            color=str(s.get("color") or "#000000"),
            line_height=_num(s.get("line_height")),
            alignment=str(s.get("alignment") or "left"),
            indent_pt=float(s.get("indent_pt") or 0.0),
            fontfile=s.get("fontfile"),
            source=str(s.get("size_source") or s.get("source") or ("extracted" if s.get("font_size_pt") or s.get("size") else "unknown")),
            confidence=float(s.get("confidence") or 0.0),
        )


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _infer_class(font: str) -> str:
    f = font.lower()
    if any(k in f for k in ("mono", "courier", "consol", "code")):
        return "mono"
    if any(k in f for k in ("sans", "arial", "helvet", "calibri", "verdana")):
        return "sans"
    return "serif"
