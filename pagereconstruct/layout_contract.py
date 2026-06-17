"""LayoutContract — géométrie + libertés de déplacement d'une unité.

Reprend `LineTemplate` (gabarits de ligne) + `BlockGeometryContext` (padding,
zones protégées) + `GraphEdge` (keep_with) du moteur legacy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class LineTemplate:
    bbox: list                          # [x0,y0,x1,y1] pt
    baseline_y: float = 0.0
    usable_width: float = 0.0
    indent_pt: float = 0.0
    alignment: str = "left"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayoutContract:
    source_bbox: list | None = None
    coverage_bbox: list | None = None
    patch_bbox: list | None = None
    layout_bbox: list | None = None
    anchor_bbox: list | None = None
    safe_bbox: list | None = None        # bbox réduite évitant zones protégées
    overflow_bbox: list | None = None    # extension verticale autorisée
    line_templates: list = field(default_factory=list)   # [LineTemplate]
    graph_edges: list = field(default_factory=list)       # keep_with relations
    # libertés de déplacement (directive: libertés explicites, pas improvisation)
    allow_reflow: bool = True
    allow_shrink: bool = True
    max_shrink: float = 0.14
    allow_vertical_expansion: bool = False
    allow_local_shift: bool = False
    bbox_locked: bool = False            # table_cell etc.

    def to_dict(self) -> dict:
        d = asdict(self)
        d["line_templates"] = [lt.to_dict() if isinstance(lt, LineTemplate) else lt for lt in self.line_templates]
        return d

    @classmethod
    def from_unit(cls, u: dict) -> "LayoutContract":
        rt = u.get("render_target") or {}
        role = str(u.get("role") or "")
        locked = role in {"table_body_cell", "table_header_cell", "table_numeric_cell"}
        return cls(
            source_bbox=u.get("bbox"),
            coverage_bbox=u.get("coverage_bbox") or rt.get("coverage_bbox"),
            patch_bbox=u.get("patch_bbox") or rt.get("patch_bbox"),
            layout_bbox=u.get("layout_bbox") or rt.get("layout_bbox") or u.get("bbox"),
            anchor_bbox=u.get("anchor_bbox") or rt.get("anchor_bbox"),
            allow_reflow=role in {"body_paragraph", "list_item", "figure_caption", "table_caption", "index_entry", "bibliography_entry"},
            allow_shrink=not locked,
            allow_vertical_expansion=role in {"body_paragraph", "list_item"},
            bbox_locked=locked,
        )
