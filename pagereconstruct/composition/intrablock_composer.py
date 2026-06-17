"""IntraBlockComposer — étape ancienne 3 : composer le contenu DANS un bloc
(lignes, runs, objets inline) AVANT le rendu. Le renderer ne compose plus.

Réutilise la mesure existante (text_measure / layout_text) — porte la logique
ancienne _compose_paragraphs_in_box / _render_with_scale (shrink/overflow) sans
réécrire le découpage de lignes (cf. legacy registry → ADAPT).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..text_measure import measure_block


@dataclass
class TextRunPlacement:
    text: str
    bbox: list                      # pt
    font_path: str
    size_pt: float
    color: list                     # [r,g,b]
    run_id: str = ""
    line_id: str = ""
    source_unit_ids: list = field(default_factory=list)
    translation_unit_id: str | None = None
    bold: bool = False
    italic: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class InlineObjectPlacement:
    object_id: str
    object_type: str
    bbox: list
    anchor_line_id: str | None = None
    anchor_mode: str = "inline"     # inline | baseline | superscript | subscript | fixed
    preservation_policy: str = "preserve_visual"

    def to_dict(self):
        return asdict(self)


@dataclass
class LineLayout:
    line_id: str
    text: str
    bbox: list                      # pt [x0,y0,x1,y1]
    alignment: str = "left"
    runs: list = field(default_factory=list)          # [TextRunPlacement]
    inline_objects: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["runs"] = [r.to_dict() if isinstance(r, TextRunPlacement) else r for r in self.runs]
        return d


@dataclass
class IntraBlockComposition:
    composition_id: str = ""
    block_id: str = ""
    role: str = ""
    source_unit_ids: list = field(default_factory=list)
    translation_unit_ids: list = field(default_factory=list)
    block_bbox: list | None = None
    content_bbox: list | None = None
    lines: list = field(default_factory=list)          # [LineLayout]
    text_runs: list = field(default_factory=list)      # [TextRunPlacement] (à plat)
    inline_objects: list = field(default_factory=list)
    overflow: bool = False
    clipping: bool = False
    reading_order_ok: bool = True
    used_font_size: float | None = None
    used_line_height: float | None = None
    findings: list = field(default_factory=list)

    def to_dict(self):
        return {"composition_id": self.composition_id, "block_id": self.block_id, "role": self.role,
                "source_unit_ids": list(self.source_unit_ids),
                "translation_unit_ids": list(self.translation_unit_ids),
                "block_bbox": self.block_bbox, "content_bbox": self.content_bbox,
                "lines": [l.to_dict() for l in self.lines],
                "text_runs": [r.to_dict() for r in self.text_runs],
                "inline_objects": [o.to_dict() if hasattr(o, "to_dict") else o for o in self.inline_objects],
                "overflow": self.overflow, "clipping": self.clipping,
                "reading_order_ok": self.reading_order_ok,
                "used_font_size": self.used_font_size, "used_line_height": self.used_line_height,
                "findings": list(self.findings)}


def _hex_rgb(value, default=(20, 20, 20)):
    s = str(value or "").lstrip("#")
    if len(s) == 6:
        try:
            return list(int(s[k:k + 2], 16) for k in (0, 2, 4))
        except ValueError:
            pass
    return list(default)


# Bornes de taille de rendu par rôle (pt). Empêche qu'une taille pageprint
# aberrante (ex: code_block extrait à 32pt) ne produise un texte géant au rendu.
# Seuls titres/headings autorisent les grandes tailles.
_RENDER_SIZE_BOUNDS = {
    "body_paragraph": (6.5, 14.0), "body": (6.5, 14.0), "paragraph": (6.5, 14.0),
    "list_item": (6.5, 14.0), "caption": (6.0, 13.0), "figure_caption": (6.0, 13.0),
    "footnote": (5.5, 11.0), "author_bio": (6.5, 13.0),
    "bibliography_entry": (6.0, 12.5), "index_entry": (6.0, 11.5),
    "code_block": (6.0, 12.0), "code_line": (6.0, 12.0), "code": (6.0, 12.0),
    "table_body_cell": (5.5, 12.0), "table_header_cell": (5.5, 12.0),
    "diagram_label": (4.5, 12.0), "axis_label": (4.5, 12.0),
    "page_number": (6.0, 12.0), "page_reference": (6.0, 12.0),
    "section_heading": (8.0, 26.0), "heading": (8.0, 26.0), "title": (9.0, 30.0),
    "subtitle": (8.0, 22.0),
}
_DEFAULT_SIZE_BOUNDS = (5.0, 15.0)


def _clamp_render_size(size: float, role: str) -> float:
    lo, hi = _RENDER_SIZE_BOUNDS.get(role or "", _DEFAULT_SIZE_BOUNDS)
    return max(lo, min(hi, float(size or 10.0)))


def _style_for_measure(block) -> dict:
    st = getattr(block, "style", None)
    if st is None:
        return {}
    flags = {"bold": getattr(st, "bold", False), "italic": getattr(st, "italic", False),
             "monospace": getattr(st, "font_class", "") == "mono",
             "serif": getattr(st, "font_class", "serif") == "serif"}
    raw = getattr(st, "font_size_pt", None) or 10.0
    size = _clamp_render_size(raw, getattr(block, "role", ""))
    return {"font_size_pt": size, "flags": flags,
            "color": getattr(st, "color", "#000000"),
            "alignment": getattr(st, "alignment", "left")}


def compose_block(block) -> IntraBlockComposition:
    """Compose un BlockReconstructionContract en lignes mesurées (pt)."""
    text = (getattr(block, "translated_text", "") or getattr(block, "source_text", "") or "").strip()
    layout = getattr(block, "layout", None)
    bbox = getattr(layout, "layout_bbox", None) or getattr(layout, "coverage_bbox", None) or getattr(layout, "source_bbox", None)
    block_id = getattr(block, "block_id", "")
    comp = IntraBlockComposition(
        composition_id=f"comp_{block_id}",
        block_id=block_id,
        role=getattr(block, "role", ""),
        source_unit_ids=list(getattr(block, "source_unit_ids", []) or []),
        translation_unit_ids=[getattr(block, "translation_unit_id", None)] if getattr(block, "translation_unit_id", None) else [],
        block_bbox=list(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
        content_bbox=list(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
    )
    if not text or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        comp.findings.append("missing_text_or_bbox")
        return comp
    style = _style_for_measure(block)
    align = style.get("alignment") if style.get("alignment") in {"left", "center", "right"} else "left"
    lay = measure_block(text, bbox, style, align=align)
    color = _hex_rgb(getattr(getattr(block, "style", None), "color", "#000000"))
    fpath = lay.get("fpath")
    size = float(lay.get("size") or 10.0)
    boxes = lay.get("line_boxes") or []
    lns = lay.get("lines") or []
    base_lh = float(lay.get("line_h") or size * 1.2)

    # Natural baseline grid: do not distribute lines across spare bbox height.
    # The publication geometry optimizer owns bbox expansion/cascade.
    # Intra-block composition only stacks wrapped lines by measured line height.
    box_h = float(bbox[3] - bbox[1])
    lh = base_lh
    comp.used_font_size = size
    comp.used_line_height = lh
    comp.overflow = bool(lay.get("overflow"))
    y0 = float(bbox[1])
    for idx, (ln, box) in enumerate(zip(lns, boxes)):
        line_id = f"{comp.block_id}_l{idx}"
        run_id = f"{line_id}_r0"
        glyph_h = float(box[3] - box[1])
        y_top = y0 + idx * lh if lh > base_lh else float(box[1])
        nbox = [box[0], y_top, box[2], y_top + glyph_h]
        run = TextRunPlacement(text=ln, bbox=nbox, font_path=fpath,
                               size_pt=size, color=color,
                               run_id=run_id, line_id=line_id,
                               source_unit_ids=list(getattr(block, "source_unit_ids", []) or []),
                               translation_unit_id=getattr(block, "translation_unit_id", None),
                               bold=style["flags"].get("bold", False), italic=style["flags"].get("italic", False))
        comp.lines.append(LineLayout(line_id=line_id, text=ln, bbox=list(nbox),
                                     alignment=align, runs=[run]))
        comp.text_runs.append(run)
    if comp.overflow:
        comp.findings.append("overflow")
    return comp


def compose_contract(contract) -> list[IntraBlockComposition]:
    return [compose_block(b) for b in (getattr(contract, "blocks", []) or [])]
