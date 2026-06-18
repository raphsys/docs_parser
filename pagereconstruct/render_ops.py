"""RenderOps — instructions de dessin plates et résolues, exécutées à
l'identique par les backends PNG/PDF. C'est ICI (et nulle part dans les
backends) que dispatch + measure ont lieu UNE fois.

Mesure en espace PT (scale 1.0). Chaque backend applique son propre sx/sy.
Reprend `BlockRenderOp` / `DrawOp` legacy (op_type + bbox + text + style + z).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .errors import PublicationReadyError


@dataclass
class BackgroundOp:
    path: str | None
    z: int = 0
    op_type: str = "background"
    mode: str = "debug"             # debug | review | publication
    is_clean: bool = False          # True => path est un clean_background vérifié

    def to_dict(self):
        return asdict(self)


def assert_publication_background_allowed(op, *, source_image_path: str | None = None) -> None:
    data = op.to_dict() if hasattr(op, "to_dict") else dict(op or {})
    if data.get("mode") != "publication":
        return
    if not data.get("is_clean"):
        raise PublicationReadyError("source_image_background_forbidden")
    if source_image_path and data.get("path") and data.get("path") == source_image_path:
        raise PublicationReadyError("source_image_background_forbidden")


@dataclass
class PatchOp:
    bbox: list                      # pt
    color: str | None = None        # hex; None => sample/white
    reason: str = "text_removal"
    z: int = 10
    op_type: str = "patch"

    def to_dict(self):
        return asdict(self)


@dataclass
class PreservationOp:
    bbox: list                      # pt
    method: str = "keep_pixels"     # keep_pixels | copy_source_region | draw_text_exact
    source_path: str | None = None
    text: str | None = None
    source_unit_ids: list = field(default_factory=list)
    z: int = 20
    op_type: str = "preservation"

    def to_dict(self):
        return asdict(self)


@dataclass
class TextOp:
    lines: list                     # [{text, x, y_top}] in pt
    size_pt: float
    font_path: str
    color: list                     # [r,g,b] 0-255
    align: str = "left"
    role: str = ""
    unit_id: str | None = None
    composition_id: str | None = None
    block_id: str | None = None
    line_id: str | None = None
    run_id: str | None = None
    source_unit_ids: list = field(default_factory=list)
    translation_unit_id: str | None = None
    source: str = "intrablock_composition"
    z: int = 30
    op_type: str = "text"

    def to_dict(self):
        return asdict(self)


def _style_dict(style_contract) -> dict:
    """StyleContract -> dict lu par les renderers (flags/font_size_pt/color/align)."""
    sc = style_contract
    return {
        "font_size_pt": sc.font_size_pt or 10.0,
        "color": sc.color,
        "alignment": sc.alignment,
        "flags": {"bold": sc.bold, "italic": sc.italic,
                  "monospace": sc.font_class == "mono", "serif": sc.font_class == "serif"},
    }


def _unit_for_renderer(block) -> dict:
    return {
        "id": block.block_id,
        "translated_text": block.translated_text or block.source_text,
        "style": _style_dict(block.style),
        "layout_bbox": block.layout.layout_bbox,
        "coverage_bbox": block.layout.coverage_bbox,
        "bbox": block.layout.source_bbox,
        "role": block.role,
        "renderer": block.render.renderer_name,
    }


def _line_bbox_of_text_op(op: TextOp) -> list[float] | None:
    lines = getattr(op, "lines", None) or []
    boxes = []
    for ln in lines:
        if all(k in ln for k in ("x", "y_top", "x1", "y_bottom")):
            try:
                b = [float(ln["x"]), float(ln["y_top"]), float(ln["x1"]), float(ln["y_bottom"])]
                if b[2] > b[0] and b[3] > b[1]:
                    boxes.append(b)
            except Exception:
                pass
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _x_overlap(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def _overlaps(a: list[float], b: list[float]) -> bool:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) > 0 and max(0.0, min(a[3], b[3]) - max(a[1], b[1])) > 0


def _shift_text_op(op: TextOp, dy: float) -> None:
    if abs(dy) < 1e-6:
        return
    for ln in getattr(op, "lines", []) or []:
        if "y_top" in ln:
            ln["y_top"] = float(ln["y_top"]) + dy
        if "y_bottom" in ln:
            ln["y_bottom"] = float(ln["y_bottom"]) + dy


def _resolve_textop_vertical_collisions(ops: list, protected_boxes: list, page_h: float | None, findings: list) -> None:
    """Last-mile visual safety net for interline writing.

    Layout solvers work on blocks; RenderOps are the actual final geometry.  This
    pass makes the final text stream monotonic in each column and below hard
    preserved objects.  It never converts preservation to text and never moves
    preserved objects.
    """
    text_ops = [op for op in ops if isinstance(op, TextOp)]
    if len(text_ops) < 2 and not protected_boxes:
        return
    locked_roles = {"page_number", "page_reference", "toc_page_reference", "toc_section_number"}
    movable = []
    for op in text_ops:
        b = _line_bbox_of_text_op(op)
        if not b:
            continue
        movable.append((op, b))
    movable.sort(key=lambda x: (x[1][1], x[1][0]))
    placed: list[list[float]] = []
    shifted = 0
    for op, b in movable:
        role = str(getattr(op, "role", "") or "")
        if role in locked_roles:
            placed.append(b)
            continue
        dy = 0.0
        cur = list(b)
        # Text/text: if same column and the next line starts in the previous
        # line body, push it below with a small typographic gap.
        for prev in placed:
            if _x_overlap(cur, prev) >= 0.18 and cur[1] < prev[3] + 1.25 and cur[3] > prev[1]:
                # ``dy`` is total displacement from the original box ``b``.
                # Computing from ``cur`` loses prior shifts in a collision
                # cascade and can leave the line overlapping the next block.
                dy = max(dy, prev[3] + 1.25 - b[1])
                cur = [cur[0], b[1] + dy, cur[2], b[3] + dy]
        # Text/preservation: never let prose run through a preserved formula or
        # figure.  Move it below the obstacle only when there is real overlap in
        # the same horizontal band.
        for pr in protected_boxes or []:
            if _x_overlap(cur, pr) >= 0.12 and _overlaps(cur, pr):
                dy = max(dy, pr[3] + 1.75 - b[1])
                cur = [b[0], b[1] + dy, b[2], b[3] + dy]
        if page_h and cur[3] > page_h - 3:
            # Last resort: keep inside page instead of pushing indefinitely.
            dy = min(dy, max(0.0, page_h - 3 - b[3]))
            cur = [b[0], b[1] + dy, b[2], b[3] + dy]
        if dy > 0.05:
            _shift_text_op(op, dy)
            shifted += 1
            b = cur
        placed.append(b)
    if shifted:
        findings.append({"type": "visual_layout_reflow_v1_textop_collision_shift", "severity": "review", "shifted_textop_count": shifted})


def build_render_ops(contract, plan: dict, *, mode: str = "debug") -> list:
    """Compile le FinalReconstructionContract (+ patches du plan) en RenderOps
    ordonnés selon LAYER_ORDER. dispatch + measure se font une seule fois ici.

    `mode` (debug|review|publication) : en publication, le fond NE PEUT PAS être
    la source — uniquement un clean_background vérifié (règle architecturale)."""
    from .renderer_dispatcher import dispatch
    from .renderers.base import hex_rgb
    from .placement_solver import solve_block

    ops: list = []
    findings: list = []
    layers = plan.get("layers") or {}
    protected_boxes = [r["bbox"] for r in (plan.get("protected_regions") or [])
                       if isinstance(r.get("bbox"), (list, tuple)) and len(r["bbox"]) == 4]
    page_size = getattr(getattr(contract, "page_info", None), "page_size", None)
    page_h = float(page_size[1]) if isinstance(page_size, (list, tuple)) and len(page_size) == 2 and page_size[1] else None
    placed_boxes: list = []

    # 1. background — publication: UNIQUEMENT clean_background vérifié (pas de
    #    fallback silencieux vers la source). Si absent, path=None (fond dégradé)
    #    et l'audit/gates bloqueront la page.
    bg = contract.background
    bg_path = bg.render_path(mode)
    is_clean = (bg.background_mode == "clean_background"
                and bg.clean_background_path is not None
                and bg_path == bg.clean_background_path
                and bg.clean_background_verified
                and bg.text_removed)
    if mode == "publication" and not is_clean:
        findings.append({"type": "publication_background_not_clean", "severity": "ko",
                         "background_mode": bg.background_mode})
    ops.append(BackgroundOp(path=bg_path, mode=mode, is_clean=is_clean))

    # 2. text_removal_patches — only a VERIFIED clean background may skip
    # patches. A cleanbg file that is not explicitly verified can still contain
    # source text; in that case patches are mandatory before translated text.
    clean_background_ready = (
        contract.background.background_mode == "clean_background"
        and contract.background.clean_background_verified
        and contract.background.text_removed
    )
    if not clean_background_ready:
        for p in layers.get("patches") or []:
            if (p.get("protected_overlap_ratio") or 0) > 0.5:
                continue
            b = p.get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                ops.append(PatchOp(bbox=[float(x) for x in b], color=p.get("background_color"),
                                   reason=str(p.get("reason") or "text_removal")))

    # 3. preserved_underlays (via OverlayManager: classés + z_index + non destructibles)
    from .overlay_manager import build_preservation_ops
    src = contract.background.source_image_path
    underlay_ops, overlay_ops = build_preservation_ops(contract, source_path=src)
    ops.extend(underlay_ops)

    # 4. translated_text — le renderer ne compose plus. Les TextOps sont émis
    #    depuis IntraBlockComposition uniquement.
    from .composition.intrablock_composer import compose_block
    try:
        from .composition.paragraph_flow_grouper import blocks_for_render
        _render_blocks = blocks_for_render(contract)
    except Exception:
        _render_blocks = list(getattr(contract, "blocks", []) or [])
    for block in _render_blocks:
        if getattr(block, 'must_render', True) is False and not (getattr(block, 'translated_text', '') or getattr(block, 'source_text', '')):
            continue
        comp = compose_block(block)
        findings.extend({"type": f, "block_id": block.block_id} for f in comp.findings)
        if not block.translation_unit_id and block.role not in {"page_number"}:
            findings.append({
                "type": "translated_block_missing_translation_unit_id",
                "severity": "ko",
                "block_id": block.block_id,
            })
            continue
        for line in comp.lines:
            for run in line.runs:
                lines = [{"text": run.text, "x": run.bbox[0], "y_top": run.bbox[1],
                          "x1": run.bbox[2], "y_bottom": run.bbox[3]}]
                ops.append(TextOp(
                    lines=lines, size_pt=float(run.size_pt), font_path=run.font_path,
                    color=list(run.color), align=line.alignment, role=block.role,
                    unit_id=block.block_id, composition_id=comp.composition_id,
                    block_id=block.block_id, line_id=line.line_id, run_id=run.run_id,
                    source_unit_ids=list(block.source_unit_ids),
                    translation_unit_id=block.translation_unit_id,
                ))
                placed_boxes.append(run.bbox)

    # 4b. Last-mile text geometry guard: no text/text interline overlap and no
    #     text crossing hard preserved visual regions.
    _resolve_textop_vertical_collisions(ops, protected_boxes, page_h, findings)

    # 5. preserved_overlays (au-dessus du texte)
    ops.extend(overlay_ops)

    if findings:
        contract.findings.extend(findings)
    if mode == "publication":
        src = contract.background.source_image_path
        for op in ops:
            if isinstance(op, BackgroundOp):
                assert_publication_background_allowed(op, source_image_path=src)
    return ops
