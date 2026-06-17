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
    for block in contract.blocks:
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
