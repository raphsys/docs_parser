"""LegacyContractBridge — rend concrètement disponible le savoir de l'ancien
pipeline (ocr_server.process_page / reconstructor / FinalDocument) dans le
contrat moderne.

Le payload legacy d'une page (page_data) porte typiquement :
  background_path, mask_master_path, source_image_path, immutable_overlays[],
  blocks[] (final_blocks: lines→phrases→spans, role, bbox, style), non_text_zones,
  text_removal_debug.

Règles de priorité (directive Phase 3) :
  1. moderne = source principale ; 2. legacy = complément ;
  3. legacy ne réécrit pas une traduction validée ;
  4. legacy ne réintroduit pas le texte source.
"""

from __future__ import annotations

from .background_contract import BackgroundContract
from .block_contract import BlockReconstructionContract, RenderPolicy
from .layout_contract import LayoutContract
from .object_contract import ObjectContract
from .preservation_contract import PreservationContract, PreservedObject
from .style_contract import StyleContract


def load_legacy_page_contract(page_data: dict) -> dict:
    return page_data.get("legacy_page_structure") or page_data or {}


def extract_legacy_background(page_data: dict) -> BackgroundContract:
    clean = page_data.get("background_path") or page_data.get("clean_background_path")
    src = page_data.get("source_image_path")
    if clean:
        return BackgroundContract(clean_background_path=clean, source_image_path=src,
                                  background_mode="clean_background",
                                  source_text_leak_risk="low", publication_allowed=True)
    return BackgroundContract(source_image_path=src, background_mode="source_background" if src else "blank_degraded")


def extract_legacy_immutable_overlays(page_data: dict) -> list[PreservedObject]:
    out = []
    for i, ov in enumerate(page_data.get("immutable_overlays") or [], 1):
        bb = ov.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        reason = str(ov.get("reason") or ov.get("kind") or ov.get("type") or "immutable")
        out.append(PreservedObject(
            object_id=ov.get("id") or f"legacy_ov_{i:04d}", bbox=[float(x) for x in bb],
            reason=reason, method="keep_pixels", z_policy="preserve_original",
            source_unit_ids=ov.get("source_unit_ids") or [],
        ))
    return out


def extract_legacy_final_blocks(page_data: dict) -> list[BlockReconstructionContract]:
    blocks = []
    for b in page_data.get("blocks") or page_data.get("final_blocks") or []:
        bb = b.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        style = _dominant_style(b)
        role = str(b.get("role") or "body_paragraph")
        blocks.append(BlockReconstructionContract(
            block_id=str(b.get("id") or "legacy_blk"),
            role=role, object_type=str(b.get("object_type") or "natural_text"),
            source_unit_ids=[b.get("id")] if b.get("id") else [],
            source_text=_block_text(b),
            translated_text=b.get("translated_text") or "",
            layout=LayoutContract(source_bbox=list(bb), layout_bbox=list(bb), coverage_bbox=list(bb)),
            style=style,
            render=RenderPolicy(renderer_name=_renderer_for(role)),
        ))
    return blocks


def extract_legacy_style_contracts(page_data: dict) -> dict:
    return {b.get("id"): _dominant_style(b) for b in (page_data.get("blocks") or []) if b.get("id")}


def extract_legacy_layout_contracts(page_data: dict) -> dict:
    out = {}
    for b in page_data.get("blocks") or []:
        if b.get("id") and isinstance(b.get("bbox"), (list, tuple)):
            out[b["id"]] = LayoutContract(source_bbox=list(b["bbox"]), layout_bbox=list(b["bbox"]))
    return out


def extract_legacy_render_policies(page_data: dict) -> dict:
    return {b.get("id"): (b.get("render_policy") or b.get("render_mode"))
            for b in (page_data.get("blocks") or []) if b.get("id")}


def extract_legacy_inpaint_masks(page_data: dict) -> list:
    masks = []
    if page_data.get("mask_master_path"):
        masks.append({"path": page_data["mask_master_path"], "kind": "mask_master"})
    for r in (page_data.get("text_removal_debug") or {}).get("inpaint_regions") or []:
        masks.append({"bbox": r, "kind": "inpaint_region"})
    return masks


def extract_legacy_quality_hints(page_data: dict) -> dict:
    return dict(page_data.get("p6_bg_audit") or {})


def convert_legacy_to_final_contract(page_data: dict):
    from .final_contract import FinalReconstructionContract, PageInfo
    pd = load_legacy_page_contract(page_data)
    objects = [ObjectContract.from_region(z if isinstance(z, dict) else {"bbox": z, "region_type": "non_text_zone"}, i)
               for i, z in enumerate(pd.get("non_text_zones") or [], 1)]
    return FinalReconstructionContract(
        page_info=PageInfo(page_index=int(pd.get("page_index") or 0) or 0),
        background=extract_legacy_background(pd),
        objects=objects,
        blocks=extract_legacy_final_blocks(pd),
        preservation=PreservationContract(objects=extract_legacy_immutable_overlays(pd)),
        legacy_compatibility={
            "inpaint_masks": extract_legacy_inpaint_masks(pd),
            "quality_hints": extract_legacy_quality_hints(pd),
            "render_policies": extract_legacy_render_policies(pd),
        },
    )


# ---- helpers ----
def _block_text(b: dict) -> str:
    parts = []
    for line in b.get("lines") or []:
        for ph in line.get("phrases") or []:
            t = ph.get("texte") or ph.get("text")
            if t:
                parts.append(t)
    return " ".join(parts) or (b.get("text") or "")


def _dominant_style(b: dict) -> StyleContract:
    for line in b.get("lines") or []:
        for ph in line.get("phrases") or []:
            for sp in ph.get("spans") or []:
                s = sp.get("style") or {}
                if s:
                    return StyleContract.from_resolved_style({
                        "font": s.get("font"), "size": s.get("size"), "color": s.get("color"),
                        "flags": s.get("flags") or {}, "alignment": b.get("alignment"),
                        "size_source": "extracted",
                    })
    return StyleContract(alignment=str(b.get("alignment") or "left"))


def _renderer_for(role: str) -> str:
    r = role.lower()
    if "head" in r or "title" in r:
        return "heading"
    if "code" in r:
        return "code"
    if "formula" in r:
        return "formula"
    if "table" in r:
        return "table"
    if "caption" in r:
        return "caption"
    return "paragraph"
