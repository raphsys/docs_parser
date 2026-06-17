"""Schema / data contracts for PAGERECONSTRUCT.

The pipeline never renders input views directly: it compiles an intermediate
``PageRenderPlan`` (layers + protections + consumed/excluded ids) which is then
handed to the PDF backend. This module holds that contract only — no rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PLAN_SCHEMA_VERSION = "pagereconstruct.plan.v1"
OUTPUT_SCHEMA_VERSION = "pagereconstruct.output.v1"


@dataclass
class ProtectedRegion:
    id: str
    source: str          # preservation_plan | exclusion_plan | unit_policy | region | visual_layer
    reason: str          # formula | image | publisher_mark | watermark | code | table_grid | ...
    bbox: list | None
    hard: bool = True
    z_policy: str = "preserve_original"  # under_text | over_text | preserve_original

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "reason": self.reason,
                "bbox": self.bbox, "hard": self.hard, "z_policy": self.z_policy}


@dataclass
class TranslatedTextUnit:
    id: str
    kind: str            # translated_text
    renderer: str        # paragraph | heading | caption | table | code | formula | anchored_label | ...
    source_unit_ids: list
    translation_unit_id: str | None
    source_text: str | None
    translated_text: str | None
    role: str | None
    object_type: str | None
    semantic_kind: str | None
    page_role: str | None
    bbox: list | None
    coverage_bbox: list | None  # union of source-unit bboxes (full area to patch/redraw)
    layout_bbox: list | None    # where the translated text is laid out
    patch_bbox: list | None     # where the old text must be erased
    bbox_reliable: bool
    style: dict
    render_target: dict
    render_contract: dict

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PreservedUnit:
    id: str
    source: str          # preservation_plan | exclusion_plan
    reason: str
    bbox: list | None
    text: str | None
    preservation_mode: str | None
    source_unit_ids: list = field(default_factory=list)
    z_policy: str = "over_text"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PatchZone:
    op_type: str
    unit_id: str
    bbox: list
    method: str = "sampled_whiteout"
    background_color: str | None = None
    protected_overlap_ratio: float = 0.0
    must_not_overlap: list = field(default_factory=list)
    padding: list = field(default_factory=lambda: [1.0, 0.5, 1.0, 0.5])

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PageRenderPlan:
    page: dict
    translated_text: list = field(default_factory=list)        # TranslatedTextUnit
    preserved_underlays: list = field(default_factory=list)    # PreservedUnit
    preserved_overlays: list = field(default_factory=list)     # PreservedUnit
    patches: list = field(default_factory=list)                # PatchZone (Passe 2)
    background: list = field(default_factory=list)
    protected_regions: list = field(default_factory=list)      # ProtectedRegion
    consumed_source_unit_ids: list = field(default_factory=list)
    excluded_source_unit_ids: list = field(default_factory=list)
    render_policy: dict = field(default_factory=dict)
    quality_expectations: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    render_ops: list = field(default_factory=list)   # frozen RenderOps (contract-driven)
    final_contract: dict = field(default_factory=dict)
    text_removal_ledger: list = field(default_factory=list)
    source_text_lifecycle_ledger: list = field(default_factory=list)
    intrablock_compositions: list = field(default_factory=list)
    page_level_contracts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "page": self.page,
            "layers": {
                "background": self.background,
                "patches": [p.to_dict() for p in self.patches],
                "preserved_underlays": [p.to_dict() for p in self.preserved_underlays],
                "translated_text": [t.to_dict() for t in self.translated_text],
                "preserved_overlays": [p.to_dict() for p in self.preserved_overlays],
            },
            "protected_regions": [r.to_dict() for r in self.protected_regions],
            "consumed_source_unit_ids": self.consumed_source_unit_ids,
            "excluded_source_unit_ids": self.excluded_source_unit_ids,
            "render_policy": self.render_policy,
            "quality_expectations": self.quality_expectations,
            "findings": self.findings,
            "render_ops": self.render_ops,
            "final_contract": self.final_contract,
            "text_removal_ledger": self.text_removal_ledger,
            "source_text_lifecycle_ledger": self.source_text_lifecycle_ledger,
            "intrablock_compositions": self.intrablock_compositions,
            "page_level_contracts": self.page_level_contracts,
        }

    def summary(self) -> dict:
        return {
            "translated_text_count": len(self.translated_text),
            "preserved_underlay_count": len(self.preserved_underlays),
            "preserved_overlay_count": len(self.preserved_overlays),
            "patch_count": len(self.patches),
            "protected_region_count": len(self.protected_regions),
            "consumed_source_unit_count": len(self.consumed_source_unit_ids),
            "excluded_source_unit_count": len(self.excluded_source_unit_ids),
            "finding_count": len(self.findings),
        }
