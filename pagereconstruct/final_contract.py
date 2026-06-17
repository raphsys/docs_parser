"""FinalReconstructionContract — source UNIQUE et non ambiguë du rendu d'une page.

Fusionne pageprint + pagetranslate (source principale) et le savoir legacy
(complément) en un contrat figé. C'est l'équivalent moderne de FinalDocument /
DocumentObjectContract. Le rendu ne lit plus les vues brutes : il lit ce contrat.

Phase 2 : structure + from_pageprint_pagetranslate() + from_legacy_contract() +
merge_legacy_and_new() + validate(). `to_render_ops()` viendra en Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .background_contract import BackgroundContract
from .block_contract import BlockReconstructionContract
from .object_contract import ObjectContract
from .preservation_contract import PreservationContract
from .quality_contract import QualityContract

# Ordre des couches obligatoire (directive: layer_order_contract).
LAYER_ORDER = [
    "background_clean",
    "underlays",
    "text_removal_patches",
    "preserved_underlays",
    "translated_text",
    "preserved_overlays",
    "debug_overlays",
]


@dataclass
class PageInfo:
    document_id: str = ""
    page_index: int = 0
    page_size: list | None = None       # [w_pt, h_pt]
    coordinate_space: str = "pt"
    dpi_reference: float = 72.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceUnitState:
    source_unit_id: str
    state: str
    owner_contract_id: str | None = None
    text_removal_entry_id: str | None = None
    textop_ids: list = field(default_factory=list)
    preservationop_ids: list = field(default_factory=list)
    findings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinalReconstructionContract:
    page_info: PageInfo = field(default_factory=PageInfo)
    background: BackgroundContract = field(default_factory=BackgroundContract)
    objects: list = field(default_factory=list)             # [ObjectContract]
    blocks: list = field(default_factory=list)              # [BlockReconstructionContract]
    preservation: PreservationContract = field(default_factory=PreservationContract)
    source_unit_states: list = field(default_factory=list)
    quality: QualityContract = field(default_factory=QualityContract)
    layer_order: list = field(default_factory=lambda: list(LAYER_ORDER))
    legacy_compatibility: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)

    # ---- construction ----
    @classmethod
    def from_pageprint_pagetranslate(cls, normalized: dict, plan: dict) -> "FinalReconstructionContract":
        """Source principale : vues modernes + plan compilé."""
        page = normalized.get("page") or {}
        pi = normalized.get("page_intelligence") or {}
        # PAGEPRINT/PAGETRANSLATE normally expose page geometry under
        # normalized["page"]["geometry"].  Older code looked only at
        # page_intelligence.page_geometry; when the adapter did not pass that
        # object, FinalReconstructionContract.page_info.page_size became None.
        # Downstream flow solvers then inferred a fake page height from block
        # extents and moved valid top-of-page text toward the bottom.
        page_geom = page.get("geometry") or {}
        intel_geom = pi.get("page_geometry") or {}
        geom = {**page_geom, **intel_geom}
        assets = normalized.get("assets") or {}
        bg = ((plan.get("layers") or {}).get("background") or plan.get("background") or [{}])
        bg0 = bg[0] if isinstance(bg, list) and bg else (bg if isinstance(bg, dict) else {})

        protected = plan.get("protected_regions") or []
        blocks = [BlockReconstructionContract.from_reconstruction_unit(u, protected_regions=protected)
                  for u in (plan.get("layers") or {}).get("translated_text") or []]
        objects = [ObjectContract.from_region(r, i) for i, r in enumerate(normalized.get("regions") or [], 1)]

        contract = cls(
            page_info=PageInfo(
                document_id=str(page.get("document_id") or ""),
                page_index=int(page.get("page_index") or page.get("page") or 0) or 0,
                page_size=(
                    [geom.get("width_pt") or geom.get("width"),
                     geom.get("height_pt") or geom.get("height")]
                    if (geom.get("width_pt") or geom.get("width")) else None
                ),
                dpi_reference=float(geom.get("render_dpi") or 72.0),
            ),
            background=BackgroundContract.from_resolved(bg0 if isinstance(bg0, dict) else {}, assets),
            objects=objects,
            blocks=blocks,
            preservation=PreservationContract.from_plan(plan),
            quality=QualityContract(),
        )
        contract.source_unit_states = build_source_unit_states(contract)
        return contract

    @classmethod
    def from_legacy_contract(cls, page_data: dict) -> "FinalReconstructionContract":
        """Source legacy (complément). Délègue au bridge pour l'extraction réelle."""
        from .legacy_contract_bridge import convert_legacy_to_final_contract
        return convert_legacy_to_final_contract(page_data)

    def merge_legacy_and_new(self, legacy: "FinalReconstructionContract") -> "FinalReconstructionContract":
        """Legacy comble les trous SANS écraser le moderne (règles de priorité)."""
        if self.background.background_mode != "clean_background" and legacy.background.clean_background_path:
            self.background.clean_background_path = legacy.background.clean_background_path
            self.background.background_mode = "clean_background"
            self.background.source_text_leak_risk = "low"
            self.background.publication_allowed = True
        # overlays legacy ajoutés s'ils manquent (ne réintroduit pas le texte source)
        have = {o.object_id for o in self.preservation.objects}
        for o in legacy.preservation.objects:
            if o.object_id not in have:
                self.preservation.objects.append(o)
        if not self.blocks and legacy.blocks:
            self.blocks = legacy.blocks   # géométrie de secours
        return self

    # ---- validation ----
    def validate(self, *, mode: str = "debug") -> dict:
        errs: list[str] = []
        if mode == "publication":
            if self.quality.require_clean_background and self.background.background_mode != "clean_background":
                errs.append("publication_blocked:no_clean_background")
            if self.quality.forbid_source_text_leak_high and self.background.source_text_leak_risk == "high":
                errs.append("publication_blocked:source_text_leak_high")
        if not self.blocks:
            self.findings.append({"type": "no_blocks", "severity": "review"})
        for b in self.blocks:
            errs.extend(b.validate())
        return {"valid": not errs, "errors": errs}

    def to_dict(self) -> dict:
        return {
            "page_info": self.page_info.to_dict(),
            "background": self.background.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "blocks": [b.to_dict() for b in self.blocks],
            "preservation": self.preservation.to_dict(),
            "source_unit_states": [s.to_dict() if hasattr(s, "to_dict") else s for s in self.source_unit_states],
            "quality": self.quality.to_dict(),
            "layer_order": list(self.layer_order),
            "legacy_compatibility": dict(self.legacy_compatibility),
            "findings": list(self.findings),
        }


def build_source_unit_states(contract: FinalReconstructionContract) -> list[SourceUnitState]:
    states: dict[str, SourceUnitState] = {}
    for b in contract.blocks:
        for sid in b.source_unit_ids or []:
            states[sid] = SourceUnitState(
                source_unit_id=sid,
                state="translated_and_rendered",
                owner_contract_id=b.block_id,
                text_removal_entry_id=None,
                textop_ids=[],
                preservationop_ids=[],
            )
    for p in getattr(contract.preservation, "objects", []) or []:
        source_ids = list(getattr(p, "source_unit_ids", []) or [])
        if not source_ids:
            object_id = getattr(p, "object_id", None)
            source_ids = [object_id] if object_id else []
        for sid in source_ids:
            if not sid:
                continue
            st = states.get(sid)
            if st:
                st.preservationop_ids.append(getattr(p, "object_id", sid))
                st.findings.append("source_unit_both_preserved_and_translated")
            else:
                states[sid] = SourceUnitState(
                    source_unit_id=sid,
                    state="preserved_exact",
                    owner_contract_id=getattr(p, "object_id", sid),
                    preservationop_ids=[getattr(p, "object_id", sid)],
                )
    return list(states.values())
