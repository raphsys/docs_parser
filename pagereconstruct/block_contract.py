"""BlockReconstructionContract — contrat complet d'un bloc textuel à rendre.

Reprend `BlockReconstructionPlan` legacy : géométrie + style + politique de
rendu + préservation + qualité. Les renderers reçoivent CE contrat, plus un dict
vague (directive Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .layout_contract import LayoutContract
from .style_contract import StyleContract


@dataclass
class RenderPolicy:
    renderer_name: str = "anchored_label_review"
    mode: str = "translated_editorial"
    strategy: str = "normal"
    overflow_policy: str = "shrink_then_reflow"
    shrink_policy: str = "max_14"
    reflow_policy: str = "within_layout_bbox"
    line_break_policy: str = "word"
    fallback_policy: str = "audited"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlockReconstructionContract:
    block_id: str
    role: str
    object_type: str
    source_unit_ids: list = field(default_factory=list)
    translation_unit_id: str | None = None
    reconstruction_unit_id: str | None = None
    source_text: str = ""
    translated_text: str = ""
    layout: LayoutContract = field(default_factory=LayoutContract)
    style: StyleContract = field(default_factory=StyleContract)
    render: RenderPolicy = field(default_factory=RenderPolicy)
    # préservation/voisinage
    protected_regions: list = field(default_factory=list)
    forbidden_overlap: float = 0.10
    # qualité
    must_render: bool = True
    must_not_clip: bool = True
    must_not_overlap: bool = True
    required_style_similarity: float = 0.95

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layout"] = self.layout.to_dict()
        d["style"] = self.style.to_dict()
        d["render"] = self.render.to_dict()
        return d

    def validate(self) -> list[str]:
        errs = []
        if not self.layout.layout_bbox:
            errs.append(f"{self.block_id}:missing_layout_bbox")
        if self.render.renderer_name in {"", "anchored_label_review"} and self.role not in {"unknown"}:
            errs.append(f"{self.block_id}:unresolved_renderer")
        if self.must_render and not (self.translated_text or self.source_text):
            errs.append(f"{self.block_id}:empty_text")
        return errs

    @classmethod
    def from_reconstruction_unit(cls, u: dict, *, protected_regions: list | None = None) -> "BlockReconstructionContract":
        role = str(u.get("role") or "unknown")
        return cls(
            block_id=u.get("id") or u.get("reconstruction_unit_id") or "blk",
            role=role,
            object_type=str(u.get("object_type") or "natural_text"),
            source_unit_ids=u.get("source_unit_ids") or [],
            translation_unit_id=u.get("translation_unit_id"),
            reconstruction_unit_id=u.get("id"),
            source_text=u.get("source_text") or "",
            translated_text=u.get("translated_text") or "",
            layout=LayoutContract.from_unit(u),
            style=StyleContract.from_resolved_style(u.get("style")),
            render=RenderPolicy(renderer_name=str(u.get("renderer") or "anchored_label_review")),
            protected_regions=protected_regions or [],
        )
