"""ObjectContract — identité/politique d'un objet de page (reprend
document_object_contract.build_document_object_contract : object_type/role/policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ObjectContract:
    object_id: str
    source_unit_ids: list = field(default_factory=list)
    object_type: str = "unknown"
    role: str = "unknown"
    bbox: list | None = None
    z_index: int = 0
    preservation_policy: str = "none"   # none | preserve_visual | preserve_text_exactly | exclude_artifact
    render_policy: str = "translated_editorial"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_region(cls, r: dict, idx: int) -> "ObjectContract":
        rt = str(r.get("region_type") or r.get("object_type") or "unknown")
        return cls(
            object_id=r.get("region_id") or f"obj_{idx:04d}",
            object_type=rt,
            role=str(r.get("role") or rt),
            bbox=r.get("bbox"),
            preservation_policy=("preserve_visual" if any(k in rt for k in ("formula", "code", "image", "drawing", "diagram", "chart", "figure")) else "none"),
            render_policy="preserve_visual" if r.get("observation_only") is False else "translated_editorial",
        )
