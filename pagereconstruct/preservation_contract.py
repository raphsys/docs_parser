"""PreservationContract — objets gardés tels quels (pixels) + texte exact.

Reprend immutable_overlays / FormulaItem (copie région source) legacy. Chaque
entrée devient une PreservationOp à l'exécution (underlay sous le texte, overlay
au-dessus). Empêche le patch destructeur sur ces zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# z_policy: original = sous le texte (figure/fond), over_text = au-dessus (page num, label)
_OVER_TEXT = {"page_number", "page_reference", "toc_page_reference", "toc_section_number",
              "caption_label", "caption_number", "diagram_label"}


@dataclass
class PreservedObject:
    object_id: str
    bbox: list
    reason: str = "preserve"
    method: str = "keep_pixels"        # keep_pixels | copy_source_region | draw_text_exact
    z_policy: str = "preserve_original"  # preserve_original (underlay) | over_text (overlay)
    text: str | None = None            # pour draw_text_exact
    source_unit_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreservationContract:
    objects: list = field(default_factory=list)   # [PreservedObject]

    def to_dict(self) -> dict:
        return {"objects": [o.to_dict() if isinstance(o, PreservedObject) else o for o in self.objects]}

    @property
    def underlays(self) -> list:
        return [o for o in self.objects if getattr(o, "z_policy", "preserve_original") != "over_text"]

    @property
    def overlays(self) -> list:
        return [o for o in self.objects if getattr(o, "z_policy", "") == "over_text"]

    @classmethod
    def from_plan(cls, plan: dict) -> "PreservationContract":
        objs: list[PreservedObject] = []
        layers = plan.get("layers") or {}
        n = 0
        for kind, items in (("under", layers.get("preserved_underlays") or []),
                            ("over", layers.get("preserved_overlays") or [])):
            for it in items:
                n += 1
                reason = str(it.get("reason") or "preserve")
                objs.append(PreservedObject(
                    object_id=it.get("id") or f"pres_{n:04d}",
                    bbox=it.get("bbox"), reason=reason,
                    method="draw_text_exact" if it.get("text") and reason in _OVER_TEXT else "keep_pixels",
                    z_policy="over_text" if kind == "over" or reason in _OVER_TEXT else "preserve_original",
                    text=it.get("text"),
                    source_unit_ids=it.get("source_unit_ids") or [],
                ))
        return cls(objects=objs)
