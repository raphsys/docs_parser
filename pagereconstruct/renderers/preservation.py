from .base import BaseRenderer


class PreservationRenderer(BaseRenderer):
    """Preserved artefacts (page numbers, logos, marks) stay on the source
    background — draw nothing, flag for explicit preservation."""

    renderer_name = "preservation"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        return [{"type": "preserved_artifact", "role": unit.get("role")}]
