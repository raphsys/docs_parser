from ..ops import RenderResult
from .base import BaseRenderer


class PreservationRenderer(BaseRenderer):
    """Preserved artefacts (page numbers, logos, marks) stay on the source
    background — measure-only, draw nothing."""

    renderer_name = "preservation"

    def measure(self, unit, sx, sy, page_w_px=None) -> RenderResult:
        return RenderResult(unit_id=unit.get("id"), renderer=self.renderer_name, status="ok",
                            planned_bbox=self._bbox(unit), findings=[{"type": "preserved_artifact"}])

    def render(self, draw, unit, sx, sy, page_w_px=None) -> list:
        return [{"type": "preserved_artifact", "role": unit.get("role")}]
