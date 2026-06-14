from ..ops import RenderResult
from .base import BaseRenderer


class FormulaRenderer(BaseRenderer):
    """Formulas are preserved: measure-only, draw nothing (keep source pixels)."""

    renderer_name = "formula"

    def measure(self, unit, sx, sy, page_w_px=None) -> RenderResult:
        return RenderResult(unit_id=unit.get("id"), renderer=self.renderer_name, status="ok",
                            planned_bbox=self._bbox(unit), findings=[{"type": "formula_preserved"}])

    def render(self, draw, unit, sx, sy, page_w_px=None) -> list:
        return [{"type": "formula_preserved"}]
