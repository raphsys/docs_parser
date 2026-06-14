from .base import BaseRenderer


class FormulaRenderer(BaseRenderer):
    """Formulas are preserved: draw nothing, keep the source pixels in place."""

    renderer_name = "formula"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        return [{"type": "formula_preserved"}]
