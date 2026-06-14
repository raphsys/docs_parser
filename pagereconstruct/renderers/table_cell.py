from .base import BaseRenderer, draw_block


class TableCellRenderer(BaseRenderer):
    """Locked bbox, no expansion, stronger shrink, review on overflow."""

    renderer_name = "table_cell"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        findings = draw_block(draw, unit.get("translated_text") or "", px, style,
                              align=style.get("alignment") or "left", min_ratio=0.75)
        for f in findings:
            if f.get("type") == "overflow_unresolved":
                f["type"] = "table_cell_overflow"
                f["severity"] = "review"
        return findings
