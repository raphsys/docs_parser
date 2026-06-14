from .base import BaseRenderer


class TableCellRenderer(BaseRenderer):
    """Locked bbox, no expansion, stronger shrink, review on overflow."""

    renderer_name = "table_cell"

    def layout_opts(self, style, page_w_px=None) -> dict:
        return {"align": style.get("alignment") or "left", "min_ratio": 0.75}
