from .base import BaseRenderer, draw_block


class ParagraphRenderer(BaseRenderer):
    renderer_name = "paragraph"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        align = style.get("alignment") or "left"
        return draw_block(draw, unit.get("translated_text") or "", px, style,
                          align=align if align in {"left", "center", "right"} else "left",
                          min_ratio=0.86)


class ListItemRenderer(ParagraphRenderer):
    renderer_name = "list_item"
