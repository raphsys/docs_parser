from .base import BaseRenderer, draw_block


class HeadingRenderer(BaseRenderer):
    """One or two lines, no narrow vertical stacking; may expand width."""

    renderer_name = "heading"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        # Allow the heading to use up to ~90% of the page width so it does not
        # stack one word per line in a narrow source box.
        expand = (page_w_px * 0.92) if page_w_px else None
        return draw_block(draw, unit.get("translated_text") or "", px, style,
                          align=style.get("alignment") or "left",
                          min_ratio=0.90, allow_lines=2, expand_width_to=expand)
