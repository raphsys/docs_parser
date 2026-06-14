from .base import BaseRenderer


class HeadingRenderer(BaseRenderer):
    """One or two lines, no narrow vertical stacking; may expand width."""

    renderer_name = "heading"

    def layout_opts(self, style, page_w_px=None) -> dict:
        return {"align": style.get("alignment") or "left", "min_ratio": 0.90,
                "allow_lines": 2, "expand_width_to": (page_w_px * 0.92) if page_w_px else None}
