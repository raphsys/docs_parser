from .base import BaseRenderer


class IndexRenderer(BaseRenderer):
    """Index entries: left aligned, compact, page refs preserved (protected),
    bounded wrapping to avoid overlap with neighbours."""

    renderer_name = "index"

    def layout_opts(self, style, page_w_px=None) -> dict:
        return {"align": "left", "min_ratio": 0.78, "allow_lines": 3}
