from .base import BaseRenderer, layout_text


class BibliographyRenderer(BaseRenderer):
    """One reference = one logical block, left aligned, hanging indent on
    continuation lines. References/DOI/arXiv preserved (protected tokens)."""

    renderer_name = "bibliography"

    def layout_opts(self, style, page_w_px=None) -> dict:
        return {"align": "left", "min_ratio": 0.84}

    def render(self, draw, unit, sx, sy, page_w_px=None) -> list:
        rr = self.measure(unit, sx, sy, page_w_px)
        lay = getattr(rr, "_lay", None)
        if lay is None:
            return rr.findings
        from .base import hex_rgb
        color = hex_rgb((unit.get("style") or {}).get("color"))
        font, indent = lay["font"], 12
        for i, (ln, box) in enumerate(zip(lay["lines"], lay["line_boxes"])):
            x = box[0] + (indent if i else 0)  # hanging indent on wrapped lines
            draw.text((x, box[1]), ln, fill=color, font=font)
        return rr.findings
