from .base import BaseRenderer, draw_block


class AnchoredLabelRenderer(BaseRenderer):
    """Locked bbox, single line prioritised, bounded shrink."""

    renderer_name = "anchored_label"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        return draw_block(draw, unit.get("translated_text") or "", px, style,
                          align=style.get("alignment") or "left", min_ratio=0.70, allow_lines=2)


class AnchoredLabelReviewRenderer(AnchoredLabelRenderer):
    """Fallback for unknown roles — never the paragraph renderer."""

    renderer_name = "anchored_label_review"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        findings = super().draw(draw, unit, px, style, page_w_px)
        findings.append({"type": "unknown_role_review", "role": unit.get("role"), "severity": "review"})
        return findings
