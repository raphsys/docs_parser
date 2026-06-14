from .base import BaseRenderer


class AnchoredLabelRenderer(BaseRenderer):
    """Locked bbox, single line prioritised, bounded shrink."""

    renderer_name = "anchored_label"

    def layout_opts(self, style, page_w_px=None) -> dict:
        return {"align": style.get("alignment") or "left", "min_ratio": 0.70, "allow_lines": 2}


class AnchoredLabelReviewRenderer(AnchoredLabelRenderer):
    """Fallback for unknown roles — never the paragraph renderer."""

    renderer_name = "anchored_label_review"

    def measure(self, unit, sx, sy, page_w_px=None):
        rr = super().measure(unit, sx, sy, page_w_px)
        rr.status = "review"
        rr.findings.append({"type": "unknown_role_review", "role": unit.get("role")})
        return rr
