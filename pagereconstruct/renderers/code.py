from .base import BaseRenderer


class CodeRenderer(BaseRenderer):
    """Monospace, left, no reflow. Code is preserved by default: render the
    source text verbatim unless explicitly marked translatable."""

    renderer_name = "code"

    def text_for(self, unit) -> str:
        if (unit.get("render_contract") or {}).get("mode") == "translate":
            return (unit.get("translated_text") or unit.get("source_text") or "").strip()
        return (unit.get("source_text") or "").strip()

    def layout_opts(self, style, page_w_px=None) -> dict:
        # Force monospace for code.
        style.setdefault("flags", {})
        style["flags"] = {**style["flags"], "monospace": True}
        return {"align": "left", "min_ratio": 0.8}
