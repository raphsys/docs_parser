from .base import BaseRenderer, draw_block


class CodeRenderer(BaseRenderer):
    """Monospace, left, no reflow. Code is preserved by default: render the
    source text verbatim unless it was explicitly marked translatable."""

    renderer_name = "code"

    def draw(self, draw, unit, px, style, page_w_px=None) -> list:
        text = unit.get("source_text") if unit.get("render_contract", {}).get("mode") != "translate" \
            else (unit.get("translated_text") or unit.get("source_text"))
        st = dict(style)
        st.setdefault("flags", {})
        st["flags"] = {**st["flags"], "monospace": True}
        findings = draw_block(draw, text or "", px, st, align="left", min_ratio=0.8)
        findings.append({"type": "code_preserved"})
        return findings
