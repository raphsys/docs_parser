import html
import os
import re


def _safe_text(s):
    return html.escape((s or "").strip())


def _block_text(block):
    if block.get("translated_text"):
        return re.sub(r"\s+", " ", block.get("translated_text", "")).strip()
    parts = []
    for line in block.get("lines", []):
        for phrase in line.get("phrases", []):
            t = phrase.get("translated_text") or phrase.get("texte") or ""
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                parts.append(t)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _css_class_id(raw):
    if not raw:
        return "cls-default"
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw)).strip("-").lower()
    return f"cls-{s or 'default'}"


def _style_to_css(style):
    style = style or {}
    flags = style.get("flags", {}) if isinstance(style.get("flags"), dict) else {}
    font = style.get("font", "Arial")
    size = float(style.get("size", 12.0))
    color = style.get("color", "#222222")
    weight = "700" if flags.get("bold") else "400"
    italic = "italic" if flags.get("italic") else "normal"
    return (
        f"font-family: '{font}', Arial, sans-serif; "
        f"font-size: {max(8.0, min(72.0, size)):.1f}px; "
        f"font-weight: {weight}; "
        f"font-style: {italic}; "
        f"color: {color};"
    )


class HtmlStyleExporter:
    def export(self, pages, output_path):
        html_doc = []
        html_doc.append("<!doctype html>")
        html_doc.append("<html lang=\"fr\">")
        html_doc.append("<head>")
        html_doc.append("<meta charset=\"utf-8\"/>")
        html_doc.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>")
        html_doc.append("<title>Reconstruction Stylée</title>")
        html_doc.append("<style>")
        html_doc.append(self._base_css())
        html_doc.append(self._component_css(pages))
        html_doc.append("</style>")
        html_doc.append("</head>")
        html_doc.append("<body>")
        html_doc.append("<main class=\"doc-wrap\">")

        for i, page in enumerate(pages):
            html_doc.append(self._render_page(page, i))

        html_doc.append("</main>")
        html_doc.append("</body>")
        html_doc.append("</html>")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_doc))
        return output_path

    def _base_css(self):
        return """
:root {
  --page-bg: #ffffff;
  --ink: #1e1f23;
  --paper-shadow: 0 14px 40px rgba(12, 15, 26, 0.12);
  --space-1: 6px;
  --space-2: 10px;
  --space-3: 16px;
  --space-4: 24px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: linear-gradient(180deg, #f4f6f9 0%, #e9edf3 100%);
  color: var(--ink);
}
.doc-wrap {
  max-width: 1040px;
  margin: 0 auto;
  padding: 24px 14px 32px;
}
.page {
  background: var(--page-bg);
  border-radius: 10px;
  box-shadow: var(--paper-shadow);
  margin: 0 0 22px;
  padding: 22px 26px;
  overflow: hidden;
}
.page.two-col .page-content {
  column-count: 2;
  column-gap: 28px;
}
.blk {
  break-inside: avoid;
  margin: 0 0 var(--space-3);
  white-space: pre-wrap;
  word-wrap: break-word;
}
.role-header, .role-footer {
  opacity: 0.86;
}
.role-caption {
  opacity: 0.94;
}
@media (max-width: 820px) {
  .page { padding: 16px 14px; border-radius: 0; }
  .doc-wrap { padding: 0; }
  .page.two-col .page-content { column-count: 1; }
}
"""

    def _component_css(self, pages):
        rules = []
        emitted = set()
        for page in pages:
            profile = page.get("style_profile", {}) if isinstance(page, dict) else {}
            comp = profile.get("component_styles", {}) if isinstance(profile, dict) else {}
            for class_id, meta in comp.items():
                css_id = _css_class_id(class_id)
                if css_id in emitted:
                    continue
                emitted.add(css_id)
                style = meta.get("style", {})
                rules.append(f".{css_id} {{{_style_to_css(style)}}}")
        if not rules:
            rules.append(".cls-default { font-family: Arial, sans-serif; font-size: 12px; color: #222; }")
        return "\n".join(rules)

    def _render_page(self, page, idx):
        layout = page.get("layout", {}) if isinstance(page, dict) else {}
        margins = layout.get("margins", {}) if isinstance(layout, dict) else {}
        grid = page.get("style_profile", {}).get("page_grid", {}) if isinstance(page, dict) else {}
        cols = int(grid.get("columns", 1) or 1)
        two_col = " two-col" if cols >= 2 else ""
        left_pad = int(margins.get("left", 0))
        right_pad = int(margins.get("right", 0))
        top_pad = int(margins.get("top", 0))
        bottom_pad = int(margins.get("bottom", 0))

        body = [f"<section class=\"page{two_col}\" id=\"page-{idx + 1}\">"]
        body.append(
            f"<div class=\"page-content\" style=\"padding:{max(6, top_pad//5)}px {max(8, right_pad//5)}px {max(6, bottom_pad//5)}px {max(8, left_pad//5)}px;\">"
        )
        for block in page.get("blocks", []):
            txt = _block_text(block)
            if not txt:
                continue
            role = (block.get("semantic", {}) or {}).get("type") or block.get("role", "body")
            role_css = f"role-{re.sub(r'[^a-zA-Z0-9_-]+', '-', str(role).lower())}"
            cls = _css_class_id(block.get("style_class"))
            align = block.get("alignment", "left")
            indent = float(block.get("indent_px", 0.0) or 0.0)
            heading_level = (block.get("semantic", {}) or {}).get("heading_level")
            tag = "p"
            if isinstance(heading_level, int) and 1 <= heading_level <= 6:
                tag = f"h{heading_level}"
            elif role in {"header", "footer"}:
                tag = "p"
            style_attr = f"text-align:{align}; margin-left:{max(0.0, indent/4.0):.1f}px;"
            body.append(f"<{tag} class=\"blk {cls} {role_css}\" style=\"{style_attr}\">{_safe_text(txt)}</{tag}>")
        body.append("</div>")
        body.append("</section>")
        return "\n".join(body)
