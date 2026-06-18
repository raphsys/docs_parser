"""Resolve documentary roles for PAGEPRINT units."""

from __future__ import annotations

import re


CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table|tab\.)\s+\d+(?:[.-]\d+)?", re.IGNORECASE)
TOC_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?\S.{3,}?(?:\.{2,}|\s{2,})\s*[ivxlcdm\d]+\s*$", re.IGNORECASE)
INDEX_RE = re.compile(r"^\s*[A-Za-z][^,]{1,80},\s*(?:\d+[,-–\s]*)+\s*$")
PAGE_REF_RE = re.compile(r"^\s*(?:[ivxlcdm]+|\d+)(?:[-–]\d+)?\s*$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s*$")
PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/[\w.-]+/|\.{1,2}/)[^\s]+")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
COMMAND_RE = re.compile(r"^(?:copy|dir|del|findstr|mkdir|rmdir|cd|ls|cat|grep|sudo|docker|npm|pip|python|git)\b", re.IGNORECASE)
# Shared publisher/watermark patterns (single source of truth).
from .structure_builders.publisher_mark_builder import PUBLISHER_RE, WATERMARK_RE


ROLE_TO_OBJECT = {
    "body_paragraph": ("natural_text", "prose"),
    "title": ("natural_text", "title"),
    "subtitle": ("natural_text", "subtitle"),
    "section_heading": ("natural_text", "heading"),
    "subsection_heading": ("natural_text", "heading"),
    "list_item": ("natural_text", "list_item_text"),
    "list_marker": ("marker", "list_marker"),
    "figure_caption": ("natural_text", "caption_text"),
    "figure_label": ("reference", "figure_label"),
    "table_caption": ("natural_text", "caption_text"),
    "table_header_cell": ("table_cell", "table_header"),
    "table_body_cell": ("table_cell", "table_body"),
    "table_numeric_cell": ("table_cell", "numeric"),
    "formula_expression": ("formula_expression", "formula"),
    "formula_explanation": ("natural_text", "formula_explanation"),
    "code_block": ("code", "code_block"),
    "code_line": ("code", "code_line"),
    "code_token": ("code_token", "code_token"),
    "command_name": ("command_name", "code_token"),
    "path": ("path", "protected_token"),
    "file_name": ("file_name", "protected_token"),
    "url": ("url", "protected_token"),
    "email": ("email", "protected_token"),
    "page_header": ("running_text", "header"),
    "page_footer": ("running_text", "footer"),
    "publisher_mark": ("publisher_mark", "artifact"),
    "watermark": ("watermark", "artifact"),
    "author_name": ("proper_name", "person_name"),
    "author_bio": ("natural_text", "biography"),
    "index_entry": ("index_entry", "index_entry"),
    "index_head_term": ("technical_term", "index_term"),
    "index_subentry": ("natural_text", "index_subentry"),
    "index_page_reference": ("page_reference", "reference"),
    "page_reference": ("page_reference", "reference"),
    "toc_title": ("natural_text", "toc_title"),
    "toc_entry": ("toc_entry", "toc_entry"),
    "toc_entry_title": ("natural_text", "toc_entry_title"),
    "toc_section_number": ("section_number", "reference"),
    "toc_page_reference": ("page_reference", "reference"),
    "toc_bullet_marker": ("marker", "toc_marker"),
    "diagram_label": ("diagram_label", "label"),
    "diagram_text_label": ("natural_text", "diagram_label_text"),
}


def resolve_roles(
    units: list[dict],
    *,
    page_intelligence: dict | None = None,
    document_context: dict | None = None,
) -> dict:
    """Assign roles in-place and return role metrics."""
    page_intelligence = page_intelligence or {}
    role_counts: dict[str, int] = {}
    unresolved = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        role, reason, confidence = resolve_unit_role(
            unit,
            page_intelligence=page_intelligence,
            document_context=document_context or {},
        )
        understanding = unit.setdefault("understanding", {})
        understanding["role"] = role
        object_type, semantic_kind = ROLE_TO_OBJECT.get(role, ("unknown", "unknown"))
        if object_type != "unknown":
            understanding["object_type"] = object_type
        elif not understanding.get("object_type"):
            understanding["object_type"] = "unknown"
        if semantic_kind != "unknown":
            understanding["semantic_kind"] = semantic_kind
        elif not understanding.get("semantic_kind"):
            understanding["semantic_kind"] = "unknown"
        unit.setdefault("confidence", {})["role"] = confidence
        unit.setdefault("provenance", {}).setdefault("decision_trace", []).append({
            "stage": "role_resolution",
            "target_id": unit.get("unit_id"),
            "decision": f"role={role}",
            "reason": reason,
            "confidence": confidence,
        })
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == "unknown":
            unresolved.append(unit.get("unit_id"))
    _propagate_block_code(units)
    return {
        "schema_version": "pageprint.role_resolution.v1",
        "role_counts": role_counts,
        "unresolved_unit_ids": unresolved,
    }


_CODE_LINE_ROLES = {"code_line", "command_name", "path", "code_block"}
# Roles never overridden by code propagation (already non-translatable visuals).
_CODE_KEEP_ROLES = _CODE_LINE_ROLES | {
    "formula_expression", "publisher_mark", "watermark", "page_number", "page_reference",
}
# Code/SQL line signature: a line that smells like source even if mislabeled prose.
_CODE_SIGNAL_RE = re.compile(
    r"\b(SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|INSERT|UPDATE|DELETE|JOIN|WITH|VALUES|"
    r"CREATE|ALTER|DROP|UNION|HAVING|LIMIT|INNER|OUTER|LEFT|RIGHT|def|class|import|"
    r"return|function|public|private|void|const|SELECT\b)\b", re.IGNORECASE)


def _line_is_codeish(line: dict) -> bool:
    role = (line.get("understanding") or {}).get("role")
    text = str((line.get("content") or {}).get("text") or "")
    if role == "formula_expression":
        return _is_strong_math(text)
    if role in _CODE_LINE_ROLES:
        return _has_code_evidence(text)
    return bool(_CODE_SIGNAL_RE.search(text))


def _propagate_block_code(units: list[dict]) -> int:
    """A code block is atomic: if a block's lines are mostly code/formula, every
    descendant text line is code (non-translatable), not prose.

    Per-line role resolution mislabels parts of a code listing as
    heading/paragraph ("➊ WITH", "➋ SELECT geo_name, …") and the pipeline then
    translates them — they stack over the code lines that stayed code_line.
    Whole-block propagation keeps a listing coherent.
    """
    lines_by_block: dict[str, list[dict]] = {}
    for u in units:
        if isinstance(u, dict) and u.get("level") == "line":
            lines_by_block.setdefault(u.get("parent_id"), []).append(u)
    code_block_ids = []
    for block_id, lines in lines_by_block.items():
        if not block_id or len(lines) < 3:
            continue
        codeish = sum(1 for l in lines if _line_is_codeish(l))
        # Whole-block code propagation is conservative: mixed explanatory
        # paragraph + listing blocks are common in books.  Do not turn the
        # explanatory lines into preserved code unless the block is mostly code.
        if codeish >= 3 and codeish >= 0.60 * len(lines):
            code_block_ids.append(block_id)
    if not code_block_ids:
        return 0
    n = 0
    for u in units:
        if not isinstance(u, dict):
            continue
        uid = u.get("unit_id") or ""
        if not any(uid == b or uid.startswith(b + "_") for b in code_block_ids):
            continue
        if u.get("level") in {"char", "word", "span"}:
            continue
        understanding = u.get("understanding") or {}
        if understanding.get("role") in _CODE_KEEP_ROLES:
            continue
        new_role = "code_block" if u.get("level") == "block" else "code_line"
        understanding["role"] = new_role
        ot, sk = ROLE_TO_OBJECT.get(new_role, ("code", "code_line"))
        understanding["object_type"] = ot
        understanding["semantic_kind"] = sk
        u["understanding"] = understanding
        n += 1
    return n


def infer_page_role(role_counts: dict | None, logical_structures: dict | None = None, *, current: str | None = None) -> str | None:
    """Promote a generic page_role to index/toc/table_page from dominant content.

    Logical structures are the reliable signal and take precedence over resolved
    role counts: a ``table_dominant`` layout makes role_resolver tag almost every
    line ``table_body_cell``, which would otherwise drown index/toc pages. So we
    count *real* table cells from detected tables, not the role tally, and let
    index/toc win over table when their logical structures are present.

    Only overrides a generic page_role (None/unknown/body…).
    """
    current_l = str(current or "").lower()
    if current_l not in {"", "none", "unknown", "body", "body_text", "body_text_two_column"}:
        return current
    ls = logical_structures or {}
    index_n = len(ls.get("index_entries") or [])
    toc_n = len(ls.get("toc_entries") or [])
    table_cells = sum(len(t.get("cells") or []) for t in ls.get("tables") or [])
    body_n = len(ls.get("body_paragraphs") or [])

    # Fall back to resolved role counts only when no logical structure exists.
    if not (index_n or toc_n or table_cells or body_n):
        rc = role_counts or {}
        index_n = rc.get("index_entry", 0) + rc.get("index_head_term", 0)
        toc_n = rc.get("toc_entry", 0) + rc.get("toc_entry_title", 0)
        table_cells = rc.get("table_body_cell", 0) + rc.get("table_header_cell", 0)
        body_n = rc.get("body_paragraph", 0)

    # Index and TOC dominate table when their structures are present.
    if index_n >= 10 and index_n >= toc_n and index_n > table_cells:
        return "index"
    if toc_n >= 10 and toc_n > index_n and toc_n > table_cells:
        return "toc"
    if table_cells >= 10 and table_cells > index_n and table_cells > toc_n:
        return "table_page"
    return current


# A PDF drawing is often a table grid, coloured callout, underline or page
# decoration.  It is not sufficient evidence that nearby PDF text is baked into
# a figure.  Raster images and explicitly classified diagrams/charts are.
_FIGURE_REGION_TYPES = ("image_region", "diagram_region", "chart_region")


def _inside_figure_zone(unit: dict) -> bool:
    """True if the unit is dominantly inside a figure/chart/drawing region.
    Body/text regions and incidental overlaps do not count."""
    for m in (unit.get("understanding") or {}).get("region_memberships") or []:
        rt = str(m.get("region_type") or "")
        if not any(k in rt for k in _FIGURE_REGION_TYPES):
            continue
        if m.get("coverage_mode") in {"full_coverage", "dominant_overlap"} or (m.get("overlap_ratio") or 0) >= 0.7:
            return True
    return False


def resolve_unit_role(unit: dict, *, page_intelligence: dict, document_context: dict) -> tuple[str, str, float]:
    understanding = unit.get("understanding") or {}
    current = str(understanding.get("role") or "").strip()
    text = str((unit.get("content") or {}).get("text") or "").strip()
    level = unit.get("level")
    page_role = str(page_intelligence.get("page_role") or "").lower()
    layout_type = str(page_intelligence.get("layout_type") or "").lower()
    resolved = (unit.get("evidence") or {}).get("resolved_understanding") or {}
    resolved_object = str(resolved.get("object_type") or "").lower()

    if text:
        if PUBLISHER_RE.search(text):
            return "publisher_mark", "publisher_mark_pattern", 0.88
        if WATERMARK_RE.search(text):
            return "watermark", "watermark_pattern", 0.88
        if level in {"block", "line", "phrase"} and PAGE_REF_RE.fullmatch(text) and _in_margin(unit, page_intelligence):
            return "page_reference", "page_number_margin", 0.82
        # Strong math wins over any legacy/page-context role: an equation line
        # ("= …", "× 6 ×") must be PRESERVED, never carried into a toc_entry/list
        # role that would translate it and self-collide with its own protection.
        if _is_strong_math(text):
            return "formula_expression", "strong_math_evidence", 0.8
    # Figure/chart internal labels must be resolved before accepting legacy
    # formula/code labels.  Upstream often tags VGG/Conv/Pool labels inside a
    # diagram as formula_expression; if we keep that role the label is both
    # preserved as a visual and translated/repainted elsewhere.  Only captions
    # escape this rule.
    if text and _inside_figure_zone(unit) and len(text.split()) <= 8 and not CAPTION_RE.match(text):
        return "diagram_label", "inside_figure_zone", 0.78
    if current and current not in {"body", "paragraph", "text", "unknown", "None"}:
        normalized = _normalize_legacy_role(current, text, page_role)
        # A legacy "formula" role is the main upstream over-detection source: it
        # fires on prose with an inline "1 × 1" ("1 × 1 convolutional layers are
        # called bottleneck layers"). Strong math already returned above; if the
        # text carries no real math evidence, drop the formula tag and resolve it
        # as ordinary text below — else it becomes a stale formula_unit that
        # preserves (protects) a line the pipeline also translates.
        if normalized == "formula_expression" and not _has_math_evidence(text):
            pass
        elif normalized in {"code_line", "code_block", "command_name"} and not _has_code_evidence(text):
            # Legacy extractors often mark prose that is visually near a listing
            # as code.  Code is non-translatable; accepting that stale label
            # silently drops real sentences from PAGETRANSLATE.  Require code
            # syntax/keywords before preserving it as code.
            pass
        elif normalized == "path" and _looks_like_prose_sentence(text):
            # A sentence containing a URL/path remains prose; the inline URL is
            # protected later by PAGETRANSLATE placeholders.
            pass
        else:
            return normalized, "legacy_role_normalized", 0.82
    if level in {"page", "region", "image", "drawing", "overlay", "table"}:
        return _container_role(level, understanding), "structural_container", 0.85
    if not text:
        return "unknown", "empty_text", 0.2
    # A short text unit sitting INSIDE a figure / chart / drawing zone is a
    # diagram label (axis tick, legend, "car 0.88", "n_units"), not a heading /
    # paragraph / index term. It belongs to the preserved figure pixels, so it is
    # kept, never repainted (that was the source of figure-label collisions).
    if _inside_figure_zone(unit) and len(text.split()) <= 8 and not CAPTION_RE.match(text):
        return "diagram_label", "inside_figure_zone", 0.74
    if page_role == "toc":
        if PAGE_REF_RE.fullmatch(text):
            return "toc_page_reference", "toc_page_reference_pattern", 0.86
        if SECTION_RE.fullmatch(text):
            return "toc_section_number", "toc_section_number_pattern", 0.86
        return "toc_entry_title" if level in {"phrase", "span"} else "toc_entry", "toc_page_context", 0.82
    if page_role == "index":
        if PAGE_REF_RE.fullmatch(text):
            return "index_page_reference", "index_page_reference_pattern", 0.86
        if INDEX_RE.fullmatch(text):
            return "index_entry", "index_pattern", 0.84
        return "index_head_term", "index_page_context", 0.76
    if CAPTION_RE.match(text):
        return ("table_caption" if text.lower().startswith(("table", "tab.")) else "figure_caption"), "caption_pattern", 0.88
    if TOC_RE.match(text):
        return "toc_entry", "toc_pattern", 0.78
    if INDEX_RE.match(text):
        return "index_entry", "index_pattern", 0.78
    if URL_RE.fullmatch(text):
        return "url", "url_pattern", 0.96
    if EMAIL_RE.fullmatch(text):
        return "email", "email_pattern", 0.96
    # Do not classify a whole prose sentence as path/url just because it embeds
    # one protected token.  Inline protection belongs to PAGETRANSLATE.
    if PATH_RE.search(text) and _token_covers_most_text(PATH_RE, text):
        return "path", "path_pattern", 0.92
    if COMMAND_RE.match(text):
        if _has_code_evidence(text):
            return "command_name" if len(text.split()) == 1 else "code_line", "command_pattern", 0.86
    if resolved_object == "formula_expression" and _has_math_evidence(text):
        return "formula_expression", "resolved_formula_evidence", 0.78
    if resolved_object in {"code_line", "code"}:
        return "code_line", "resolved_code_evidence", 0.78
    # Only a real cell is a table cell. A table_dominant layout must NOT turn all
    # editorial text into cells without grid evidence (directive PR-Lot 2).
    if level == "cell":
        return "table_body_cell", "table_context", 0.72
    if level in {"block", "line", "phrase", "span"}:
        heading = _looks_like_heading(unit, text)
        if heading:
            return heading, "heading_style_evidence", 0.74
        # A short line without heading evidence is a paragraph fragment, not a title.
        return "body_paragraph", "textual_default", 0.68
    return "unknown", "no_role_rule_matched", 0.35


_HEADING_NUM_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S")
_MATH_RE = re.compile(r"[=≈≠≤≥±×÷∑∫√∞∂∆∇∏λµπσαβγθΩ]|[A-Za-z0-9)]\s*[=+\-*/^]\s*[A-Za-z0-9(]")


def _has_math_evidence(text: str) -> bool:
    """Real formula, not prose with an inline symbol.

    A formula is either strong math (operator-led / operator-dense, ≤2 words) or
    a line carrying at least two operators with at most two alphabetic words.
    "The 1 × 1 convolutional layer" (3 words, one ×) is prose, not a formula.
    """
    s = str(text or "")
    if not _MATH_RE.search(s):
        return False
    if _is_strong_math(s):
        return True
    words = re.findall(r"[A-Za-z]{2,}", s)
    operators = len(re.findall(r"[=+\-*/^≈≠≤≥±×÷∑∫√]", s))
    return operators >= 2 and len(words) <= 2


def _is_strong_math(text: str) -> bool:
    """Math that must be PRESERVED, not translated (directive: sur-détection amont).

    A formula line/continuation, not prose with an inline symbol:
      - starts with a binary operator ("= …", "× 6 ×", "+ …"), OR
      - carries math operators with at most two alphabetic words.
    Prose with an inline "1 × 1" (≥3 words, no leading operator) is NOT strong
    math and stays translatable.
    """
    s = str(text or "").strip()
    if not s:
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+(?:[,.])?", s):
        return False
    words = re.findall(r"[A-Za-z]{2,}", s)
    # Symbol/Wingdings glyphs (U+F000–U+F0FF) mark a formula fragment (piecewise
    # braces, big operators) ONLY when the line is not prose: a single leading PUA
    # glyph is a list BULLET (" Input image—In filter…") and stays prose.
    if any(0xF000 <= ord(ch) <= 0xF0FF for ch in s) and len(words) <= 2:
        return True
    if not _MATH_RE.search(s):
        return False
    if s[0] in "=×÷±∑∫√≈≠≤≥":
        return True
    operators = len(re.findall(r"[=+\-*/^≈≠≤≥±×÷∑∫√∂]", s))
    return operators >= 1 and len(words) <= 2


def _looks_like_heading(unit: dict, text: str) -> str | None:
    """A heading needs evidence (style/number/caps), not just a short length."""
    words = text.split()
    if not (1 <= len(words) <= 9):
        return None
    if text.rstrip()[-1:] in {".", ",", ";", ":", "?", "!"} and not _HEADING_NUM_RE.match(text):
        return None
    style = (unit.get("visual") or {}).get("style") or {}
    flags = style.get("flags") or {}
    size = style.get("font_size_pt")
    bold = bool(flags.get("bold"))
    upper = bool(flags.get("uppercase")) or (text.isupper() and len(text) > 3)
    numbered = bool(_HEADING_NUM_RE.match(text))
    larger = bool(size and size >= 12.0)
    if numbered:
        return "section_heading"
    if bold or upper or larger:
        return "section_heading" if len(words) <= 9 else "title"
    return None



_PROSE_SENTENCE_END_RE = re.compile(r"[.;:!?)]$|[A-Za-zÀ-ÿ]{3,}\s+[A-Za-zÀ-ÿ]{3,}")
_CODE_LITERAL_RE = re.compile(
    r"""
    (?:
        \b(?:SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|INSERT|UPDATE|DELETE|JOIN|WITH|VALUES|CREATE|ALTER|DROP|UNION|HAVING|LIMIT|RETURN|DEF|CLASS|IMPORT)\b
        | [;{}]
        | ->
        | :=
        | \b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _looks_like_prose_sentence(text: str) -> bool:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if not s:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*", s)
    if len(words) >= 7:
        return True
    if len(words) >= 4 and _PROSE_SENTENCE_END_RE.search(s):
        return True
    return False


def _has_code_evidence(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if _looks_like_prose_sentence(s) and not re.search(r"\b(?:SELECT|CREATE|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN)\b", s, re.IGNORECASE):
        return False
    return bool(_CODE_LITERAL_RE.search(s))


def _token_covers_most_text(pattern: re.Pattern, text: str, *, min_ratio: float = 0.72) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    matches = list(pattern.finditer(s))
    if not matches:
        return False
    covered = sum(max(0, m.end() - m.start()) for m in matches)
    return covered / max(1, len(s)) >= min_ratio

def _normalize_legacy_role(role: str, text: str, page_role: str) -> str:
    role_l = role.lower()
    if role_l in {"body", "paragraph", "text"}:
        return "body_paragraph"
    if role_l in ROLE_TO_OBJECT:
        return role_l
    if "caption" in role_l:
        return "table_caption" if text.lower().startswith(("table", "tab.")) else "figure_caption"
    if "heading" in role_l or "title" in role_l:
        return "title" if "title" in role_l else "section_heading"
    if "footer" in role_l:
        return "page_footer"
    if "header" in role_l:
        return "page_header"
    if "toc" in role_l:
        return "toc_entry_title" if "title" in role_l else "toc_entry"
    if "index" in role_l:
        return "index_page_reference" if "reference" in role_l else "index_entry"
    if "formula" in role_l or "equation" in role_l:
        return "formula_expression"
    if "code" in role_l:
        return "code_line"
    if page_role == "toc":
        return "toc_entry"
    if page_role == "index":
        return "index_entry"
    return "unknown"


def _container_role(level: str, understanding: dict) -> str:
    object_type = str(understanding.get("object_type") or "").lower()
    if "table" in object_type:
        return "table_body_cell" if level == "cell" else "unknown"
    if "publisher" in object_type or "logo" in object_type:
        return "publisher_mark"
    return "unknown"


def _in_margin(unit: dict, page_intelligence: dict) -> bool:
    bbox = (unit.get("geometry") or {}).get("bbox")
    page_geometry = page_intelligence.get("page_geometry") or {}
    height = float(page_geometry.get("height") or 0.0)
    dpi = float(page_geometry.get("render_dpi") or 72.0)
    if height and dpi:
        height = height * 72.0 / dpi
    if not height or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    y0 = float(bbox[1])
    y1 = float(bbox[3])
    return y1 < height * 0.12 or y0 > height * 0.88
