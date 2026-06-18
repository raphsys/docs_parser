"""Render-time text sanitizer for translated prose.

This is deliberately narrow: it removes assistant/tool artefacts and internal
markup that must never reach the visual renderer, without rewriting legitimate
technical content such as formulas, code identifiers, SQL, or punctuation.
"""
from __future__ import annotations

import re

_INTERNAL_REF_RE = re.compile(r"\ue200[^\ue201\n]{0,240}\ue201")
_ID_ATTR_RE = re.compile(r"\s+id\s*=\s*(['\"])[^'\"]{1,80}\1")
_BRACKET_ID_RE = re.compile(r"\[[^\]\n]{0,120}\bid\s*=\s*['\"][^'\"]+['\"][^\]\n]{0,120}\]")
_TAG_WITH_ID_RE = re.compile(r"<[^>\n]{0,160}\bid\s*=\s*['\"][^'\"]+['\"][^>\n]{0,160}>")
_WRITING_FENCE_RE = re.compile(r"^\s*:::\s*(?:writing)?\{[^\n]*\}\s*$", re.I | re.M)
_CODE_FENCE_RE = re.compile(r"^\s*```+\s*$", re.M)
_META_SENTENCES = (
    "je garde la précision",
    "j'ai gardé la précision",
    "j’ai gardé la précision",
    "je conserve la précision",
    "i keep the precision",
    "i kept the precision",
    "traduction fidèle",
    "voici la traduction",
)


def _drop_meta_sentences(text: str) -> str:
    out = text
    for phrase in _META_SENTENCES:
        # Remove the common assistant meta-fragment wherever it was injected.
        # These sentences are not document content and have no valid technical
        # meaning in source pages.
        out = re.sub(rf"(?i)\b{re.escape(phrase)}\b\s*[.;!?]?", " ", out)
    return out


def sanitize_render_text(text: str | None) -> tuple[str, list[str]]:
    """Return clean text and a list of sanitizer findings.

    The renderer receives only the cleaned value.  Findings are intentionally
    short strings because `compose_block()` stores findings as compact labels.
    """
    original = str(text or "")
    cleaned = original
    findings: list[str] = []

    transforms = [
        (_INTERNAL_REF_RE, ""),
        (_BRACKET_ID_RE, ""),
        (_TAG_WITH_ID_RE, ""),
        (_WRITING_FENCE_RE, ""),
        (_CODE_FENCE_RE, ""),
    ]
    for rx, repl in transforms:
        new = rx.sub(repl, cleaned)
        if new != cleaned:
            findings.append("render_text_internal_markup_removed")
            cleaned = new

    new = _ID_ATTR_RE.sub("", cleaned)
    if new != cleaned:
        findings.append("render_text_id_attribute_removed")
        cleaned = new

    new = _drop_meta_sentences(cleaned)
    if new != cleaned:
        findings.append("render_text_assistant_meta_removed")
        cleaned = new

    # Collapse horizontal whitespace but keep text readable.  Renderers do not
    # understand hard line breaks inside one run anyway.
    cleaned = " ".join(cleaned.replace("\u00a0", " ").split())
    if cleaned != " ".join(original.replace("\u00a0", " ").split()):
        findings.append("render_text_sanitized")
    return cleaned, list(dict.fromkeys(findings))
