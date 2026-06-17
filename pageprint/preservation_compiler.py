"""Compile PAGEPRINT preservation modes from resolved roles/evidence."""

from __future__ import annotations

import re


PRESERVATION_MODES = {
    "none",
    "protect_token_inside_translation",
    "preserve_text_exactly",
    "preserve_as_visual_overlay",
    "exclude_as_artifact",
}

TOKEN_PROTECT_ROLES = {
    "author_name",
}

UPPERCASE_TITLE_ROLES = {
    "title",
    "toc_title",
    "toc_entry_title",
    "section_heading",
    "subsection_heading",
    "chapter_heading",
    "body_heading",
}

CONFIRMED_TECHNICAL_ACRONYMS = {
    "MLP",
    "CNN",
    "SQL",
    "API",
    "OCR",
    "GPU",
    "CPU",
    "PDF",
    "JSON",
    "XML",
    "HTML",
    "CSS",
    "ReLU",
}

TEXT_EXACT_ROLES = {
    "page_reference",
    "section_number",
    "list_marker",
    "toc_section_number",
    "toc_page_reference",
    "toc_bullet_marker",
    "index_page_reference",
    "command_name",
    "path",
    "file_name",
    "url",
    "email",
    "code_line",
    "code_token",
}

VISUAL_OVERLAY_ROLES = {
    "formula_expression",
    # A label baked inside a figure/chart belongs to the preserved figure pixels;
    # keep it as a visual overlay (never translated, never repainted).
    "diagram_label",
}

ARTIFACT_ROLES = {
    "watermark",
    "publisher_mark",
}

ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9&./+-]{1,12}$")
PROPER_NAME_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$")


def compile_preservation(units: list[dict], *, page_intelligence: dict | None = None) -> dict:
    """Compile preservation policy for every unit in-place."""
    counts = {mode: 0 for mode in PRESERVATION_MODES}
    unit_modes: dict[str, dict] = {}
    for unit in units:
        policy = compile_unit_preservation(unit, page_intelligence=page_intelligence or {})
        unit.setdefault("policy", {})["preservation_mode"] = policy["mode"]
        unit["policy"]["preservation_reason"] = policy["reason"]
        unit["policy"]["protected_tokens"] = policy.get("protected_tokens") or []
        unit["preservation_policy"] = policy
        counts[policy["mode"]] += 1
        unit_modes[unit.get("unit_id")] = policy
    return {
        "schema_version": "pageprint.preservation.v1",
        "counts": counts,
        "unit_preservation": unit_modes,
    }


def compile_unit_preservation(unit: dict, *, page_intelligence: dict) -> dict:
    role = str((unit.get("understanding") or {}).get("role") or "unknown")
    object_type = str((unit.get("understanding") or {}).get("object_type") or "")
    semantic_kind = str((unit.get("understanding") or {}).get("semantic_kind") or "")
    text = str((unit.get("content") or {}).get("text") or "").strip()
    tags = {role, object_type, semantic_kind}

    if tags & ARTIFACT_ROLES:
        return _policy("exclude_as_artifact", "artifact_role")
    if role in VISUAL_OVERLAY_ROLES:
        if _formula_can_be_redrawn_exactly(text):
            return _policy("preserve_text_exactly", "simple_formula_text")
        return _policy("preserve_as_visual_overlay", "formula_expression")
    if role in TEXT_EXACT_ROLES:
        return _policy("preserve_text_exactly", f"{role}_role")
    if role in TOKEN_PROTECT_ROLES:
        return _policy("protect_token_inside_translation", f"{role}_role", [text] if text else [])
    protected_tokens = _inline_protected_tokens(text, role=role)
    if protected_tokens:
        return _policy("protect_token_inside_translation", "inline_protected_tokens", protected_tokens)
    if object_type in {"publisher_mark", "watermark"}:
        return _policy("exclude_as_artifact", "artifact_object_type")
    return _policy("none", "natural_text_or_unclassified")


def _policy(mode: str, reason: str, protected_tokens: list[str] | None = None) -> dict:
    return {
        "mode": mode,
        "reason": reason,
        "protected_tokens": list(dict.fromkeys(t for t in (protected_tokens or []) if t)),
    }


def _inline_protected_tokens(text: str, *, role: str = "unknown") -> list[str]:
    if not text:
        return []
    tokens = []
    for raw in re.findall(r"\b[A-Z0-9][A-Z0-9&./+-]{1,12}\b", text):
        if _is_protected_acronym(raw, role=role):
            tokens.append(raw)
    for proper in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text):
        if PROPER_NAME_RE.fullmatch(proper):
            tokens.append(proper)
    for path in re.findall(r"(?:[A-Za-z]:\\|/[\w.-]+/|\.{1,2}/)[^\s]+", text):
        tokens.append(path)
    return list(dict.fromkeys(tokens))


def _is_protected_acronym(token: str, *, role: str) -> bool:
    if token in CONFIRMED_TECHNICAL_ACRONYMS:
        return True
    if role in UPPERCASE_TITLE_ROLES and token.isupper() and len(token) > 3:
        return False
    if not ACRONYM_RE.fullmatch(token):
        return False
    if token.isupper() and 2 <= len(token) <= 5:
        return True
    return bool(any(ch.isdigit() for ch in token) and len(token) <= 8)


def _formula_can_be_redrawn_exactly(text: str) -> bool:
    return bool(text and len(text) <= 80 and "\n" not in text)
