from __future__ import annotations

import re
from typing import Any


SCHEMA_VERSION = "document_object_contract.v1"


PRESERVE_OBJECT_TYPES = {
    "reference_link",
    "url",
    "web_url",
    "email_address",
    "doi_reference",
    "arxiv_reference",
    "code_block",
    "code_line",
    "inline_code",
    "technical_identifier",
    "formula_block",
    "formula_line",
    "formula_symbol",
    "formula_equation",
    "inline_formula",
    "inline_formula_cluster",
    "chemical_formula",
    "page_number",
    "abbreviation",
}


VISUAL_OBJECT_TYPES = {
    "figure_region",
    "image_region",
    "drawing_region",
    "chart_region",
    "dense_diagram_region",
    "complex_vector_region",
    "clipping_region",
    "mask_region",
    "overlay_stack_region",
    "seal_region",
}


TABLE_OBJECT_TYPES = {
    "table_block",
    "table_cell",
    "table_cell_micro",
    "table_cell_text",
    "table_cell_numeric",
    "table_cell_symbolic",
    "table_row",
    "dense_table_region",
}


VISUAL_LABEL_OBJECT_TYPES = {
    "figure_axis_label",
    "figure_label",
    "axis_label",
    "diagram_label",
    "chart_label",
    "legend_label",
    "short_label",
    "micro_label",
}


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


_INLINE_SEGMENT_RE = re.compile(
    r"("
    r"https?://[^\s<>\])]+"
    r"|www\.[^\s<>\])]+"
    r"|[\w\.-]+@[\w\.-]+\.\w+"
    r"|doi:\s*\S+"
    r"|arxiv:\s*\S+"
    r"|10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
    r"|(?:/[A-Za-z0-9_.\-]+){2,}/?"
    r"|[A-Za-z0-9_.\-]+\.(?:app|dmg|exe|py|json|yaml|yml|csv|txt|md|pdf|docx|xml|html|js|ts|sql)"
    r"|\b(?:sudo|mkdir|echo|tee|postgresapp|pgAdmin|PostgreSQL|Postgres\.app|ReLU|CNN|ANN|DL|CV|SQL)\b"
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\([^)\n]{0,64}\)"
    r"|\b[A-Za-z0-9]+\s*[=<>±×÷]\s*[-+A-Za-z0-9_\\^{}()./]+\b"
    r"|[A-Za-z0-9]+(?:_[A-Za-z0-9{}]+|\^[A-Za-z0-9{}]+)+"
    r"|(?:\d+(?:[.,]\d+)*|\d+/\d+)"
    r")",
    flags=re.IGNORECASE,
)


def text_from_unit(unit: dict | None) -> str:
    if not isinstance(unit, dict):
        return ""
    for key in ("translated_text", "line_text", "text", "texte", "raw_text"):
        value = clean_text(unit.get(key))
        if value:
            return value
    parts: list[str] = []
    for line in unit.get("lines") or []:
        if isinstance(line, dict):
            parts.append(text_from_unit(line))
    for phrase in unit.get("phrases") or []:
        if isinstance(phrase, dict):
            parts.append(text_from_unit(phrase))
    for span in unit.get("spans") or []:
        if isinstance(span, dict):
            parts.append(text_from_unit(span))
    return clean_text(" ".join(part for part in parts if part))


def _infer_inline_type(text: str) -> str:
    s = clean_text(text)
    if not s:
        return "plain_text"
    if re.fullmatch(r"https?://[^\s<>\])]+|www\.[^\s<>\])]+", s, flags=re.IGNORECASE):
        return "web_url"
    if re.fullmatch(r"[\w\.-]+@[\w\.-]+\.\w+", s):
        return "email_address"
    if re.fullmatch(r"doi:\s*\S+", s, flags=re.IGNORECASE):
        return "doi_reference"
    if re.fullmatch(r"arxiv:\s*\S+", s, flags=re.IGNORECASE):
        return "arxiv_reference"
    if re.fullmatch(r"[A-Za-z0-9_.\-]+\.(?:app|dmg|exe|py|json|yaml|yml|csv|txt|md|pdf|docx|xml|html|js|ts|sql)", s, flags=re.IGNORECASE):
        return "technical_identifier"
    if re.fullmatch(r"\b(?:sudo|mkdir|echo|tee|postgresapp|pgAdmin|PostgreSQL|Postgres\.app|ReLU|CNN|ANN|DL|CV|SQL)\b", s, flags=re.IGNORECASE):
        return "technical_identifier"
    if re.fullmatch(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\([^)\n]{0,64}\)", s):
        return "function_call"
    if re.fullmatch(r"\b[A-Za-z0-9]+\s*[=<>±×÷]\s*[-+A-Za-z0-9_\\^{}()./]+\b", s):
        return "inline_formula"
    if re.fullmatch(r"[A-Za-z0-9]+(?:_[A-Za-z0-9{}]+|\^[A-Za-z0-9{}]+)+", s):
        return "technical_identifier"
    if re.fullmatch(r"(?:\d+(?:[.,]\d+)*|\d+/\d+)", s):
        return "measurement_value"
    if re.fullmatch(r"[A-Z]{2,8}", s):
        return "abbreviation"
    return "plain_text"


def _infer_inline_subtype(text: str) -> str:
    inline_type = _infer_inline_type(text)
    if inline_type in {"technical_identifier", "inline_formula"}:
        return inline_type
    return inline_type if inline_type != "plain_text" else ""


def _translation_hint_for_inline(text: str) -> str:
    inline_type = _infer_inline_type(text)
    if inline_type in {"web_url", "email_address", "doi_reference", "arxiv_reference", "technical_identifier", "inline_formula"}:
        return "preserve"
    return "translate"


def extract_inline_segments(text: Any) -> list[dict[str, Any]]:
    src = clean_text(text)
    if not src:
        return []
    segments: list[dict[str, Any]] = []
    cursor = 0
    for match in _INLINE_SEGMENT_RE.finditer(src):
        start, end = match.span()
        if start > cursor:
            plain = clean_text(src[cursor:start])
            if plain:
                segments.append(
                    {
                        "start": cursor,
                        "end": start,
                        "text": plain,
                        "inline_object_type": "plain_text",
                        "inline_object_subtype": "",
                        "translation_hint": "translate",
                        "preserve_exact_text": False,
                        "reason": "plain_gap",
                    }
                )
        chunk = clean_text(match.group(0))
        if chunk:
            translation_hint = _translation_hint_for_inline(chunk)
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": chunk,
                    "inline_object_type": _infer_inline_type(chunk),
                    "inline_object_subtype": _infer_inline_subtype(chunk),
                    "translation_hint": translation_hint,
                    "preserve_exact_text": translation_hint == "preserve",
                    "reason": "inline_pattern",
                }
            )
        cursor = end
    if cursor < len(src):
        tail = clean_text(src[cursor:])
        if tail:
            segments.append(
                {
                    "start": cursor,
                    "end": len(src),
                    "text": tail,
                    "inline_object_type": "plain_text",
                    "inline_object_subtype": "",
                    "translation_hint": "translate",
                    "preserve_exact_text": False,
                    "reason": "plain_tail",
                }
            )
    if not segments:
        segments.append(
            {
                "start": 0,
                "end": len(src),
                "text": src,
                "inline_object_type": "plain_text",
                "inline_object_subtype": "",
                "translation_hint": "translate",
                "preserve_exact_text": False,
                "reason": "default_plain",
            }
        )
    return segments


def inline_structure_for_text(text: Any) -> dict[str, Any]:
    segments = extract_inline_segments(text)
    counts: dict[str, int] = {}
    subtype_counts: dict[str, int] = {}
    special = []
    for seg in segments:
        inline_type = clean_text(seg.get("inline_object_type") or "")
        if inline_type and inline_type != "plain_text":
            counts[inline_type] = counts.get(inline_type, 0) + 1
            subtype = clean_text(seg.get("inline_object_subtype") or "")
            if subtype:
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
            special.append(seg)
    dominant_type = ""
    dominant_subtype = ""
    if counts:
        dominant_type = max(sorted(counts), key=lambda key: counts[key])
    if subtype_counts:
        dominant_subtype = max(sorted(subtype_counts), key=lambda key: subtype_counts[key])
    return {
        "segments": segments,
        "has_special_inline_objects": bool(special),
        "inline_object_counts": dict(sorted(counts.items())),
        "inline_object_subtype_counts": dict(sorted(subtype_counts.items())),
        "dominant_inline_object_type": dominant_type,
        "dominant_inline_object_subtype": dominant_subtype,
    }


def object_context(unit: dict | None) -> dict[str, str]:
    unit = unit if isinstance(unit, dict) else {}
    payload = unit.get("object_comprehension") if isinstance(unit.get("object_comprehension"), dict) else {}
    return {
        "object_class": clean_text(unit.get("object_class") or payload.get("object_class")).lower(),
        "object_type": clean_text(unit.get("object_type") or payload.get("object_type")).lower(),
        "object_subtype": clean_text(unit.get("object_subtype") or payload.get("object_subtype")).lower(),
        "inline_object_type": clean_text(unit.get("inline_object_type") or payload.get("inline_object_type")).lower(),
        "inline_object_subtype": clean_text(unit.get("inline_object_subtype") or payload.get("inline_object_subtype")).lower(),
        "role": clean_text(unit.get("role")).lower(),
        "density_profile": clean_text(unit.get("density_profile") or payload.get("density_profile")).lower(),
    }


def looks_like_toc(text: str, unit: dict | None = None) -> bool:
    text = clean_text(text)
    unit = unit if isinstance(unit, dict) else {}
    lines = [line for line in unit.get("lines") or [] if isinstance(line, dict)]
    if re.search(r"(?:\.\s*){6,}", text):
        return True
    leader_lines = 0
    for line in lines:
        line_text = clean_text(line.get("line_text") or line.get("text") or line.get("translated_text"))
        if re.search(r"(?:\.\s*){4,}", line_text):
            leader_lines += 1
    return len(lines) >= 8 and leader_lines >= 3


def parse_toc_line(text: str) -> dict[str, str]:
    text = clean_text(text)
    if not text:
        return {"kind": "empty", "prefix": "", "title": "", "leader": "", "page": "", "text": ""}
    match = re.match(
        r"^(?P<prefix>(?:\d+(?:[.,]\d+)*\.?|[A-Z]\.?)\s+)?(?P<title>.*?)(?P<leader>(?:\s*\.\s*){3,})(?P<page>\s*\d+\s*)?$",
        text,
    )
    if not match:
        return {"kind": "plain", "prefix": "", "title": text, "leader": "", "page": "", "text": text}
    prefix = clean_text(match.group("prefix") or "")
    if prefix:
        prefix = re.sub(r"(?<=\d),(?=\d)", ".", prefix)
        prefix = re.sub(r"\s+", " ", prefix)
    return {
        "kind": "toc_leader_row",
        "prefix": prefix,
        "title": clean_text(match.group("title") or ""),
        "leader": clean_text(match.group("leader") or ""),
        "page": clean_text(match.group("page") or ""),
        "text": text,
    }


def _base_policy(object_class: str, object_type: str) -> dict[str, Any]:
    if object_type in VISUAL_OBJECT_TYPES or object_class == "visual":
        return {
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "background_only",
            "reinject_mode": "source_overlay",
            "contract_key": "figure_region",
            "geometry_mode": "source_overlay",
            "preserve_visual_structure": True,
        }
    if object_type in PRESERVE_OBJECT_TYPES or object_class in {"technical", "formula"}:
        contract_key = "formula_block" if object_class == "formula" or "formula" in object_type else "code_block"
        if object_type in {"reference_link", "url", "web_url", "email_address", "doi_reference", "arxiv_reference"}:
            contract_key = "url_reference"
        return {
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "fixed_preserve",
            "reinject_mode": "fixed_overlay",
            "contract_key": contract_key,
            "geometry_mode": "fixed_slot",
            "preserve_visual_structure": True,
        }
    if object_type in TABLE_OBJECT_TYPES or object_class == "tabular":
        return {
            "translatable": True,
            "translation_strategy": "layout_constrained",
            "render_policy": "cell_locked",
            "reinject_mode": "cell_locked",
            "contract_key": "table_cell",
            "geometry_mode": "cell_locked",
            "preserve_visual_structure": True,
        }
    if object_type in VISUAL_LABEL_OBJECT_TYPES or object_class == "visual_label":
        return {
            "translatable": True,
            "translation_strategy": "layout_constrained",
            "render_policy": "anchored_text",
            "reinject_mode": "anchored_text",
            "contract_key": "figure_label",
            "geometry_mode": "anchored",
            "preserve_visual_structure": True,
        }
    return {
        "translatable": True,
        "translation_strategy": "semantic_reflow",
        "render_policy": "translated_editorial",
        "reinject_mode": "paragraph_reflow",
        "contract_key": "paragraph",
        "geometry_mode": "paragraph_reflow",
        "preserve_visual_structure": False,
    }


def build_document_object_contract(unit: dict | None, *, level: str = "", text: str = "") -> dict[str, Any]:
    unit = unit if isinstance(unit, dict) else {}
    ctx = object_context(unit)
    unit_text = clean_text(text) or text_from_unit(unit)
    inline_structure = inline_structure_for_text(unit_text)
    object_type = ctx["object_type"] or "plain_text"
    object_class = ctx["object_class"] or "editorial"
    policy = _base_policy(object_class, object_type)
    structural_kind = "generic_text"

    if object_type in {"toc_entry", "toc_leader"} or ctx["role"] in {"toc", "toc_entry"} or looks_like_toc(unit_text, unit):
        structural_kind = "toc"
        object_type = "toc_entry"
        object_class = "navigational"
        policy.update(
            {
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "render_policy": "toc_row_locked",
                "reinject_mode": "toc_row_locked",
                "contract_key": "toc_entry",
                "geometry_mode": "toc_row_locked",
                "preserve_visual_structure": True,
            }
        )
    elif object_type in TABLE_OBJECT_TYPES or object_class == "tabular":
        structural_kind = "table"
    elif object_type in VISUAL_OBJECT_TYPES or object_class == "visual":
        structural_kind = "visual"
    elif object_type in PRESERVE_OBJECT_TYPES or object_class in {"technical", "formula"}:
        structural_kind = "immutable_inline" if level in {"span", "phrase"} else "immutable_block"
    protection = list(_translation_protection_for(policy["contract_key"]))
    if inline_structure.get("has_special_inline_objects") and "special_inline" not in protection:
        protection.append("special_inline")

    return {
        "schema_version": SCHEMA_VERSION,
        "level": level or clean_text(unit.get("structural_context", {}).get("level") if isinstance(unit.get("structural_context"), dict) else ""),
        "object_class": object_class,
        "object_type": object_type,
        "object_subtype": ctx["object_subtype"],
        "inline_object_type": ctx["inline_object_type"],
        "inline_object_subtype": ctx["inline_object_subtype"],
        "structural_kind": structural_kind,
        "translation": {
            "translatable": bool(policy["translatable"]),
            "strategy": policy["translation_strategy"],
            "coverage_required": "strict",
            "protection": protection,
        },
        "reconstruction": {
            "contract_key": policy["contract_key"],
            "render_policy": policy["render_policy"],
            "reinject_mode": policy["reinject_mode"],
            "geometry_mode": policy["geometry_mode"],
            "preserve_visual_structure": bool(policy["preserve_visual_structure"]),
            "preserve_line_breaks": policy["contract_key"] in {"toc_entry", "table_cell", "figure_label", "url_reference", "code_block", "formula_block"},
        },
        "inline_structure": inline_structure,
        "visual_structure": _visual_structure_for(object_type, unit_text),
    }


def _translation_protection_for(contract_key: str) -> list[str]:
    if contract_key == "toc_entry":
        return ["toc_numbering", "toc_leaders", "page_numbers", "special_inline"]
    if contract_key == "table_cell":
        return ["table_structure", "cell_boundaries", "special_inline"]
    if contract_key in {"code_block", "formula_block", "inline_formula", "url_reference"}:
        return ["reserved_inline", "technical_tokens"]
    if contract_key == "figure_region":
        return ["visual_content", "all_inline_text"]
    return ["special_inline", "reserved_tokens"]


def _visual_structure_for(object_type: str, text: str) -> dict[str, Any]:
    if object_type == "toc_entry" or looks_like_toc(text):
        parsed = parse_toc_line(text)
        return {
            "kind": "toc_row",
            "segments": ["prefix", "title", "leader", "page"],
            "leader_pattern": "dot_leader" if parsed.get("leader") else "",
            "parsed_preview": parsed,
        }
    return {"kind": "text"}


def apply_contract_to_unit(unit: dict, *, level: str = "", text: str = "") -> dict[str, Any]:
    contract = build_document_object_contract(unit, level=level, text=text)
    unit["document_object_contract"] = contract
    translation = contract["translation"]
    reconstruction = contract["reconstruction"]
    unit["translatable"] = bool(translation["translatable"])
    unit["translation_strategy"] = translation["strategy"]
    unit["coverage_required"] = translation["coverage_required"]
    unit["render_policy"] = reconstruction["render_policy"]
    unit["translation_policy"] = {
        **dict(unit.get("translation_policy") or {}),
        "translatable": bool(translation["translatable"]),
        "translation_strategy": translation["strategy"],
        "coverage_required": translation["coverage_required"],
        "render_policy": reconstruction["render_policy"],
        "translation_protection": list(translation.get("protection") or []),
        "reinject_mode": reconstruction["reinject_mode"],
        "contract_key": reconstruction["contract_key"],
        "document_object_contract_schema": contract["schema_version"],
    }
    unit["inline_structure"] = dict(contract.get("inline_structure") or {})
    unit["has_special_inline_objects"] = bool((contract.get("inline_structure") or {}).get("has_special_inline_objects"))
    return contract
