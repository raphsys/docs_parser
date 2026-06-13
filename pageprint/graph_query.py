"""Small graph/role query helpers shared by PAGEPRINT/PAGETRANSLATE fallback."""

from __future__ import annotations


STRUCTURAL_ROLES = {
    "toc_entry",
    "toc_entry_title",
    "toc_page_reference",
    "index_entry",
    "index_page_reference",
    "table_header_cell",
    "table_body_cell",
    "table_numeric_cell",
    "code_line",
    "command_name",
    "path",
    "formula_expression",
    "figure_caption",
    "table_caption",
    "list_item",
}


def role_of(unit: dict) -> str:
    return str((unit.get("understanding") or {}).get("role") or unit.get("role") or "unknown")


def is_same_paragraph(unit_a: dict, unit_b: dict) -> bool:
    return role_of(unit_a) == role_of(unit_b) == "body_paragraph" and unit_a.get("block_id") == unit_b.get("block_id")


def is_same_list_item(unit_a: dict, unit_b: dict) -> bool:
    return role_of(unit_a) == role_of(unit_b) == "list_item" and unit_a.get("block_id") == unit_b.get("block_id")


def is_same_table_cell(unit_a: dict, unit_b: dict) -> bool:
    return role_of(unit_a).startswith("table_") and role_of(unit_b).startswith("table_") and unit_a.get("unit_id") == unit_b.get("unit_id")


def is_same_toc_entry(unit_a: dict, unit_b: dict) -> bool:
    return role_of(unit_a).startswith("toc_") and role_of(unit_b).startswith("toc_") and unit_a.get("block_id") == unit_b.get("block_id")


def is_same_index_entry(unit_a: dict, unit_b: dict) -> bool:
    return role_of(unit_a).startswith("index_") and role_of(unit_b).startswith("index_") and unit_a.get("block_id") == unit_b.get("block_id")


def is_caption_of(unit: dict, figure_unit: dict) -> bool:
    return "caption" in role_of(unit) and figure_unit.get("level") in {"image", "drawing", "region"}


def is_inside_region(unit: dict, region_type: str) -> bool:
    return any(m.get("region_type") == region_type for m in (unit.get("understanding") or {}).get("region_memberships") or [])


def has_partial_protected_overlap(unit: dict) -> bool:
    for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
        if str(membership.get("region_type") or "").endswith("_candidate_region") and membership.get("coverage_mode") == "partial_inline":
            return True
    return False


def nearest_heading(unit: dict) -> None:
    return None


def reading_predecessor(unit: dict) -> str | None:
    return (unit.get("relations") or {}).get("previous_unit_id")


def reading_successor(unit: dict) -> str | None:
    return (unit.get("relations") or {}).get("next_unit_id")


def can_merge_for_translation(unit_a: dict, unit_b: dict) -> bool:
    role_a = role_of(unit_a)
    role_b = role_of(unit_b)
    if role_a in STRUCTURAL_ROLES or role_b in STRUCTURAL_ROLES:
        return False
    if role_a != role_b:
        return False
    if role_a not in {"body_paragraph", "paragraph", "body"}:
        return False
    if unit_a.get("block_id") and unit_b.get("block_id") and unit_a.get("block_id") != unit_b.get("block_id"):
        return False
    return True
