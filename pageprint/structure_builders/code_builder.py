from __future__ import annotations

import re

from .common import bbox_of, eligible_text_units, role_of, text_of


CODE_ROLES = {"code_block", "code_line", "command_name", "path"}
FIGURE_REGION_MARKERS = ("image_region", "drawing_region", "diagram", "chart", "figure")


_CODE_SIGNAL_RE = re.compile(
    r"\b(?:SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|INSERT|UPDATE|DELETE|JOIN|WITH|VALUES|CREATE|ALTER|DROP|UNION|HAVING|LIMIT|def|class|import|return)\b|[;{}]|:=|->|\b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)",
    re.IGNORECASE,
)


def _looks_like_prose(text: str) -> bool:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*", s)
    return len(words) >= 7 or (len(words) >= 4 and s.endswith(('.', ':', ';', '?', '!', ')')))


def _has_code_evidence(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if _looks_like_prose(s) and not re.search(r"\b(?:SELECT|CREATE|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN)\b", s, re.IGNORECASE):
        return False
    return bool(_CODE_SIGNAL_RE.search(s))


def _inside_preserved_figure(unit: dict) -> bool:
    for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
        region_type = str(membership.get("region_type") or "").lower()
        if not any(marker in region_type for marker in FIGURE_REGION_MARKERS):
            continue
        if membership.get("coverage_mode") in {"full_coverage", "dominant_overlap"}:
            return True
        try:
            if float(membership.get("overlap_ratio") or 0.0) >= 0.55:
                return True
        except Exception:
            pass
    return False


def build_code_blocks(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    candidates = [
        u for u in eligible_text_units(units)
        if role_of(u) in CODE_ROLES
        and not _inside_preserved_figure(u)
        and _has_code_evidence(text_of(u))
    ]
    code_units = _prefer_code_granularity(candidates, by_id)
    return [
        {
            "logical_unit_id": f"code_{idx:04d}",
            "type": "code_line" if role_of(unit) != "code_block" else "code_block",
            "text": text_of(unit),
            "source_unit_ids": [unit["unit_id"]],
            "translation_mode": "preserve_text_exactly",
            "bbox": bbox_of(unit),
        }
        for idx, unit in enumerate(code_units, start=1)
    ]


def _prefer_code_granularity(candidates: list[dict], by_id: dict[str, dict]) -> list[dict]:
    """Emit one code unit per branch.

    Code/listing extraction is atomic. If a block is already represented by
    code lines, do not also emit phrase/span copies of the same text. This avoids
    both translation leakage and duplicate preserved overlays.
    """
    ids = {u.get("unit_id") for u in candidates}
    has_code_child: dict[str, bool] = {str(u.get("unit_id")): False for u in candidates if u.get("unit_id")}
    for unit in candidates:
        cursor = unit
        while cursor.get("parent_id") in by_id:
            parent_id = cursor.get("parent_id")
            if parent_id in ids:
                has_code_child[str(parent_id)] = True
            cursor = by_id[parent_id]

    selected_ids: set[str] = set()
    for unit in candidates:
        uid = str(unit.get("unit_id") or "")
        if not uid:
            continue
        level = unit.get("level")
        if level in {"phrase", "span"} and _has_text_parent(unit, by_id, ids):
            continue
        if level == "block" and has_code_child.get(uid):
            continue
        selected_ids.add(uid)
    return [u for u in candidates if u.get("unit_id") in selected_ids]


def _has_text_parent(unit: dict, by_id: dict[str, dict], candidate_ids: set[str]) -> bool:
    cursor = unit
    while cursor.get("parent_id") in by_id:
        parent_id = cursor.get("parent_id")
        if parent_id in candidate_ids:
            return True
        cursor = by_id[parent_id]
    return False
