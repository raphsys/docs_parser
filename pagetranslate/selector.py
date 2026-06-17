"""Translation unit selection from PagePrint INPUT_DATA."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .schema import AUXILIARY_TEXT_LEVELS, PRIMARY_TEXT_LEVELS
from .text_utils import ancestor_id, bbox_union, normalize_spaces, reading_order, unit_text


EXCLUDED_CLASSES = {
    "publisher_mark",
    "author_name",
    "code",
    "formula",
    "url",
    "doi",
    "acronym",
    "reference",
    "references",
    "bibliography",
    "citation",
    "reference_link",
    "page_number",
    "diagram_label",
    "axis_label",
    "legend_label",
    "chart_tick",
    "figure_label",
}

EXCLUDED_STRATEGIES = {"exact_preserve", "keep_original", "background_only"}

URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DOI_RE = re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+\b", re.IGNORECASE)
PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/[\w.-]+/|\.{1,2}/)[^\s]+")
REFERENCE_RE = re.compile(r"^\s*(?:\[\d+(?:,\s*\d+)*\]|\(\d{4}[a-z]?\)|references?)\s*$", re.IGNORECASE)
ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9&./+-]{1,12}$")
FORMULA_RE = re.compile(r"^(?:[A-Za-z]\w*\s*)?[=≈≃<>≤≥∑∫√]|.*(?:\b[a-zA-Z]\s*[=+*/^]\s*[\w(]).*")


def select_translation_units(input_data: dict) -> list[dict]:
    """Select semantic units first, then fall back by block.

    Priority is semantic_phrase > semantic_group > phrase > line > block.
    Visual lines are used only inside blocks where no higher semantic/phrase
    unit exists. Word/char are never translation units.
    """
    units_by_id = {
        unit.get("unit_id"): unit
        for unit in input_data.get("units") or []
        if isinstance(unit, dict) and unit.get("unit_id")
    }
    legacy_id_index = _legacy_id_index(units_by_id)

    selected: list[dict] = []
    semantic_units, blocked_source_ids, blocked_block_ids = _select_semantic_system_units(
        input_data,
        units_by_id,
        legacy_id_index,
    )
    if semantic_units:
        selected.extend(semantic_units)

    semantic_source_ids = {
        source_id
        for item in semantic_units
        for source_id in item.get("source_unit_ids") or []
    }
    selected_blocks = {
        item.get("block_id")
        for item in semantic_units
        if item.get("block_id")
    }
    selected.extend(_select_pageprint_units_by_block(
        input_data,
        units_by_id,
        selected_blocks,
        semantic_source_ids,
        blocked_source_ids,
        blocked_block_ids,
    ))

    selected.sort(key=lambda item: item.get("reading_order_index") or 0)
    for idx, item in enumerate(selected, start=1):
        item["translation_unit_id"] = f"tu_{idx:04d}"
    return selected


def _select_semantic_system_units(
    input_data: dict,
    units_by_id: dict[str, dict],
    legacy_id_index: dict[str, list[str]],
) -> tuple[list[dict], set[str], set[str]]:
    semantic_system = input_data.get("semantic_system") or {}
    by_block: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    blocked_source_ids: set[str] = set()
    blocked_block_ids: set[str] = set()
    for level, key in (("semantic_phrase", "semantic_phrases"), ("semantic_group", "semantic_groups")):
        entries = semantic_system.get(key) or []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            text = normalize_spaces(entry.get("text") or entry.get("texte"))
            source_ids = _entry_source_ids(entry, units_by_id, legacy_id_index)
            source_units = [units_by_id[sid] for sid in source_ids if sid in units_by_id]
            block_id = (
                ((entry.get("structural_context") or {}).get("block_unit_id"))
                or next((ancestor_id(unit["unit_id"], units_by_id, level="block") for unit in source_units), None)
            )
            bbox = entry.get("bbox") or bbox_union([(unit.get("geometry") or {}).get("bbox") for unit in source_units])
            raw_source_ids = _raw_entry_source_ids(entry)
            if raw_source_ids and not source_units and not block_id:
                blocked_source_ids.update(raw_source_ids)
                continue
            if level in {"semantic_phrase", "semantic_group"} and not source_ids and not block_id and not bbox:
                continue
            if _entry_not_translatable(entry):
                blocked_source_ids.update(source_ids)
                if block_id and not source_ids:
                    blocked_block_ids.add(block_id)
                continue
            if not text or _is_excluded_text(text, role=entry.get("role")):
                blocked_source_ids.update(source_ids)
                continue
            if source_units and any(_is_excluded_unit(unit) for unit in source_units):
                blocked_source_ids.update(source_ids)
                continue
            sample = source_units[0] if source_units else {}
            item = _make_item(
                unit_id=entry.get("unit_id") or f"{level}:{idx}",
                level=level,
                source_text=text,
                unit=sample,
                source_unit_ids=source_ids,
                block_id=block_id,
                bbox=bbox,
                semantic_entry=entry,
            )
            by_block[block_id or "__page__"][level].append(item)

    output = []
    for levels in by_block.values():
        # Respect the contract literally: semantic_phrase wins over semantic_group
        # for the same block, otherwise the same source can be translated twice.
        chosen = levels.get("semantic_phrase") or levels.get("semantic_group") or []
        output.extend(chosen)
    return output, blocked_source_ids, blocked_block_ids



def _prefix_ancestor(ancestor: str | None, descendant: str | None) -> bool:
    """Conservative PagePrint id ancestry fallback.

    PagePrint ids usually follow pXXX_block_NNN_line_MMM_phrase_KKK... .  The
    explicit children graph is the source of truth when present, but the prefix
    convention is often the only available signal in compact views.
    """
    return bool(ancestor) and bool(descendant) and descendant != ancestor and str(descendant).startswith(str(ancestor) + "_")


def _related_to_any_source(unit_id: str | None, source_ids: set[str], units_by_id: dict[str, dict]) -> bool:
    if not unit_id:
        return False
    if unit_id in source_ids:
        return True
    # Avoid parent/child double selection: if a semantic phrase was selected,
    # the line/block ancestor must not be selected as a whole; if a semantic
    # group selected a parent, its children must not be reselected either.
    for sid in source_ids:
        if _prefix_ancestor(unit_id, sid) or _prefix_ancestor(sid, unit_id):
            return True
        anc = ancestor_id(unit_id, units_by_id, level="block")
        sid_anc = ancestor_id(sid, units_by_id, level="block") if sid in units_by_id else None
        if anc and sid_anc and anc == sid_anc and (unit_id == sid or _prefix_ancestor(unit_id, sid) or _prefix_ancestor(sid, unit_id)):
            return True
    return False


def strip_running_header_page_number(text: str, *, role: str | None = None, bbox=None) -> tuple[str, dict]:
    """Remove page numbers accidentally fused with running headers.

    The page number is preserved by PAGEPRINT as a dedicated immutable overlay.
    If the translation unit also contains it (e.g. "70 CHAPTER 2 ..."), the
    renderer draws it twice and the translator may translate/relocate it.  This
    function strips only plausible running-header cases, not ordinary numbered
    section titles such as "2.5.4 Mean square error" or table cells like
    "4 bytes ...".
    """
    s = normalize_spaces(text)
    if not s:
        return s, {}
    m = re.match(r"^(\d{1,4})\s+(.+)$", s)
    if not m:
        return s, {}
    number, rest = m.group(1), normalize_spaces(m.group(2))
    if not rest or re.match(r"^\d+(?:[.)]|\.\d)", rest):
        return s, {}
    role_l = str(role or "").lower()
    y0 = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None
    upper_header = y0 is not None and y0 <= 55.0
    chapterish = bool(re.match(r"^(?:CHAPTER|Chapter|PART|Part)\b", rest))
    titleish = bool(re.match(r"^[A-Z][A-Za-z].{4,}$", rest)) and not re.match(r"^(?:bytes?|decimal|floating)\b", rest, re.I)
    if upper_header and (chapterish or role_l in {"running_header", "section_heading", "chapter_heading"} and titleish):
        return rest, {"stripped_page_number": number, "strip_reason": "running_header_fused_page_number"}
    return s, {}

def _select_pageprint_units_by_block(
    input_data: dict,
    units_by_id: dict[str, dict],
    selected_blocks: set[str | None],
    semantic_source_ids: set[str],
    blocked_source_ids: set[str],
    blocked_block_ids: set[str],
) -> list[dict]:
    candidates = _candidate_pageprint_units(input_data, units_by_id)
    by_block: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for unit in candidates:
        if unit.get("unit_id") in semantic_source_ids:
            continue
        if unit.get("unit_id") in blocked_source_ids:
            continue
        if _is_excluded_unit(unit):
            continue
        level = unit.get("level")
        if level not in {"phrase", "line", "block"}:
            continue
        block_id = unit["unit_id"] if level == "block" else ancestor_id(unit["unit_id"], units_by_id, level="block")
        block_key = block_id or "__page__"
        # Do NOT skip a whole block merely because PAGEPRINT emitted one or two
        # semantic phrases for it.  Semantic coverage can be incomplete; skipping
        # the block here is the main reason large parts of pages remain in the
        # original language.  Skip only explicit non-translatable blocks and
        # individual units already covered by semantic translation.
        if block_key in blocked_block_ids:
            continue
        if _related_to_any_source(unit["unit_id"], semantic_source_ids, units_by_id):
            # For line ancestors, keep the block in play: unselected phrase
            # children may still be translated below.
            if level != "line":
                continue
        by_block[block_key][level].append(unit)

    output = []
    for block_key, levels in by_block.items():
        phrases = sorted(levels.get("phrase") or [], key=reading_order)
        lines = sorted(levels.get("line") or [], key=reading_order)
        blocks = sorted(levels.get("block") or [], key=reading_order)
        if lines:
            phrase_parent_ids = {unit.get("parent_id") for unit in phrases}
            for line in lines:
                line_phrases = [
                    phrase for phrase in phrases
                    if phrase.get("parent_id") == line.get("unit_id")
                    and not _related_to_any_source(phrase.get("unit_id"), semantic_source_ids, units_by_id)
                ]
                # If a semantic phrase already covers part of this line, do not
                # fall back to the whole line: that would duplicate the semantic
                # phrase.  Translate remaining phrase children when they exist;
                # otherwise leave the missing fragment to the lifecycle audit.
                line_has_semantic_descendant = any(_prefix_ancestor(line.get("unit_id"), sid) for sid in semantic_source_ids)
                selected_units = line_phrases or ([] if line_has_semantic_descendant else [line])
                for unit in selected_units:
                    if _related_to_any_source(unit.get("unit_id"), semantic_source_ids, units_by_id):
                        continue
                    output.append(_make_item(
                        unit_id=unit["unit_id"],
                        level=unit.get("level"),
                        source_text=unit_text(unit),
                        unit=unit,
                        source_unit_ids=[unit["unit_id"]],
                        block_id=block_key,
                        bbox=(unit.get("geometry") or {}).get("bbox"),
                    ))
            orphan_phrases = [
                phrase for phrase in phrases
                if phrase.get("parent_id") not in phrase_parent_ids
                and not ancestor_id(phrase["unit_id"], units_by_id, level="line")
                and not _related_to_any_source(phrase.get("unit_id"), semantic_source_ids, units_by_id)
            ]
            for unit in orphan_phrases:
                output.append(_make_item(
                    unit_id=unit["unit_id"],
                    level=unit.get("level"),
                    source_text=unit_text(unit),
                    unit=unit,
                    source_unit_ids=[unit["unit_id"]],
                    block_id=block_key,
                    bbox=(unit.get("geometry") or {}).get("bbox"),
                ))
            continue
        selected_units = phrases or blocks
        for unit in selected_units:
            if _related_to_any_source(unit.get("unit_id"), semantic_source_ids, units_by_id):
                continue
            output.append(_make_item(
                unit_id=unit["unit_id"],
                level=unit.get("level"),
                source_text=unit_text(unit),
                unit=unit,
                source_unit_ids=[unit["unit_id"]],
                block_id=block_key,
                bbox=(unit.get("geometry") or {}).get("bbox"),
            ))
    return output


def _candidate_pageprint_units(input_data: dict, units_by_id: dict[str, dict]) -> list[dict]:
    view_ids = [
        item.get("unit_id")
        for item in ((input_data.get("views") or {}).get("translation_units") or [])
        if isinstance(item, dict)
    ]
    candidates = [units_by_id[unit_id] for unit_id in view_ids if unit_id in units_by_id]
    return candidates or list(units_by_id.values())


def _entry_not_translatable(entry: dict) -> bool:
    if entry.get("translatable") is False:
        return True
    strategy = str(entry.get("translation_strategy") or "").lower()
    render_policy = str(entry.get("render_policy") or "").lower()
    if strategy in EXCLUDED_STRATEGIES:
        return True
    if render_policy == "background_only":
        return True
    tags = {
        str(entry.get("role") or "").lower(),
        str(entry.get("object_type") or "").lower(),
        str(entry.get("object_class") or "").lower(),
        str(entry.get("semantic_kind") or entry.get("kind") or "").lower(),
    }
    return bool(tags & EXCLUDED_CLASSES)


def _entry_source_ids(entry: dict, units_by_id: dict[str, dict], legacy_id_index: dict[str, list[str]]) -> list[str]:
    resolved = []
    for source_id in _raw_entry_source_ids(entry):
        source_key = str(source_id)
        if source_key in units_by_id:
            resolved.append(source_key)
        elif source_key in legacy_id_index:
            resolved.extend(legacy_id_index[source_key])
        else:
            resolved.append(source_key)
    return list(dict.fromkeys(resolved))


def _raw_entry_source_ids(entry: dict) -> list[str]:
    return [
        sid for sid in (
            entry.get("source_unit_ids")
            or entry.get("unit_ids")
            or entry.get("phrase_unit_ids")
            or []
        )
        if sid
    ]


def _legacy_id_index(units_by_id: dict[str, dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for unit in units_by_id.values():
        legacy_id = unit.get("legacy_id")
        if legacy_id is not None:
            index.setdefault(str(legacy_id), []).append(unit["unit_id"])
    return index


def _is_excluded_unit(unit: dict) -> bool:
    level = unit.get("level")
    if level in AUXILIARY_TEXT_LEVELS or level not in PRIMARY_TEXT_LEVELS:
        return True
    text = unit_text(unit)
    if not text or _is_excluded_text(text, role=(unit.get("understanding") or {}).get("role")):
        return True
    policy = unit.get("policy") or {}
    understanding = unit.get("understanding") or {}
    constraints = unit.get("constraints") or {}
    if policy.get("translatable") is not True:
        return True
    if constraints.get("skip_translation"):
        return True
    if policy.get("render_policy") == "background_only":
        return True
    if str(policy.get("translation_strategy") or "").lower() in EXCLUDED_STRATEGIES:
        return True
    if _covered_by_protected_visual(unit):
        return True
    tags = {
        str(policy.get("unit_type") or "").lower(),
        str(understanding.get("role") or "").lower(),
        str(understanding.get("object_type") or "").lower(),
        str(understanding.get("object_class") or "").lower(),
        str(understanding.get("semantic_kind") or "").lower(),
    }
    return bool(tags & EXCLUDED_CLASSES)


def _covered_by_protected_visual(unit: dict) -> bool:
    if (unit.get("relations") or {}).get("covered_by_protected_region_id"):
        return True
    for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
        if membership.get("region_type") == "protected_visual_region":
            return True
    return False


def _is_excluded_text(text: str, *, role: str | None = None) -> bool:
    s = normalize_spaces(text)
    if not s:
        return True
    if URL_RE.fullmatch(s) or EMAIL_RE.fullmatch(s) or DOI_RE.fullmatch(s) or PATH_RE.fullmatch(s):
        return True
    if REFERENCE_RE.fullmatch(s):
        return True
    if _is_probable_acronym(s, role=role):
        return True
    if FORMULA_RE.fullmatch(s) and len(s.split()) <= 6:
        return True
    if re.fullmatch(r"[\d\s.,:;/%+-]+", s):
        return True
    return False


def _is_probable_acronym(text: str, *, role: str | None = None) -> bool:
    s = normalize_spaces(text)
    if not re.fullmatch(r"[A-Z0-9&./+-]{2,12}", s):
        return False
    role_l = str(role or "").lower()
    if role_l in {"title", "section_heading", "heading"} and len(s) > 5:
        return False
    if len(s) > 8:
        return False
    letters = re.sub(r"[^A-Z]", "", s)
    if len(letters) > 6 and re.search(r"[AEIOUY]{2,}", letters):
        return False
    return True


def _make_item(
    *,
    unit_id: str,
    level: str,
    source_text: str,
    unit: dict,
    source_unit_ids: list[str],
    block_id: str | None,
    bbox: Any,
    semantic_entry: dict | None = None,
) -> dict:
    understanding = unit.get("understanding") or {}
    policy = unit.get("policy") or {}
    entry = semantic_entry or {}
    role = entry.get("role") or understanding.get("role")
    source_text, preprocess = strip_running_header_page_number(source_text, role=role, bbox=bbox)
    return {
        "translation_unit_id": "",
        "unit_id": unit_id,
        "level": level,
        "parent_id": unit.get("parent_id") or block_id,
        "source_unit_ids": source_unit_ids,
        "block_id": block_id,
        "source_text": normalize_spaces(source_text),
        "preprocess": preprocess,
        "bbox": bbox,
        "reading_order_index": (unit.get("geometry") or {}).get("reading_order_index") or entry.get("reading_order_index") or 0,
        "role": role,
        "object_type": entry.get("object_type") or understanding.get("object_type") or policy.get("unit_type"),
        "object_class": entry.get("object_class") or understanding.get("object_class"),
        "semantic_kind": entry.get("semantic_kind") or entry.get("kind") or understanding.get("semantic_kind"),
        "strategy": entry.get("translation_strategy") or policy.get("translation_strategy") or "layout_constrained",
        "render_policy": entry.get("render_policy") or policy.get("render_policy"),
        "coverage_required": entry.get("coverage_required") or policy.get("coverage_required"),
        "protected": list(entry.get("protected") or policy.get("translation_protection") or []),
        "translatable": bool(entry.get("translatable", policy.get("translatable", True))),
    }
