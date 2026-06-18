"""Compile translated_input_data into a PageRenderPlan (no PDF rendering).

Merges the four views (reconstruction_units + reconstruction/preservation/
exclusion plans) plus units geometry and visual layers, computing consumed /
excluded source ids and the protected-region index, while forbidding
parent/child double rendering (directive §5, §6, §17.1).
"""

from __future__ import annotations

from .background_resolver import resolve_background
from .input_adapter import PageReconstructInputAdapter
from .layout_box_resolver import resolve_layout
from .patch_planner import plan_patches
from .protected_region_index import build_protected_region_index
from .schema import PageRenderPlan, PreservedUnit, TranslatedTextUnit, ProtectedRegion, PatchZone
from .style_resolver import resolve_style
from source_ownership import (
    build_source_ownership,
    is_non_translatable_owner,
    source_ids_have_non_translatable_owner,
    all_source_ids_non_translatable,
    audit_source_ownership,
)

RENDERER_BY_ROLE = {
    "body_paragraph": "paragraph", "list_item": "paragraph", "author_bio": "paragraph",
    "index_subentry": "paragraph", "formula_explanation": "paragraph",
    "title": "heading", "subtitle": "heading", "section_heading": "heading",
    "subsection_heading": "heading", "chapter_heading": "heading",
    "figure_caption": "caption", "figure_caption_text": "caption",
    "table_caption": "caption", "table_caption_text": "caption",
    "table_header_cell": "table", "table_body_cell": "table", "table_numeric_cell": "table",
    "toc_entry_title": "anchored_label", "toc_entry": "anchored_label",
    "index_entry": "index", "index_head_term": "index", "index_subentry": "index",
    "index_page_reference": "index", "bibliography_entry": "bibliography",
    "diagram_label": "anchored_label", "diagram_text_label": "anchored_label",
    "axis_label": "anchored_label", "legend_label": "anchored_label",
    "code_line": "code", "code_block": "code",
    "formula_expression": "formula",
}

DEFAULT_RENDER_POLICY = {
    "fail_on_missing_reconstruction_units": True,
    "fail_on_unresolved_style": False,
    "allow_legacy_blocks_fallback": False,
}
DEFAULT_QUALITY_EXPECTATIONS = {
    "require_text_coverage": True,
    "require_no_protected_overlap": True,
    "require_no_source_text_leak": True,
}


def choose_renderer(role, object_type) -> str:
    role = str(role or "")
    if role in RENDERER_BY_ROLE:
        return RENDERER_BY_ROLE[role]
    ot = str(object_type or "").lower()
    if ot in {"code_block", "inline_code", "code"}:
        return "code"
    if ot in {"formula_block", "equation", "formula_expression"}:
        return "formula"
    if ot in {"table_cell"}:
        return "table"
    # Unknown role must NOT silently become paragraph (directive §8).
    return "anchored_label_review"


def _union(boxes) -> list | None:
    boxes = [b for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _coverage_bbox(source_unit_ids, unit_index, fallback) -> list | None:
    boxes = []
    for sid in source_unit_ids or []:
        u = unit_index.get(sid)
        if u:
            b = (u.get("geometry") or {}).get("bbox") or u.get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append(b)
    if fallback:
        boxes.append(fallback)
    return _union(boxes) or fallback


def _ops_overlap_cost(ops) -> float:
    """Somme des chevauchements texte/texte des TextOps (proxy de collision pour
    le garde net-improvement des enhancers externes)."""
    boxes = []
    for op in ops or []:
        d = op.to_dict() if hasattr(op, "to_dict") else op
        if d.get("op_type") != "text":
            continue
        lines = d.get("lines") or []
        xs = [l for l in lines if "x1" in l]
        if xs:
            boxes.append([min(l["x"] for l in xs), min(l["y_top"] for l in xs),
                          max(l["x1"] for l in xs), max(l["y_bottom"] for l in xs)])
    cost = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if ix > 0:
                cost += ix
    return cost


def _descendants(uid: str, children_map: dict) -> set:
    out, stack = set(), list(children_map.get(uid) or [])
    while stack:
        c = stack.pop()
        if c in out:
            continue
        out.add(c)
        stack.extend(children_map.get(c) or [])
    return out


# --- Zone-as-source-of-truth (formula/code) -------------------------------
SPECIAL_ZONE_CONF_MIN = 0.7      # trust only confident detector zones
SPECIAL_ZONE_MERGE_GAP = 8.0     # pt: merge fragments of one equation/listing
SPECIAL_ZONE_CONTAIN = 0.6       # a unit ≥60% inside a zone is part of it


def _boxes_touch(a, b, gap: float) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0]
                or a[3] + gap < b[1] or b[3] + gap < a[1])


def _contained_ratio(inner, outer) -> float:
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _build_special_zones(regions) -> list[dict]:
    """Merge formula/code/protected regions into tight preserved blocks.

    These zones are hard objects, not advisory candidates.  They must survive to
    the final contract as both protected regions and preserved underlay blocks.
    """
    raw = []
    for r in regions or []:
        rt = str(r.get("region_type") or r.get("object_type") or "").lower()
        if rt not in {"formula_region", "code_region", "protected_visual_region"}:
            continue
        kind = "formula" if "formula" in rt or "equation" in rt or "math" in rt else "code" if "code" in rt else "protected_visual" if "protected_visual" in rt else None
        if not kind:
            continue
        # Keep confirmed special regions even at moderate confidence.  Formula
        # detection is allowed to be conservative upstream; it must not be
        # re-neutralized here.
        conf = float(r.get("confidence") or 0.0)
        if conf and conf < max(0.20, SPECIAL_ZONE_CONF_MIN * 0.50):
            continue
        b = r.get("bbox") or r.get("visual_bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        members = r.get("members") or {}
        source_ids = []
        for key in ("block_ids", "line_ids", "phrase_ids", "span_ids", "word_ids", "char_ids"):
            source_ids.extend([sid for sid in members.get(key) or [] if sid])
        raw.append({
            "kind": kind,
            "bbox": [float(x) for x in b],
            "source_unit_ids": list(dict.fromkeys(source_ids)),
            "region_id": r.get("region_id"),
            "confidence": conf or 0.5,
            "source": r.get("source") or r.get("detection_source") or "pageprint_region",
        })
    merged: list[dict] = []
    for z in raw:
        hit = None
        for m in merged:
            if m["kind"] == z["kind"] and _boxes_touch(m["bbox"], z["bbox"], SPECIAL_ZONE_MERGE_GAP):
                hit = m
                break
        if hit:
            hit["bbox"] = _union([hit["bbox"], z["bbox"]])
            hit["source_unit_ids"] = list(dict.fromkeys((hit.get("source_unit_ids") or []) + (z.get("source_unit_ids") or [])))
            hit["confidence"] = max(float(hit.get("confidence") or 0), float(z.get("confidence") or 0))
            hit.setdefault("region_ids", []).append(z.get("region_id"))
        else:
            merged.append({**z, "region_ids": [z.get("region_id")] if z.get("region_id") else []})
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if (merged[i]["kind"] == merged[j]["kind"]
                        and _boxes_touch(merged[i]["bbox"], merged[j]["bbox"], SPECIAL_ZONE_MERGE_GAP)):
                    merged[i]["bbox"] = _union([merged[i]["bbox"], merged[j]["bbox"]])
                    merged[i]["source_unit_ids"] = list(dict.fromkeys((merged[i].get("source_unit_ids") or []) + (merged[j].get("source_unit_ids") or [])))
                    merged[i]["confidence"] = max(float(merged[i].get("confidence") or 0), float(merged[j].get("confidence") or 0))
                    merged[i].setdefault("region_ids", []).extend(merged[j].get("region_ids") or [])
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    return merged


def _is_duplicate_box(bbox, rendered_boxes, min_mutual: float = 0.8) -> bool:
    """A new translated box that mutually covers an already-rendered one (≥80%
    both ways) is the same region emitted twice — a stacked duplicate."""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return False
    for r in rendered_boxes:
        if _contained_ratio(bbox, r) >= min_mutual and _contained_ratio(r, bbox) >= min_mutual:
            return True
    return False


def _in_special_zone(bbox, zones, min_ratio: float = SPECIAL_ZONE_CONTAIN) -> dict | None:
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    for z in zones:
        if _contained_ratio([float(x) for x in bbox], z["bbox"]) >= min_ratio:
            return z
    return None



def _bbox_width(b) -> float:
    return float(b[2]) - float(b[0]) if isinstance(b, (list, tuple)) and len(b) == 4 else 0.0


def _bbox_height(b) -> float:
    return float(b[3]) - float(b[1]) if isinstance(b, (list, tuple)) and len(b) == 4 else 0.0


def _normalise_role(role: str | None, bbox, item: dict, page_height: float | None = None) -> str | None:
    """Correct obvious role overreach before renderer/style decisions.

    PAGEPRINT can mark narrow text embedded in a figure as section_heading.  That
    makes the typography engine pick a heading size and produces huge labels.
    A true section heading is usually wide or near the top of a text block; a
    narrow multi-line box in the middle of the page is an anchored diagram label.
    """
    r = str(role or "")
    if r in {"section_heading", "chapter_heading", "subsection_heading"}:
        w, h = _bbox_width(bbox), _bbox_height(bbox)
        y0 = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else 0.0
        mid_page = page_height is None or (70.0 < y0 < max(70.0, page_height - 70.0))
        text = str(item.get("source_text") or item.get("text") or "")
        labelish = bool(w and h and w < 120.0 and h > 14.0 and mid_page)
        if labelish and not text.lstrip().startswith(("CHAPTER", "Chapter")):
            return "diagram_label"
    return role


def _prefix_ancestor(ancestor: str | None, descendant: str | None) -> bool:
    return bool(ancestor) and bool(descendant) and descendant != ancestor and str(descendant).startswith(str(ancestor) + "_")


def _source_related_to_consumed(source_ids: list[str], consumed: set[str]) -> bool:
    for sid in source_ids or []:
        if sid in consumed:
            return True
        if any(_prefix_ancestor(c, sid) or _prefix_ancestor(sid, c) for c in consumed):
            return True
    return False




def _unit_bbox(unit: dict) -> list | None:
    b = (unit.get("geometry") or {}).get("bbox") or unit.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(x) for x in b]
    return None


def _unit_text(unit: dict) -> str:
    return str(unit.get("text") or (unit.get("content") or {}).get("text") or "").strip()


def _unit_role(unit: dict) -> str:
    return str(unit.get("role") or (unit.get("understanding") or {}).get("role") or "")


def _unit_level(unit: dict) -> str:
    return str(unit.get("level") or unit.get("unit_level") or unit.get("type") or unit.get("unit_type") or "")


def _source_unit_is_rendered_or_covered(uid: str, rendered: set[str], consumed: set[str], excluded: set[str]) -> bool:
    if uid in rendered or uid in consumed or uid in excluded:
        return True
    for owner in (rendered | consumed | excluded):
        if _prefix_ancestor(owner, uid) or _prefix_ancestor(uid, owner):
            return True
    return False


def _is_text_survival_candidate(unit: dict) -> bool:
    """Line-level safety net: every visible source text line must survive.

    This is intentionally conservative for WYSIWYG: it does not decide whether
    the line should have been translated; it only guarantees that a missed line
    is rendered with its source text rather than disappearing.  Translation
    completeness can be improved upstream, but text disappearance is forbidden.
    """
    text = _unit_text(unit)
    if not text:
        return False
    level = _unit_level(unit)
    if level and level not in {"line", "text_line"}:
        return False
    role = _unit_role(unit)
    if role in {"page_number", "page_reference", "watermark", "publisher_mark"}:
        return False
    if role in {"word", "char", "glyph"}:
        return False
    b = _unit_bbox(unit)
    if not b:
        return False
    # Ignore microscopic artifacts; real lines have some physical size.
    if _bbox_width(b) < 2.0 or _bbox_height(b) < 2.0:
        return False
    return True

def _is_valid_page_number_preservation(p: dict, page_width: float | None, page_height: float | None) -> bool:
    text = str(p.get("text") or "").strip()
    if not text or not text.isdigit() or len(text) > 4:
        return False
    b = p.get("bbox")
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return False
    w = float(page_width or 0.0)
    h = float(page_height or 0.0)
    x0, y0, x1, y1 = [float(x) for x in b]
    if not w or not h:
        return True
    near_vertical_edge = y0 <= 60.0 or y1 >= h - 60.0
    near_horizontal_edge = x0 <= w * 0.24 or x1 >= w * 0.76
    return near_vertical_edge and near_horizontal_edge


def _must_preserve_even_if_consumed(p: dict, page_width: float | None, page_height: float | None) -> bool:
    reason = str(p.get("reason") or "").lower()
    role = str(p.get("role") or "").lower()
    tags = {reason, role, str(p.get("object_type") or "").lower(), str(p.get("semantic_kind") or "").lower()}
    if "page_number" in tags or reason == "page_number":
        return _is_valid_page_number_preservation(p, page_width, page_height)
    immutable = {
        "formula_region", "formula", "formula_expression", "equation", "math_expression", "code_region", "code", "code_line", "code_block",
        "diagram_label", "axis_label", "legend_label", "publisher_mark", "watermark",
        "protected_visual_region", "caption_number", "caption_label",
        "toc_page_reference", "toc_section_number",
    }
    return bool(tags & immutable)


def _source_ids_of_item(item) -> set[str]:
    if isinstance(item, dict):
        vals = item.get("source_unit_ids") or []
    else:
        vals = getattr(item, "source_unit_ids", None) or []
    return {str(s) for s in vals if s}


def _set_source_ids_on_item(item, source_ids: set[str]) -> None:
    merged = sorted(_source_ids_of_item(item) | {str(s) for s in (source_ids or set()) if s})
    if isinstance(item, dict):
        item["source_unit_ids"] = merged
    else:
        try:
            item.source_unit_ids = merged
        except Exception:
            pass


def _dedupe_preserved_units(items: list) -> list:
    seen: dict[tuple, object] = {}
    out = []
    for item in items:
        bbox = getattr(item, "bbox", None)
        text = getattr(item, "text", None)
        reason = getattr(item, "reason", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            rb = tuple(round(float(x), 1) for x in bbox)
        else:
            rb = None
        key = (str(reason or ""), str(text or ""), rb)
        if key in seen:
            # Dedupe must never drop ownership descendants. Merge source ids into
            # the retained object so audit/render contract still sees them.
            _set_source_ids_on_item(seen[key], _source_ids_of_item(item))
            continue
        seen[key] = item
        out.append(item)
    return out


# --- Ownership/Lifecycle v2.1: render-contract propagation lock -----------
def _valid_box(b) -> bool:
    return isinstance(b, (list, tuple)) and len(b) == 4


def _box_area(b) -> float:
    if not _valid_box(b):
        return 0.0
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _box_intersection(a, b) -> float:
    if not (_valid_box(a) and _valid_box(b)):
        return 0.0
    x0, y0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x1, y1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_overlap_ratio(inner, outer) -> float:
    return _box_intersection(inner, outer) / max(1e-6, _box_area(inner))


def _box_union(boxes) -> list | None:
    boxes = [[float(x) for x in b] for b in boxes or [] if _valid_box(b)]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _box_key_1(b) -> tuple | None:
    if not _valid_box(b):
        return None
    return tuple(round(float(x), 1) for x in b)


def _bbox_of_item(item):
    return item.get("bbox") if isinstance(item, dict) else getattr(item, "bbox", None)


def _covered_by_box_or_source_ids(bbox, source_ids: set[str], items, *, threshold: float = 0.72) -> bool:
    if source_ids:
        for item in items or []:
            if source_ids & _source_ids_of_item(item):
                return True
    if not _valid_box(bbox):
        return False
    for item in items or []:
        b = _bbox_of_item(item)
        if _valid_box(b) and (_box_overlap_ratio(bbox, b) >= threshold or _box_overlap_ratio(b, bbox) >= 0.92):
            return True
    return False


def _merge_source_ids_into_covering_items(bbox, source_ids: set[str], items, *, threshold: float = 0.72) -> int:
    """Attach all descendant ownership ids to every covering render item.

    v2.1 could decide that a region was already represented because one ancestor
    or sibling was present.  The audit, however, checks each descendant id.  A
    covering ProtectedRegion/PreservedUnit must therefore explicitly carry all
    descendants of the special region.
    """
    if not source_ids:
        return 0
    changed = 0
    for item in items or []:
        hit = bool(source_ids & _source_ids_of_item(item))
        b = _bbox_of_item(item)
        if not hit and _valid_box(b) and _valid_box(bbox):
            hit = (_box_overlap_ratio(bbox, b) >= threshold or _box_overlap_ratio(b, bbox) >= 0.92)
        if hit:
            before = _source_ids_of_item(item)
            _set_source_ids_on_item(item, source_ids)
            if _source_ids_of_item(item) != before:
                changed += 1
    return changed


def _subtract_box(rect, hole) -> list[list[float]]:
    """Return rect minus hole as up to four non-overlapping rectangles."""
    if not (_valid_box(rect) and _valid_box(hole)) or _box_intersection(rect, hole) <= 0:
        return [[float(x) for x in rect]] if _valid_box(rect) else []
    rx0, ry0, rx1, ry1 = [float(x) for x in rect]
    hx0, hy0, hx1, hy1 = [float(x) for x in hole]
    hx0, hy0, hx1, hy1 = max(rx0, hx0), max(ry0, hy0), min(rx1, hx1), min(ry1, hy1)
    out = []
    if hy0 > ry0:
        out.append([rx0, ry0, rx1, hy0])
    if hy1 < ry1:
        out.append([rx0, hy1, rx1, ry1])
    my0, my1 = max(ry0, hy0), min(ry1, hy1)
    if hx0 > rx0:
        out.append([rx0, my0, hx0, my1])
    if hx1 < rx1:
        out.append([hx1, my0, rx1, my1])
    return [r for r in out if (r[2] - r[0]) > 0.50 and (r[3] - r[1]) > 0.50]


def _patch_to_dict(patch) -> dict:
    return patch.to_dict() if hasattr(patch, "to_dict") else dict(patch or {})


def _patch_from_dict(d: dict, bbox: list, suffix: str | int = "") -> PatchZone:
    return PatchZone(
        op_type=str(d.get("op_type") or "patch_text_zone"),
        unit_id=str(d.get("unit_id") or "patch") + (f"_safe{suffix}" if suffix != "" else ""),
        bbox=[float(x) for x in bbox],
        method=str(d.get("method") or "sampled_color_patch"),
        background_color=d.get("background_color"),
        protected_overlap_ratio=0.0,
        must_not_overlap=list(d.get("must_not_overlap") or []),
        padding=list(d.get("padding") or [1.8, 0.9, 1.8, 0.9]),
    )


def _preserved_visual_groups_from_ownership(source_ownership: dict) -> list[dict]:
    """Compact preserved_visual entries into renderable groups.

    The ownership table can contain a region plus its line/span/word/char
    descendants. Render/preservation must not materialise thousands of tiny
    duplicate objects.  Region id is therefore the primary grouping key; source
    id is a fallback for orphan entries.
    """
    groups: dict[str, dict] = {}
    for sid, entry in (source_ownership or {}).items():
        if not isinstance(entry, dict) or entry.get("state") != "preserved_visual":
            continue
        bbox = entry.get("bbox")
        if not _valid_box(bbox):
            continue
        region_id = str(entry.get("region_id") or "").strip()
        preservation_id = str(entry.get("preservation_id") or "").strip()
        key = (
            f"region:{region_id}" if region_id
            else f"preservation:{preservation_id}" if preservation_id
            else f"unit:{sid}"
        )
        g = groups.setdefault(key, {
            "key": key,
            "region_id": region_id or None,
            "reason": str(entry.get("reason") or "protected_visual"),
            "source_unit_ids": [],
            "boxes": [],
        })
        g["source_unit_ids"].append(str(sid))
        g["boxes"].append([float(x) for x in bbox])
    out = []
    for idx, g in enumerate(groups.values(), start=1):
        bbox = _box_union(g.get("boxes") or [])
        if not bbox:
            continue
        out.append({
            "id": f"ownlife_v21_{idx:04d}",
            "region_id": g.get("region_id"),
            "reason": g.get("reason") or "protected_visual",
            "bbox": bbox,
            "source_unit_ids": list(dict.fromkeys(g.get("source_unit_ids") or [])),
        })
    return out


def _filter_patches_around_preserved_visual(patches: list, preserved_boxes: list[list[float]], findings: list[dict]) -> list:
    if not patches or not preserved_boxes:
        return patches or []
    safe: list[PatchZone] = []
    removed = 0
    split = 0
    for patch in patches or []:
        d = _patch_to_dict(patch)
        bbox = d.get("bbox")
        if not _valid_box(bbox):
            continue
        rects = [[float(x) for x in bbox]]
        touched = False
        for hole in preserved_boxes:
            nxt = []
            for r in rects:
                if _box_intersection(r, hole) > 0:
                    touched = True
                    nxt.extend(_subtract_box(r, hole))
                else:
                    nxt.append(r)
            rects = nxt
            if not rects:
                break
        if not touched:
            safe.append(patch if hasattr(patch, "to_dict") else _patch_from_dict(d, bbox))
            continue
        if not rects:
            removed += 1
            continue
        split += max(0, len(rects) - 1)
        for i, r in enumerate(rects, start=1):
            safe.append(_patch_from_dict(d, r, i))
    if removed or split:
        findings.append({
            "type": "ownership_lifecycle_v2_1_patch_clipped_around_preserved_visual",
            "severity": "info",
            "removed_patch_count": removed,
            "additional_patch_piece_count": split,
        })
    return safe


def _enforce_preserved_visual_render_contract(plan: PageRenderPlan, normalized: dict, source_ownership: dict, findings: list[dict]) -> None:
    """Force v1 preserved_visual ownership into the render contract.

    This is the v2.1 lock: a source unit owned by a preserved visual region must
    have render-level representation (protected region + preserved layer, which
    subsequently produces PreservationOp) and no patch is allowed to cover it.
    """
    groups = _preserved_visual_groups_from_ownership(source_ownership)
    if not groups:
        return

    added_prot = 0
    added_pres = 0
    for g in groups:
        bbox = g["bbox"]
        source_ids = set(g.get("source_unit_ids") or [])
        # ProtectedRegion: hard obstacle for layout, patch planning and audit.
        if not _covered_by_box_or_source_ids(bbox, source_ids, plan.protected_regions, threshold=0.72):
            idx = len(plan.protected_regions) + 1
            plan.protected_regions.append(ProtectedRegion(
                id=f"ownlife_v21_prot_{idx:04d}",
                source="source_ownership_v2_1",
                reason=g.get("reason") or "preserved_visual",
                bbox=[float(x) for x in bbox],
                hard=True,
                z_policy="preserve_original",
                source_unit_ids=list(g.get("source_unit_ids") or []),
            ))
            added_prot += 1
        else:
            _merge_source_ids_into_covering_items(bbox, source_ids, plan.protected_regions, threshold=0.72)
        # PreservedUnit: source pixels/visual object in the layer contract.
        preserved_items = list(plan.preserved_underlays or []) + list(plan.preserved_overlays or [])
        if not _covered_by_box_or_source_ids(bbox, source_ids, preserved_items, threshold=0.72):
            idx = len(plan.preserved_underlays) + len(plan.preserved_overlays) + 1
            plan.preserved_underlays.append(PreservedUnit(
                id=f"ownlife_v21_pres_{idx:04d}",
                source="source_ownership_v2_1",
                reason=g.get("reason") or "preserved_visual",
                bbox=[float(x) for x in bbox],
                text=None,
                preservation_mode="preserve_as_visual_overlay",
                source_unit_ids=list(g.get("source_unit_ids") or []),
                z_policy="preserve_original",
            ))
            added_pres += 1
        else:
            _merge_source_ids_into_covering_items(bbox, source_ids, preserved_items, threshold=0.72)

    # Final exhaustive pass: every preserved_visual descendant id must be
    # explicitly attached to at least one covering ProtectedRegion and one
    # covering PreservedUnit.  This is the v2.2 closure over descendants.
    merged_prot_sids = 0
    merged_pres_sids = 0
    for g in groups:
        bbox = g.get("bbox")
        source_ids = set(g.get("source_unit_ids") or [])
        merged_prot_sids += _merge_source_ids_into_covering_items(bbox, source_ids, plan.protected_regions, threshold=0.72)
        merged_pres_sids += _merge_source_ids_into_covering_items(bbox, source_ids, list(plan.preserved_underlays or []) + list(plan.preserved_overlays or []), threshold=0.72)

    # Re-dedupe after forced insertions, then rebuild patch layer around the full
    # preserved-visual box set.  This catches patches planned before v2.1 added
    # missing protections.
    plan.preserved_underlays = _dedupe_preserved_units(plan.preserved_underlays)
    plan.preserved_overlays = _dedupe_preserved_units(plan.preserved_overlays)
    preserved_boxes = [g["bbox"] for g in groups if _valid_box(g.get("bbox"))]
    plan.patches = _filter_patches_around_preserved_visual(plan.patches, preserved_boxes, findings)
    if added_prot or added_pres:
        findings.append({
            "type": "ownership_lifecycle_v2_1_render_contract_enforced",
            "severity": "info",
            "preserved_visual_group_count": len(groups),
            "added_protected_region_count": added_prot,
            "added_preserved_layer_count": added_pres,
            "merged_protected_source_id_item_count": locals().get("merged_prot_sids", 0),
            "merged_preserved_source_id_item_count": locals().get("merged_pres_sids", 0),
        })

def compile_page_render_plan(translated_input_data: dict, *, reconstruction_mode: str = "debug") -> PageRenderPlan:
    normalized = PageReconstructInputAdapter().normalize(translated_input_data)

    units = normalized["units"]
    unit_index = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    children_map = {u.get("unit_id"): (u.get("children_ids") or []) for u in units if isinstance(u, dict)}

    translated_units = normalized["translated_units"]
    preservation_plan = normalized["preservation_plan"]
    exclusion_plan = normalized["exclusion_plan"]
    reconstruction_plan = normalized["reconstruction_plan"]
    style_system = normalized.get("style_system") or {}
    geom = (normalized["page"].get("geometry") or {})
    page_width_pt = geom.get("width")
    page_height_pt = geom.get("height")

    # Zone = source of truth: hard formula/code/protected detector zones.
    # They are never neutralized merely because an upstream translation unit
    # overlaps them; instead the overlapping translation unit is skipped below
    # and the zone is preserved as an object block.
    special_zones = _build_special_zones(normalized["regions"])
    source_ownership = build_source_ownership(normalized)
    normalized["source_ownership"] = source_ownership

    # Index the reconstruction_plan so each translated unit can pull its style /
    # render_contract / role (directive §9, lot 1).
    plan_by_tuid, plan_by_uid, plan_by_sid = {}, {}, {}
    for entry in reconstruction_plan:
        if entry.get("translation_unit_id"):
            plan_by_tuid[entry["translation_unit_id"]] = entry
        if entry.get("unit_id"):
            plan_by_uid[entry["unit_id"]] = entry
        for sid in entry.get("source_unit_ids") or []:
            plan_by_sid.setdefault(sid, entry)

    def _plan_for(item):
        return (plan_by_tuid.get(item.get("translation_unit_id"))
                or plan_by_uid.get(item.get("unit_id"))
                or next((plan_by_sid.get(s) for s in (item.get("source_unit_ids") or []) if s in plan_by_sid), None))

    findings: list[dict] = []

    # consumed: source ids covered by a chosen reconstruction unit (+ descendants).
    consumed: set = set()
    for item in translated_units:
        sids = list(item.get("source_unit_ids") or [])
        if all_source_ids_non_translatable(source_ownership, sids):
            continue
        for sid in sids:
            if is_non_translatable_owner(source_ownership, sid):
                continue
            consumed.add(sid)
            if item.get("consume_source_units") or item.get("preferred_over_children"):
                consumed |= {d for d in _descendants(sid, children_map) if not is_non_translatable_owner(source_ownership, d)}

    excluded: set = set()
    for e in exclusion_plan:
        for sid in e.get("source_unit_ids") or []:
            excluded.add(sid)

    protected_index = build_protected_region_index(
        units=unit_index, preservation_plan=preservation_plan, exclusion_plan=exclusion_plan,
        regions=normalized["regions"], visual_layers=normalized["visual_layers"],
        translated_source_ids=consumed, special_zones=special_zones,
    )

    # translated text layer (no parent/child double render).
    translated_text: list[TranslatedTextUnit] = []
    rendered_ids: set = set()
    rendered_boxes: list = []
    for i, item in enumerate(translated_units, start=1):
        sids = list(item.get("source_unit_ids") or [])
        if all_source_ids_non_translatable(source_ownership, sids):
            for s in sids:
                excluded.add(s)
                rendered_ids.add(s)
            findings.append({
                "type": "translated_unit_suppressed_by_source_ownership",
                "translation_unit_id": item.get("translation_unit_id"),
                "source_unit_ids": sids,
                "severity": "info",
            })
            continue
        if sids and all(s in rendered_ids for s in sids):
            findings.append({"type": "duplicate_render_skipped", "source_unit_ids": sids,
                             "translation_unit_id": item.get("translation_unit_id")})
            continue
        rt = item.get("render_target") or {}
        bbox = item.get("layout_bbox") or rt.get("layout_bbox") or rt.get("bbox") or item.get("bbox")
        # Inside a confident formula/code zone: keep original pixels, do not
        # translate or paint this fragment (zone-as-source-of-truth).
        zone = _in_special_zone(item.get("patch_bbox") or rt.get("patch_bbox") or bbox, special_zones)
        if zone is not None:
            for s in sids:
                excluded.add(s)
                rendered_ids.add(s)
            findings.append({"type": "unit_inside_special_zone", "zone_kind": zone["kind"],
                             "translation_unit_id": item.get("translation_unit_id"), "severity": "info"})
            continue
        anchor_line = item.get("anchor_bbox") or rt.get("anchor_bbox")
        budget = item.get("layout_budget") or {}
        plan_item = _plan_for(item)
        # Role: most specific available (reconstruction_plan > translation unit).
        role = _normalise_role((plan_item or {}).get("role") or item.get("role"), bbox, item, page_height=page_height_pt)
        coverage = item.get("coverage_bbox") or rt.get("coverage_bbox") or _coverage_bbox(sids, unit_index, bbox)
        # Font size uses the single-line anchor height (not the full block).
        style = resolve_style(item, plan_item, unit_index, style_system, role=role, line_bbox=anchor_line or bbox)
        if style.get("confidence", 0) <= 0.2:
            findings.append({"type": "unresolved_style", "translation_unit_id": item.get("translation_unit_id"),
                             "severity": "review"})
        # ABSOLUTE PRIORITY fix: flow text must lay out in the full block, not one line.
        layout_bbox, patch_bbox, _anchor, lb_findings = resolve_layout(role, bbox, coverage)
        patch_bbox = item.get("patch_bbox") or rt.get("patch_bbox") or patch_bbox
        # Geometric duplicate guard: the same region emitted twice (a block as
        # body_paragraph AND list_item, or two blocks with identical text/bbox)
        # would stack and self-collide. Keep the first, drop near-coincident ones.
        if _is_duplicate_box(layout_bbox, rendered_boxes):
            findings.append({"type": "duplicate_render_skipped", "source_unit_ids": sids,
                             "translation_unit_id": item.get("translation_unit_id")})
            continue
        rendered_boxes.append(layout_bbox)
        for f in lb_findings:
            findings.append({**f, "translation_unit_id": item.get("translation_unit_id")})
        translated_text.append(TranslatedTextUnit(
            id=f"ru_{i:04d}", kind="translated_text",
            renderer=choose_renderer(role, item.get("object_type")),
            source_unit_ids=sids, translation_unit_id=item.get("translation_unit_id"),
            source_text=item.get("text"), translated_text=item.get("translated_text"),
            role=role, object_type=item.get("object_type"), semantic_kind=item.get("semantic_kind"),
            page_role=item.get("page_role"), bbox=bbox,
            coverage_bbox=coverage, layout_bbox=layout_bbox, patch_bbox=patch_bbox,
            bbox_reliable=bool(budget.get("bbox_reliable", bbox is not None)),
            style=style, render_target=rt,
            render_contract=(plan_item or {}).get("render_contract") or item.get("render_contract") or {},
        ))
        for s in sids:
            rendered_ids.add(s)
            rendered_ids |= _descendants(s, children_map)

    # 100% TEXT SURVIVAL SAFETY NET.
    # If PAGEPRINT/PAGETRANSLATE failed to promote a visible text line into
    # translated_units, do NOT let it disappear. Render it in-place with its
    # source text, marked as identity_fallback. This is not a linguistic success;
    # it is a non-negotiable reconstruction invariant: source text must survive.
    next_ru_index = len(translated_text) + 1
    for u in units:
        uid = u.get("unit_id")
        if not uid or _source_unit_is_rendered_or_covered(uid, rendered_ids, consumed, excluded):
            continue
        if is_non_translatable_owner(source_ownership, uid):
            continue
        if not _is_text_survival_candidate(u):
            continue
        bbox_u = _unit_bbox(u)
        text_u = _unit_text(u)
        role_u = _normalise_role(_unit_role(u) or "body_paragraph", bbox_u, {"source_text": text_u}, page_height=page_height_pt)
        if role_u in {"formula_expression", "code_line", "code_block", "diagram_label", "axis_label", "legend_label"}:
            # Preserve technical/diagram text as source, but still make it a
            # block so it is drawn/copied in the final rendering path.
            translated_value = text_u
        else:
            translated_value = text_u
        style_u = resolve_style({"source_unit_ids": [uid], "bbox": bbox_u, "text": text_u}, None, unit_index, style_system, role=role_u, line_bbox=bbox_u)
        tunit = TranslatedTextUnit(
            id=f"ru_{next_ru_index:04d}", kind="translated_text",
            renderer=choose_renderer(role_u, u.get("object_type")),
            source_unit_ids=[uid], translation_unit_id=f"identity_fallback::{uid}",
            source_text=text_u, translated_text=translated_value,
            role=role_u or "body_paragraph", object_type=u.get("object_type") or "natural_text",
            semantic_kind=u.get("semantic_kind"), page_role=normalized["page"].get("page_role"),
            bbox=bbox_u, coverage_bbox=bbox_u, layout_bbox=bbox_u, patch_bbox=bbox_u,
            bbox_reliable=True, style=style_u, render_target={"identity_fallback": True},
            render_contract={"mode": "identity_fallback_text_survival"},
        )
        translated_text.append(tunit)
        rendered_ids.add(uid)
        consumed.add(uid)
        rendered_boxes.append(bbox_u)
        next_ru_index += 1
        findings.append({"type": "identity_fallback_text_survival", "source_unit_id": uid, "severity": "review"})

    # preserved layers from preservation_plan (over_text vs original).
    # A source unit cannot be both translated and preserved, except for a small
    # immutable child such as a real page number, formula/code or protected label.
    # This removes the typical duplicate overlays "C" / "2" from CHAPTER headers
    # and other false page-number fragments.
    underlays, overlays = [], []

    # Existing preservation_plan entries already represent some formula/code/
    # protected visual objects. Special detector zones strengthen protection,
    # but must not create a second preserved layer for the same source object.
    existing_preserve_sids = set()
    existing_preserve_boxes = []
    for _p in preservation_plan or []:
        existing_preserve_sids.update(_p.get("source_unit_ids") or [])
        _b = _p.get("bbox")
        if isinstance(_b, (list, tuple)) and len(_b) == 4:
            existing_preserve_boxes.append([float(x) for x in _b])

    def _zone_already_preserved(zone: dict) -> bool:
        z_sids = set(zone.get("source_unit_ids") or [])
        if z_sids and (z_sids & existing_preserve_sids):
            return True
        z_box = zone.get("bbox")
        if not (isinstance(z_box, (list, tuple)) and len(z_box) == 4):
            return False
        z_box = [float(x) for x in z_box]
        for _b in existing_preserve_boxes:
            if _contained_ratio(z_box, _b) >= 0.80 and _contained_ratio(_b, z_box) >= 0.80:
                return True
        return False

    preserve_index = 1
    for zidx, zone in enumerate(special_zones or [], start=1):
        if _zone_already_preserved(zone):
            for sid in zone.get("source_unit_ids") or []:
                excluded.add(sid)
                rendered_ids.add(sid)
            findings.append({
                "type": "special_zone_preservation_deduped",
                "zone_kind": zone.get("kind"),
                "bbox": zone.get("bbox"),
                "source_unit_ids": list(zone.get("source_unit_ids") or []),
                "severity": "info",
            })
            continue
        pu = PreservedUnit(
            id=f"pres_special_{zidx:04d}",
            source="special_zone",
            reason=zone.get("kind") or "protected_visual",
            bbox=zone.get("bbox"),
            text=None,
            preservation_mode="preserve_as_visual_overlay",
            source_unit_ids=list(zone.get("source_unit_ids") or []),
            z_policy="preserve_original",
        )
        underlays.append(pu)
        for sid in zone.get("source_unit_ids") or []:
            excluded.add(sid)
            rendered_ids.add(sid)
        findings.append({
            "type": "special_zone_preserved_as_underlay",
            "zone_kind": zone.get("kind"),
            "bbox": zone.get("bbox"),
            "source_unit_ids": list(zone.get("source_unit_ids") or []),
            "severity": "info",
        })
    preserve_index = len(underlays) + 1
    for p in preservation_plan:
        source_ids = list(p.get("source_unit_ids") or [])
        if _source_related_to_consumed(source_ids, consumed) and not _must_preserve_even_if_consumed(p, page_width_pt, page_height_pt):
            findings.append({
                "type": "preservation_skipped_consumed_source_unit",
                "source_unit_ids": source_ids,
                "reason": p.get("reason"),
                "severity": "info",
            })
            continue
        if str(p.get("reason") or "").lower() == "page_number" and not _is_valid_page_number_preservation(p, page_width_pt, page_height_pt):
            findings.append({
                "type": "invalid_page_number_preservation_skipped",
                "text": p.get("text"),
                "bbox": p.get("bbox"),
                "severity": "info",
            })
            continue
        mode = p.get("preservation_mode")
        z = "over_text" if mode == "preserve_text_exactly" else "preserve_original"
        pu = PreservedUnit(id=f"pres_{preserve_index:04d}", source="preservation_plan", reason=str(p.get("reason") or "preserve"),
                           bbox=p.get("bbox"), text=p.get("text"), preservation_mode=mode,
                           source_unit_ids=source_ids, z_policy=z)
        preserve_index += 1
        (overlays if z == "over_text" else underlays).append(pu)
    underlays = _dedupe_preserved_units(underlays)
    overlays = _dedupe_preserved_units(overlays)

    if normalized["page"].get("page_role") and not translated_text:
        findings.append({"type": "no_translated_text", "severity": "review"})

    page = {
        "page_index": (normalized["page"].get("page_index")),
        "width_pt": geom.get("width"), "height_pt": geom.get("height"),
        "rotation": normalized["page"].get("rotation") or 0,
        "coordinate_unit": geom.get("unit") or "pt",
        "coordinate_origin": geom.get("origin") or "top_left",
        "render_width_px": geom.get("render_width_px"), "render_height_px": geom.get("render_height_px"),
        "page_role": normalized["page"].get("page_role"),
    }

    patches, patch_findings = plan_patches(translated_text, protected_index)
    findings.extend(patch_findings)

    background = resolve_background(normalized)
    findings.extend(background.get("findings") or [])

    # Publication gate: incomplete/ko translations cannot be reconstructed as
    # publication-ready (directive PR-Lot 1).
    tr = normalized.get("translation_result") or {}
    ling = tr.get("linguistic_quality_status")
    coverage = (((tr.get("linguistic_quality_validation") or {}).get("translation_coverage_ratio"))
                or (tr.get("quality") or {}).get("translation_coverage_ratio"))
    publication_blocked = False
    if reconstruction_mode == "publication":
        reasons = []
        if ling and ling != "ok":
            reasons.append(f"linguistic_quality_status={ling}")
        if coverage is not None and coverage < 0.98:
            reasons.append(f"translation_coverage_ratio={coverage}")
        if reasons:
            publication_blocked = True
            findings.append({"type": "publication_blocked", "severity": "ko", "reasons": reasons})
    render_policy = dict(DEFAULT_RENDER_POLICY)
    render_policy.update({"reconstruction_mode": reconstruction_mode,
                          "publication_blocked": publication_blocked,
                          "translation_coverage_ratio": coverage,
                          "source_ownership": source_ownership,
                          "source_ownership_summary": {
                              "total": len(source_ownership),
                          "non_translatable": sum(1 for e in source_ownership.values() if e.get("state") in {"preserved_visual", "preserved_text_exact", "excluded", "background_only", "background_visual"}),
                          }})

    plan = PageRenderPlan(
        page=page, translated_text=translated_text, background=[background],
        preserved_underlays=underlays, preserved_overlays=overlays, patches=patches,
        protected_regions=protected_index.regions,
        consumed_source_unit_ids=sorted(consumed), excluded_source_unit_ids=sorted(excluded),
        render_policy=render_policy, quality_expectations=dict(DEFAULT_QUALITY_EXPECTATIONS),
        findings=findings,
    )

    _enforce_preserved_visual_render_contract(plan, normalized, source_ownership, findings)

    # Freeze the FinalReconstructionContract + RenderOps so backends EXECUTE only
    # (no dispatch/measure/decision in the backend). Failure here never blocks the
    # legacy render path (backends fall back when render_ops is empty).
    try:
        from .final_contract import FinalReconstructionContract
        from .render_ops import build_render_ops
        from .text_removal_ledger import build_ledger
        from .composition.intrablock_composer import compose_contract
        from .page_level_contracts import PageNumberContract
        from .templates.book_figure_page import BookFigurePageTemplate
        plan_dict = plan.to_dict()
        contract = FinalReconstructionContract.from_pageprint_pagetranslate(normalized, plan_dict)
        plan.text_removal_ledger = [
            e.to_dict() for e in build_ledger(
                plan_dict,
                background_mode=background.get("mode"),
                clean_background_verified=background.get("clean_background_verified"),
            )
        ]
        plan.intrablock_compositions = [c.to_dict() for c in compose_contract(contract)]
        page_numbers = []
        for p in preservation_plan:
            if (p.get("role") == "page_number" or p.get("reason") == "page_number") and p.get("text"):
                page_numbers.append(PageNumberContract(
                    page_number=str(p.get("text")),
                    bbox=p.get("bbox") or [],
                    placement="header" if (p.get("bbox") or [0, 9999])[1] < 100 else "footer",
                ).to_dict())
        plan.page_level_contracts = {"page_numbers": page_numbers}
        tmpl = BookFigurePageTemplate()
        if tmpl.match(contract).matched:
            contract = tmpl.apply(contract)
        import os as _os
        enh_flag = _os.getenv("RECON_DISABLE_ENHANCERS", "").strip().lower() not in {"1", "true", "yes"}
        # (1) TYPOGRAPHIE em — géométrie-NEUTRE (n'écrase pas la taille de rendu) :
        # appliquée inconditionnellement. Une taille résolue depuis l'IMAGE
        # (cap/x-height, confiance ≥0.7) n'est plus "réparée" → retrait du finding
        # font_size_repaired (le score typo cesse d'être plafonné), sans bouger le rendu.
        if enh_flag and _os.getenv("RECON_DISABLE_TYPOGRAPHY", "").lower() not in {"1", "true", "yes"}:
            try:
                from .ocr_typography_engine import (
                    enhance_contract_typography, apply_typography_patches_in_place)
                tyres = enhance_contract_typography(contract, pageprint_data=normalized,
                                                    page_image_path=contract.background.source_image_path)
                apply_typography_patches_in_place(contract, tyres)
                em_ok = {b.block_id for b in contract.blocks
                         if getattr(b.style, "source", "") == "ocr_em_estimator"
                         and float(getattr(b.style, "confidence", 0) or 0) >= 0.70
                         and any(k in str(getattr(b.style, "typo_method", "")) for k in ("cap_height", "x_height"))}
                for t in plan.translated_text:
                    if t.id in em_ok and isinstance(t.style, dict):
                        kept = [f for f in (t.style.get("findings") or [])
                                if f.get("type") not in {"font_size_repaired_from_line_geometry", "font_size_inferred_from_line_geometry"}]
                        kept.append({"type": "font_size_resolved_from_image_em"})
                        t.style["findings"] = kept
                        t.style["size_source"] = "ocr_em_estimator"
                render_policy["typography_image_em"] = len(em_ok)
            except Exception as exc:  # pragma: no cover
                findings.append({"type": "external_typography_failed", "message": str(exc)})
        # (1.2) LOCK — text/background invariants. Diagnostic only: never removes text.
        try:
            from .invariant_guard import summarize_text_render_invariants
            _pre_inv = summarize_text_render_invariants(normalized, contract, [], background)
            render_policy["text_render_invariants_pre_reflow"] = _pre_inv
            if _pre_inv.get("missing_source_line_count"):
                findings.append({"type": "text_presence_invariant_violation_pre_reflow",
                                 "severity": "ko", "detail": _pre_inv})
        except Exception as exc:  # pragma: no cover
            findings.append({"type": "text_invariant_guard_failed", "message": str(exc), "severity": "review"})

        # (1.4) SPACING REFLOW — conservative CPU-only layout repacking.
        if enh_flag and _os.getenv("RECON_DISABLE_SPACING_REFLOW", "").lower() not in {"1", "true", "yes"}:
            import copy as _copy
            try:
                from .layout_reflow_solver import solve_spacing_reflow, apply_spacing_reflow_patches_in_place
                rf = solve_spacing_reflow(contract, enabled=True, normalized=normalized)
                if rf.status != "ko" and rf.patches_by_block_id:
                    cand = _copy.deepcopy(contract)
                    apply_spacing_reflow_patches_in_place(cand, rf)
                    cur_ops = build_render_ops(contract, plan_dict, mode=reconstruction_mode)
                    cand_ops = build_render_ops(cand, plan_dict, mode=reconstruction_mode)
                    if _ops_overlap_cost(cand_ops) <= _ops_overlap_cost(cur_ops) + 1e-6:
                        contract = cand
                        render_policy["spacing_reflow_applied"] = len(rf.patches_by_block_id)
                        render_policy["spacing_reflow_metrics"] = dict(rf.metrics)
                        findings.extend(rf.findings)
                    else:
                        render_policy["spacing_reflow_rejected"] = "overlap_cost_worse"
            except Exception as exc:  # pragma: no cover
                findings.append({"type": "spacing_reflow_failed", "message": str(exc), "severity": "review"})


        # (1.5) EXPANSION + REFLUX VOISINS — porte l'ancien comportement : garder
        # la police d'origine en ÉTENDANT le bloc vers le bas + poussant les blocs
        # voisins (cascade), plutôt que tasser le texte en petite police. Borné par
        # la page et les obstacles. Adopté seulement si n'aggrave pas les collisions.
        # Disabled by default: the expansion solver is experimental and has
        # been observed to move valid source bboxes far down the page when
        # obstacles/page size are incomplete.  100% text survival and bbox
        # fidelity take priority. Enable explicitly with RECON_ENABLE_EXPANSION=1.
        if enh_flag and _os.getenv("RECON_ENABLE_EXPANSION", "").lower() in {"1", "true", "yes"}:
            import copy as _copy
            try:
                from .block_expansion_solver import solve_block_expansion
                from .multiblock_layout_solver import apply_layout_patches_in_place as _apply_exp
                exp = solve_block_expansion(contract, enabled=True, normalized=normalized)
                if exp.status != "ko" and exp.patches_by_block_id:
                    cand = _copy.deepcopy(contract)
                    _apply_exp(cand, exp)
                    cand_ops = build_render_ops(cand, plan_dict, mode=reconstruction_mode)
                    cur_ops = build_render_ops(contract, plan_dict, mode=reconstruction_mode)
                    if _ops_overlap_cost(cand_ops) <= _ops_overlap_cost(cur_ops) + 1e-6:
                        contract = cand
                        render_policy["block_expansion_applied"] = len(exp.patches_by_block_id)
                        render_policy["block_expansion_engine"] = "flow_geometry_optimizer_v2"
                        try:
                            render_policy["block_expansion_regions"] = [getattr(r, "mode", "") for r in getattr(exp, "regions", [])]
                        except Exception:
                            pass
            except Exception as exc:  # pragma: no cover
                findings.append({"type": "block_expansion_failed", "message": str(exc)})

        # (2) MULTIBLOCK — change la géométrie : gardé net-improvement only.
        # AI-assisted publication geometry optimizer v3 is quarantined by default.
        # Enable only for controlled experiments: RECON_ENABLE_AI_LAYOUT_REFLOW=1.
        if enh_flag and _os.getenv("RECON_ENABLE_AI_LAYOUT_REFLOW", "").lower() in {"1", "true", "yes"}:
            try:
                from .flow_geometry_optimizer import solve_flow_geometry
                gf = solve_flow_geometry(contract, normalized=normalized, enabled=True)
                if gf.status != "ko" and gf.patches_by_block_id:
                    render_policy["ai_layout_reflow_v3_applied"] = len(gf.patches_by_block_id)
                    render_policy["ai_layout_reflow_v3_findings"] = list(gf.findings)[:20]
                    findings.extend(gf.findings)
            except Exception as exc:
                findings.append({"type": "ai_layout_reflow_v3_failed", "message": str(exc), "severity": "review"})

        # LOCK: layout optimizers must never invalidate text-safe geometry.
        try:
            from .layout_regression_guard import sanitize_contract_layouts_in_place
            sanitize_contract_layouts_in_place(contract, findings=findings, render_policy=render_policy)
        except Exception as exc:  # pragma: no cover
            findings.append({"type": "layout_regression_guard_failed", "message": str(exc), "severity": "review"})

        baseline_ops = build_render_ops(contract, plan_dict, mode=reconstruction_mode)
        chosen_ops = baseline_ops
        if enh_flag and _os.getenv("RECON_DISABLE_MULTIBLOCK", "").lower() not in {"1", "true", "yes"}:
            import copy as _copy
            try:
                from .multiblock_layout_solver import (
                    solve_multiblock_layout, apply_layout_patches_in_place)
                enh = _copy.deepcopy(contract)
                mb = solve_multiblock_layout(enh, enabled=True)
                if mb.status != "ko" and mb.patches_by_block_id:
                    apply_layout_patches_in_place(enh, mb)
                    # Multiblock runs after the general regression guard.  Guard
                    # its candidate as well, otherwise the last optimizer can
                    # move source-anchored lines without any final validation.
                    sanitize_contract_layouts_in_place(enh, findings=findings, render_policy=render_policy)
                    enh_ops = build_render_ops(enh, plan_dict, mode=reconstruction_mode)
                    if _ops_overlap_cost(enh_ops) < _ops_overlap_cost(baseline_ops) - 1e-6:
                        contract, chosen_ops = enh, enh_ops
                        render_policy["multiblock_applied"] = len(mb.patches_by_block_id)
            except Exception as exc:  # pragma: no cover
                findings.append({"type": "external_multiblock_failed", "message": str(exc)})
        plan.render_ops = [op.to_dict() for op in chosen_ops]
        try:
            from .invariant_guard import summarize_text_render_invariants
            _post_inv = summarize_text_render_invariants(normalized, contract, chosen_ops, background)
            render_policy["text_render_invariants_post_reflow"] = _post_inv
            if _post_inv.get("missing_source_line_count"):
                findings.append({"type": "text_presence_invariant_violation_post_reflow",
                                 "severity": "ko", "detail": _post_inv})
            if not _post_inv.get("clean_background_verified"):
                findings.append({"type": "clean_background_not_locked", "severity": "ko", "detail": _post_inv})
        except Exception as exc:  # pragma: no cover
            findings.append({"type": "text_invariant_guard_failed_post", "message": str(exc), "severity": "review"})
        from .source_text_lifecycle_ledger import build_source_text_lifecycle_ledger
        plan.source_text_lifecycle_ledger = [
            e.to_dict() for e in build_source_text_lifecycle_ledger(plan.to_dict(), normalized)
        ]
        plan.final_contract = contract.to_dict()
        try:
            ownership_audit = audit_source_ownership(plan.to_dict(), normalized)
            plan.render_policy["source_ownership_audit"] = {
                "status": ownership_audit.get("status"),
                "conflict_count": ownership_audit.get("conflict_count"),
                "hard_blockers": ownership_audit.get("hard_blockers") or [],
            }
            if ownership_audit.get("status") != "ok":
                findings.append({"type": "source_ownership_conflict", "severity": "ko", "detail": ownership_audit})
        except Exception as exc:  # pragma: no cover
            findings.append({"type": "source_ownership_audit_failed", "message": str(exc), "severity": "review"})

        # Ownership/Lifecycle v2: verify that preserved_visual ownership really
        # propagates to the render contract: protected region + preserved layer +
        # PreservationOp, with no leak into translation units, translated_text,
        # TextOp or destructive PatchOp.  This is a hard diagnostic gate; it does
        # not repair layout, it prevents silent contract regression.
        try:
            from render_contract_audit import audit_render_contract, compact_render_contract_audit
            render_contract_audit = audit_render_contract(plan.to_dict(), normalized)
            plan.render_policy["render_contract_audit"] = compact_render_contract_audit(render_contract_audit)
            if render_contract_audit.get("status") != "ok":
                findings.append({
                    "type": "render_contract_propagation_conflict",
                    "severity": "ko",
                    "detail": compact_render_contract_audit(render_contract_audit, max_rows=10),
                })
        except Exception as exc:  # pragma: no cover
            findings.append({"type": "render_contract_audit_failed", "message": str(exc), "severity": "review"})
    except Exception as exc:  # pragma: no cover
        findings.append({"type": "render_ops_build_failed", "message": str(exc)})

    return plan


class PageRenderPlanCompiler:
    def compile(self, normalized_or_data: dict) -> PageRenderPlan:
        # Accept either raw translated_input_data or a dict already carrying views.
        return compile_page_render_plan(normalized_or_data)
