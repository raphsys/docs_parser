from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fitz


@dataclass
class LayoutPlacement:
    block_id: str
    dx: float = 0.0
    dy: float = 0.0
    reason: str = ""


def _rect(value: Any) -> fitz.Rect | None:
    if isinstance(value, fitz.Rect):
        rect = fitz.Rect(value)
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            rect = fitz.Rect([float(v) for v in value])
        except Exception:
            return None
    else:
        return None
    return rect if rect.get_area() > 0 else None


def _union(rects: list[fitz.Rect]) -> fitz.Rect | None:
    usable = [rect for rect in rects if isinstance(rect, fitz.Rect) and rect.get_area() > 0]
    if not usable:
        return None
    out = fitz.Rect(usable[0])
    for rect in usable[1:]:
        out |= rect
    return out


def _overlap_area(a: fitz.Rect, b: fitz.Rect) -> float:
    return max(0.0, (a & b).get_area())


def _h_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    return inter / max(1.0, min(a.width, b.width))


def _same_column(a: fitz.Rect, b: fitz.Rect) -> bool:
    return _h_overlap_ratio(a, b) >= 0.42 or abs(a.x0 - b.x0) <= 18.0


def _text_group_key(item: dict) -> tuple[float, float, str]:
    rect = item["rect"]
    return (round(rect.y0, 2), round(rect.x0, 2), str(item.get("block_id") or ""))


def _make_groups(text_items: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for item in sorted(text_items, key=_text_group_key):
        rect = item["rect"]
        role = str(item.get("role") or "").lower()
        placed = False
        for group in groups:
            g_rect = group["rect"]
            gap = rect.y0 - g_rect.y1
            if gap < -2.0:
                continue
            if gap <= 14.0 and _same_column(rect, g_rect):
                group["items"].append(item)
                group["rect"] = g_rect | rect
                group["roles"].add(role)
                placed = True
                break
        if not placed:
            groups.append({"items": [item], "rect": fitz.Rect(rect), "roles": {role}})
    return groups


def _collides(rect: fitz.Rect, occupied: list[fitz.Rect], tolerance: float = 0.5) -> bool:
    return any(_overlap_area(rect, other) > tolerance for other in occupied)


def _next_clear_y(rect: fitz.Rect, occupied: list[fitz.Rect], gap: float) -> float:
    y = rect.y0
    changed = True
    while changed:
        changed = False
        test = fitz.Rect(rect.x0, y, rect.x1, y + rect.height)
        for other in occupied:
            if _overlap_area(test, other) <= 0.5:
                continue
            y = max(y, other.y1 + gap)
            changed = True
            break
    return y


def _candidate_y_positions(rect: fitz.Rect, occupied: list[fitz.Rect], page: fitz.Rect, margin: float, gap: float) -> list[float]:
    values = [rect.y0, page.y0 + margin]
    for other in occupied:
        values.append(other.y1 + gap)
        values.append(other.y0 - gap - rect.height)
    out = []
    seen = set()
    for value in values:
        value = max(page.y0 + margin, min(float(value), page.y1 - margin - rect.height))
        key = round(value, 2)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return sorted(out, key=lambda y: (abs(y - rect.y0), y))


def _candidate_x_positions(rect: fitz.Rect, page: fitz.Rect, margin: float) -> list[float]:
    values = [
        rect.x0,
        page.x0 + margin,
        page.x1 - margin - rect.width,
        page.x0 + margin + max(0.0, (page.width - 2 * margin - rect.width) / 2.0),
    ]
    out = []
    seen = set()
    for value in values:
        value = max(page.x0 + margin, min(float(value), page.x1 - margin - rect.width))
        key = round(value, 2)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return sorted(out, key=lambda x: (abs(x - rect.x0), x))


def compile_page_layout(
    *,
    page_bbox: tuple[float, float, float, float] | list[float],
    text_items: list[dict],
    fixed_items: list[dict],
    margin: float = 4.0,
    gap: float = 4.0,
    allow_partial: bool = False,
) -> dict:
    """Compile a page-level no-overlap layout from existing rendered extents.

    This compiler moves whole text blocks/groups only. It never shrinks text,
    never moves fixed regions, and returns no placement if constraints cannot
    be satisfied inside the page.
    """
    page = _rect(page_bbox) or fitz.Rect(0, 0, 10_000, 10_000)
    fixed_rects = [_rect(item.get("bbox")) for item in fixed_items or []]
    fixed_rects = [rect for rect in fixed_rects if rect is not None]
    normalized_text = []
    for item in text_items or []:
        rect = _rect(item.get("bbox"))
        if rect is None:
            continue
        normalized_text.append({**item, "rect": rect})
    groups = _make_groups(normalized_text)
    def place_groups(*, nearest: bool) -> tuple[bool, list[LayoutPlacement], list[str], int, list[str]]:
        occupied = list(fixed_rects)
        placements: list[LayoutPlacement] = []
        warnings: list[str] = []
        moved_groups = 0
        overflow_ids: list[str] = []
        ordered_groups = sorted(groups, key=lambda g: (g["rect"].y0, g["rect"].x0))
        for group in ordered_groups:
            original = group["rect"]
            best = None
            x_values = _candidate_x_positions(original, page, margin)
            if not nearest:
                x_values = sorted(x_values)
            for target_x in x_values:
                candidate = fitz.Rect(target_x, original.y0, target_x + original.width, original.y1)
                y_values = _candidate_y_positions(candidate, occupied, page, margin, gap)
                if not nearest:
                    y_values = sorted(y_values)
                for target_y in y_values:
                    if target_y < original.y0 - 2.0:
                        continue
                    trial = fitz.Rect(target_x, target_y, target_x + original.width, target_y + original.height)
                    if trial.y0 < page.y0 + margin or trial.y1 > page.y1 - margin:
                        continue
                    if _collides(trial, occupied):
                        continue
                    best = trial
                    break
                if best is not None:
                    break
            if best is None and nearest:
                for target_x in x_values:
                    candidate = fitz.Rect(target_x, original.y0, target_x + original.width, original.y1)
                    target_y = _next_clear_y(candidate, occupied, gap)
                    trial = fitz.Rect(target_x, target_y, target_x + original.width, target_y + original.height)
                    if trial.y0 >= page.y0 + margin and trial.y1 <= page.y1 - margin and not _collides(trial, occupied):
                        best = trial
                        break
            if best is None:
                group_ids = [str(i.get("block_id") or "") for i in group["items"] if str(i.get("block_id") or "")]
                warnings.append(f"no_free_slot:{','.join(group_ids)}")
                overflow_ids.extend(group_ids)
                if allow_partial:
                    for rest_group in ordered_groups[ordered_groups.index(group) + 1:]:
                        overflow_ids.extend(str(i.get("block_id") or "") for i in rest_group["items"] if str(i.get("block_id") or ""))
                    break
                return False, [], warnings, moved_groups, overflow_ids
            candidate = best
            if _collides(candidate, occupied):
                warnings.append(f"residual_collision:{','.join(str(i.get('block_id') or '') for i in group['items'])}")
                if allow_partial:
                    overflow_ids.extend(str(i.get("block_id") or "") for i in group["items"] if str(i.get("block_id") or ""))
                    for rest_group in ordered_groups[ordered_groups.index(group) + 1:]:
                        overflow_ids.extend(str(i.get("block_id") or "") for i in rest_group["items"] if str(i.get("block_id") or ""))
                    break
                return False, [], warnings, moved_groups, overflow_ids
            dx = candidate.x0 - original.x0
            dy = candidate.y0 - original.y0
            if abs(dx) >= 0.25 or abs(dy) >= 0.25:
                moved_groups += 1
            for item in group["items"]:
                placements.append(
                    LayoutPlacement(
                        block_id=str(item.get("block_id") or ""),
                        dx=dx,
                        dy=dy,
                        reason="document_layout_compiler_group_pack" if nearest else "document_layout_compiler_global_repack",
                    )
                )
            occupied.append(candidate)
        return True, placements, warnings, moved_groups, overflow_ids

    ok, placements, warnings, moved_groups, overflow_ids = place_groups(nearest=True)
    if not ok:
        ok, placements, warnings, moved_groups, overflow_ids = place_groups(nearest=False)
    if not ok:
        return {"ok": False, "placements": [], "warnings": warnings, "overflow_block_ids": overflow_ids}

    return {
        "ok": True,
        "partial": bool(overflow_ids),
        "placements": [placement.__dict__ for placement in placements if placement.block_id],
        "warnings": warnings,
        "moved_groups": moved_groups,
        "overflow_block_ids": sorted(set(overflow_ids)),
    }
