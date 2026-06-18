#!/usr/bin/env python3
"""Audit one exclusive final owner for every visible PAGEPRINT source unit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_ownership import build_source_ownership, bbox_of, overlap_ratio


def _key(path: Path) -> str:
    name = path.stem
    for prefix in ("translated_input_data_", "pageprint_full_", "pagereconstruct_plan_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _related(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "_") or right.startswith(left + "_")


def _source_ids(items: list[dict], op_type: str | None = None) -> set[str]:
    out = set()
    for item in items or []:
        if not isinstance(item, dict) or (op_type and item.get("op_type") != op_type):
            continue
        out.update(str(s) for s in item.get("source_unit_ids") or [] if s)
    return out


def audit(plan: dict, data: dict) -> dict:
    ownership = build_source_ownership(data)
    children = {}
    for unit in data.get("units") or []:
        if not isinstance(unit, dict) or not unit.get("unit_id"):
            continue
        parent = unit.get("parent_id") or unit.get("parent_unit_id")
        if parent:
            children.setdefault(parent, []).append(unit["unit_id"])
    ops = [x for x in plan.get("render_ops") or [] if isinstance(x, dict)]
    text_ids = _source_ids(ops, "text")
    preservation_ids = _source_ids(ops, "preservation")
    patches = [x for x in ops if x.get("op_type") == "patch"]
    rows = []
    blockers = []

    for sid, entry in ownership.items():
        if not entry.get("bbox"):
            continue
        state = entry.get("state")
        # Containers may legitimately contain children with different owners
        # (translated title + exact page number). Audit atomic text leaves and
        # non-text visual objects, not mixed hierarchical containers.
        if entry.get("text") and children.get(sid):
            continue
        owners = []
        if any(_related(sid, root) for root in text_ids):
            owners.append("TextOp")
        if any(_related(sid, root) for root in preservation_ids):
            owners.append("PreservationOp")
        if state == "background_visual":
            owners.append("background_visual")
        if state in {"excluded", "background_only"}:
            owners.append("excluded")

        visible = bool(entry.get("text")) or state in {
            "preserved_visual", "preserved_text_exact", "background_visual"
        }
        if not visible:
            continue
        row_blockers = []
        if len(set(owners)) == 0:
            row_blockers.append("visible_source_without_final_owner")
        elif len(set(owners)) > 1:
            row_blockers.append("visible_source_has_multiple_final_owners")
        if owners and any(overlap_ratio(entry["bbox"], bbox_of(p)) >= 0.12 for p in patches if bbox_of(p)):
            row_blockers.append("patch_covers_final_owner")
        blockers.extend(row_blockers)
        if row_blockers:
            rows.append({"source_unit_id": sid, "state": state, "owners": owners, "blockers": row_blockers})

    blockers = sorted(set(blockers))
    return {
        "status": "ko" if blockers else "ok",
        "hard_blockers": blockers,
        "conflict_count": len(rows),
        "conflicts": rows[:100],
        "owner_counts": {
            "TextOp": len(text_ids),
            "PreservationOp": len(preservation_ids),
            "background_visual": sum(1 for e in ownership.values() if e.get("state") == "background_visual"),
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: audit_final_visual_ownership.py results/<run>", file=sys.stderr)
        return 2
    run = Path(argv[1])
    inputs = {_key(p): p for p in run.glob("translated_input_data_*.json")}
    reports = []
    for plan_path in sorted(run.glob("pagereconstruct_plan_*.json")):
        input_path = inputs.get(_key(plan_path))
        if not input_path:
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        data = json.loads(input_path.read_text(encoding="utf-8"))
        reports.append({"page_key": _key(plan_path), **audit(plan, data)})
    blockers = sorted({b for r in reports for b in r.get("hard_blockers") or []})
    result = {
        "status": "ko" if blockers else "ok",
        "hard_blockers": blockers,
        "report_count": len(reports),
        "ko_report_count": sum(r.get("status") == "ko" for r in reports),
        "reports": reports,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
