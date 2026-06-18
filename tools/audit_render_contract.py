#!/usr/bin/env python3
"""Audit Ownership/Lifecycle v2 render propagation for one file or demo dir."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from render_contract_audit import audit_render_contract, compact_render_contract_audit


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _page_key(path: Path) -> str:
    name = path.name
    name = re.sub(r"\.json$", "", name)
    for prefix in (
        "pageprint_full_",
        "translated_input_data_",
        "pagereconstruct_plan_",
        "render_plan_",
        "plan_",
    ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _candidate_jsons(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    pats = (
        "**/pageprint_full_*.json",
        "**/translated_input_data_*.json",
        "**/pagereconstruct_plan_*.json",
        "**/render_plan_*.json",
    )
    files: list[Path] = []
    for pat in pats:
        files.extend(path.glob(pat))
    return sorted(set(files))


def _index_inputs(files: list[Path]) -> dict[str, Path]:
    # Prefer translated_input_data because it carries translation/runtime fields;
    # fall back to pageprint_full.
    idx: dict[str, Path] = {}
    for f in files:
        if f.name.startswith("pageprint_full_"):
            idx.setdefault(_page_key(f), f)
    for f in files:
        if f.name.startswith("translated_input_data_"):
            idx[_page_key(f)] = f
    return idx


def _plan_files(files: list[Path]) -> list[Path]:
    return [f for f in files if f.name.startswith(("pagereconstruct_plan_", "render_plan_", "plan_"))]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python tools/audit_render_contract.py <json-file-or-demo-dir>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.exists():
        print(f"introuvable: {root}", file=sys.stderr)
        return 2

    files = _candidate_jsons(root)
    inputs = _index_inputs(files)
    reports = []
    hard_blockers: set[str] = set()

    if root.is_file():
        data = _load_json(root)
        if isinstance(data, dict):
            # Single file mode: only meaningful if the file contains both units and render plan.
            if data.get("units") and (data.get("render_ops") or data.get("layers")):
                audit = audit_render_contract(data, data)
                reports.append({"file": str(root), **compact_render_contract_audit(audit)})
                hard_blockers.update(audit.get("hard_blockers") or [])
            else:
                reports.append({
                    "file": str(root),
                    "status": "skipped",
                    "reason": "single_file_requires_units_and_render_plan",
                })
    else:
        for pf in _plan_files(files):
            key = _page_key(pf)
            input_path = inputs.get(key)
            plan = _load_json(pf)
            data = _load_json(input_path) if input_path else None
            if not isinstance(plan, dict) or not isinstance(data, dict):
                reports.append({
                    "plan_file": str(pf),
                    "input_file": str(input_path) if input_path else None,
                    "status": "skipped",
                    "reason": "missing_plan_or_matching_input_data",
                })
                continue
            audit = audit_render_contract(plan, data)
            compact = compact_render_contract_audit(audit)
            reports.append({
                "page_key": key,
                "plan_file": str(pf),
                "input_file": str(input_path),
                **compact,
            })
            hard_blockers.update(audit.get("hard_blockers") or [])

    status = "ko" if hard_blockers or any(r.get("status") == "ko" for r in reports) else "ok"
    output = {
        "schema_version": "render_contract_audit.batch.v2",
        "status": status,
        "hard_blockers": sorted(hard_blockers),
        "report_count": len(reports),
        "ko_report_count": sum(1 for r in reports if r.get("status") == "ko"),
        "reports": reports,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if status == "ko" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
