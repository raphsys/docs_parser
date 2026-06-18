#!/usr/bin/env python3
"""Audit Ownership/Lifecycle v1 for PAGEPRINT/PAGETRANSLATE/PAGERECONSTRUCT outputs.

Usage:
  python tools/audit_source_ownership.py <json-file-or-demo-dir>

For a demo directory, the script scans common pageprint/translated/reconstruct
JSON files and prints a compact conflict report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow direct execution from tools/:
#   python tools/audit_source_ownership.py results/<run>
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from source_ownership import audit_source_ownership, build_source_ownership


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _candidate_jsons(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    names = []
    for pat in ("**/*pageprint*.json", "**/*translated*.json", "**/*pagereconstruct*.json", "**/*render_plan*.json", "**/*plan*.json"):
        names.extend(path.glob(pat))
    return sorted(set(names))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python tools/audit_source_ownership.py <json-file-or-demo-dir>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.exists():
        print(f"introuvable: {root}", file=sys.stderr)
        return 2
    reports = []
    for jf in _candidate_jsons(root):
        data = _load_json(jf)
        if not isinstance(data, dict):
            continue
        # A render plan has layers/render_ops. A PAGEPRINT/translated input has units/regions.
        if data.get("units"):
            ownership = build_source_ownership(data)
            reports.append({
                "file": str(jf),
                "kind": "input_data",
                "ownership_count": len(ownership),
                "preserved_visual": sum(1 for e in ownership.values() if e.get("state") == "preserved_visual"),
                "preserved_text_exact": sum(1 for e in ownership.values() if e.get("state") == "preserved_text_exact"),
                "excluded": sum(1 for e in ownership.values() if e.get("state") in {"excluded", "background_only"}),
            })
        if data.get("layers") or data.get("render_ops"):
            # Use the same dict as plan and normalized when no separate normalized file is available.
            res = audit_source_ownership(data, data)
            reports.append({
                "file": str(jf),
                "kind": "render_plan",
                "status": res.get("status"),
                "conflict_count": res.get("conflict_count"),
                "hard_blockers": res.get("hard_blockers"),
            })
    print(json.dumps({"status": "ok", "reports": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
