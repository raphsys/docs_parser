#!/usr/bin/env python3
"""Run rev_04 functional audit over every JSON page payload in a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_functional_audit import audit_pages, _pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_folder", help="Folder containing PAGEPRINT JSON files.")
    parser.add_argument("--run-pagetranslate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder = Path(args.audit_folder)
    pages = []
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for page in _pages(payload):
            page.setdefault("debug", {}).setdefault("audit_source_file", str(path))
            pages.append(page)
    output = audit_pages(pages, run_pagetranslate=args.run_pagetranslate, dry_run=args.dry_run)
    output["audit_folder"] = str(folder)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["functional_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
