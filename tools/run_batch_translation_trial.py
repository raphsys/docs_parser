#!/usr/bin/env python3
"""Run translation trials over every JSON page payload in a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_functional_audit import _pages
from tools.run_translation_trial import _trial_passes, run_translation_trial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_folder", help="Folder containing PAGEPRINT JSON files.")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--inventory", default=None, help="Path to model_inventory.json")
    parser.add_argument("--model", default=None, help="Preferred model name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--fail-on", choices=["functional", "runtime", "quality", "publication", "any"], default="functional")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    folder = Path(args.audit_folder)
    pages = []
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages.extend(_pages(payload))
    result = run_translation_trial(
        pages,
        engine_name=args.engine,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        engine_kwargs={
            "inventory_path": args.inventory,
            "model_name": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
        },
        batch_size=args.batch_size,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if _trial_passes(result, args.fail_on) else 1


if __name__ == "__main__":
    raise SystemExit(main())
