#!/usr/bin/env python3
"""Translate a single sentence through a translation engine (smoke test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_engines import create_translation_engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="ct2")
    parser.add_argument("--inventory", default=None, help="Path to model_inventory.json")
    parser.add_argument("--model", default=None, help="Preferred model name")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--text", required=True, help="Source text to translate")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report instead of plain text")
    args = parser.parse_args()

    engine = create_translation_engine(
        args.engine,
        inventory_path=args.inventory,
        model_name=args.model,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
    )
    outputs = engine.translate_batch([
        {"text": args.text, "source_lang": args.source_lang, "target_lang": args.target_lang, "context": {}}
    ])
    output = outputs[0] if outputs else {"translated_text": "", "metadata": {}}
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(output.get("translated_text") or "")
    return 0 if (output.get("translated_text") or "").strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
