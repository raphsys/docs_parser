#!/usr/bin/env python3
"""Report translation engine health."""

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
    parser.add_argument("--engine", default=None)
    parser.add_argument("--inventory", default=None, help="Path to model_inventory.json")
    parser.add_argument("--model", default=None, help="Preferred model name")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
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
    health = engine.healthcheck()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    return 0 if health.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
