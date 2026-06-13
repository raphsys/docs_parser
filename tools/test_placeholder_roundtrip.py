#!/usr/bin/env python3
"""Smoke-test placeholder styles against the configured translation engine.

By default only the engine's production placeholder style is tested (the one
``choose_placeholder_style`` would use), so the reported corruption rate matches
what the pipeline really uses. Pass ``--all-styles`` to sweep every style and
discover which format survives a given model, and ``--write-policy`` to persist
the best clean style to ``placeholder_policy.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagetranslate.protection import audit_placeholders, protect_text, restore_text
from translation_engines import create_translation_engine
from translation_engines.placeholder_policy import (
    PLACEHOLDER_STYLES,
    build_placeholder,
    choose_placeholder_style,
    placeholder_variants,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=None)
    parser.add_argument("--inventory", default=None, help="Path to model_inventory.json")
    parser.add_argument("--model", default=None, help="Preferred model name")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--all-styles", action="store_true", help="Test every placeholder style, not just the engine's.")
    parser.add_argument("--write-policy", default=None, help="Write the chosen clean style to this JSON path.")
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
    engine_style = choose_placeholder_style(engine_name=getattr(engine, "profile", None))
    styles = list(PLACEHOLDER_STYLES) if args.all_styles else [engine_style]

    results = []
    total_corruption = 0
    for idx, style in enumerate(styles, start=1):
        placeholder = build_placeholder(idx, style)
        source = f"Keep https://example.org/{idx} safe."
        protected, protections = protect_text(source, placeholder_style=style)
        translated_protected = engine.translate(protected, args.source_lang, args.target_lang, {})
        restored = restore_text(translated_protected, protections)
        audit = audit_placeholders(restored, protections)
        total_corruption += int(audit["placeholder_corruption_count"])
        roundtrip_ok = audit["placeholder_corruption_count"] == 0 or any(
            variant in translated_protected for variant in placeholder_variants(idx)
        )
        results.append({
            "style": style,
            "placeholder": placeholder,
            "roundtrip_ok": bool(roundtrip_ok),
            "placeholder_corruption_count": audit["placeholder_corruption_count"],
            "audit": audit,
            "restored": restored,
        })

    clean_styles = [item["style"] for item in results if item["placeholder_corruption_count"] == 0]
    chosen_style = engine_style if engine_style in clean_styles else (clean_styles[0] if clean_styles else None)
    payload = {
        "engine": getattr(engine, "profile", type(engine).__name__),
        "engine_style": engine_style,
        "tested_styles": styles,
        "clean_styles": clean_styles,
        "chosen_style": chosen_style,
        "results": results,
        "placeholder_corruption_count": total_corruption,
        "placeholder_corruption_rate": round(total_corruption / max(1, len(results)), 3),
    }

    if args.write_policy and chosen_style:
        Path(args.write_policy).write_text(
            json.dumps({"placeholder_style": chosen_style, "engine": payload["engine"]}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        payload["policy_written"] = args.write_policy

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if total_corruption == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
