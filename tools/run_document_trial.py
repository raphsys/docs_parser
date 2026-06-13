#!/usr/bin/env python3
"""Extract a PDF through the pipeline, then run a controlled translation trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.orchestrator import PipelineOrchestrator
from tools.run_translation_trial import _trial_passes, run_translation_trial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf", help="PDF source file")
    parser.add_argument("--pages", default="1", help="Pages to process, e.g. '1' or '1,3-5' (1-based)")
    parser.add_argument("--engine", default=None, help="Translation engine name")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--inventory", default=None, help="Path to model_inventory.json")
    parser.add_argument("--model", default=None, help="Preferred model name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--fail-on", choices=["functional", "runtime", "quality", "publication", "any"], default="functional")
    parser.add_argument("--output-dir", default=None, help="Folder to store input_data and trial_result JSON")
    parser.add_argument("--enable-ocr", action="store_true")
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(enable_ocr=args.enable_ocr, enable_understanding=True, enable_postprocessors=True, enable_special_regions=True)
    document_result = orchestrator.run(args.source_pdf, pages=args.pages, language={"source_lang": args.source_lang, "target_lang": args.target_lang})
    pages = [page["input_data"] for page in document_result.get("pages") or [] if page.get("status") == "ok"]
    trial = run_translation_trial(
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

    payload = {
        "source_pdf": args.source_pdf,
        "document_result": document_result,
        "trial_result": trial,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "document_result.json").write_text(json.dumps(document_result, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "trial_result.json").write_text(json.dumps(trial, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "payload.json").write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    return 0 if _trial_passes(trial, args.fail_on) else 1


if __name__ == "__main__":
    raise SystemExit(main())
