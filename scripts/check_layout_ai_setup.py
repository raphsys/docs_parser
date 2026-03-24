#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layout_ai_enricher import LayoutAIEnricher  # noqa: E402


def main():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    enricher = LayoutAIEnricher()
    root = Path(enricher.models_root).expanduser() if enricher.models_root else None

    report = {
        "enabled": enricher.enabled,
        "backend": enricher.backend,
        "models_root": str(root) if root else "",
        "env": {
            "LAYOUT_AI_ENABLE": os.getenv("LAYOUT_AI_ENABLE"),
            "LAYOUT_AI_BACKEND": os.getenv("LAYOUT_AI_BACKEND"),
            "LAYOUT_AI_MODELS_ROOT": os.getenv("LAYOUT_AI_MODELS_ROOT"),
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": os.getenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"),
        },
        "required_model_dirs": {},
        "pipeline_init": {
            "attempted": False,
            "success": False,
            "error": None,
        },
    }

    for arg_name, model_dir_name in enricher.MINIMAL_MODELS.items():
        model_path = (root / model_dir_name) if root else None
        report["required_model_dirs"][model_dir_name] = {
            "arg_name": arg_name,
            "path": str(model_path) if model_path else "",
            "exists": bool(model_path and model_path.is_dir()),
        }

    can_try = enricher.enabled and root and all(v["exists"] for v in report["required_model_dirs"].values())
    if can_try:
        report["pipeline_init"]["attempted"] = True
        pipeline = enricher._get_pipeline()
        report["pipeline_init"]["success"] = pipeline is not None
        report["pipeline_init"]["error"] = enricher._load_error

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
