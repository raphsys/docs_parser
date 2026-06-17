"""AI-assisted layout policy bridge for pagereconstruct.

This module *does not* let an LLM rewrite geometry directly.  It connects the
existing local AI assets as advisors and then returns a deterministic policy used
by the Python geometry solver.

Available advisors it can use when installed/enabled:
- pipeline_agents P3/P5/P7 (layout mode, render strategy, publication layout);
- ai_models/visual_layout/smolvlm-500m inventory signal;
- ai_models/ppstructurev3 layout-model inventory signal;
- ai_models/element_relations_nli inventory signal.

The solver remains deterministic and safe.  AI suggestions only bias policy:
prefer horizontal extension, merge atomic lines, keep baseline grid, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class AILayoutPolicy:
    provider: str = "heuristic"
    models_seen: list[str] | None = None
    prefer_horizontal_growth: bool = True
    merge_atomic_lines: bool = True
    preserve_baseline_grid: bool = True
    paragraph_mode: str = "paragraph_recompose"
    heading_mode: str = "prefer_width_then_height"
    min_gap_pt: float = 2.4
    paragraph_gap_pt: float = 4.0
    heading_gap_pt: float = 5.0
    max_width_growth_pt: float = 220.0
    max_heading_width_growth_pt: float = 280.0
    max_vertical_shift_pt: float = 180.0
    confidence: float = 0.62
    notes: list[str] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models_seen"] = list(self.models_seen or [])
        d["notes"] = list(self.notes or [])
        return d


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _models_root() -> Path:
    return _project_root() / "ai_models"


def _available_models() -> list[str]:
    root = _models_root()
    seen: list[str] = []
    checks = {
        "visual_layout.smolvlm-500m": root / "visual_layout" / "smolvlm-500m" / "model.safetensors",
        "ppstructurev3.PP-DocLayout_plus-L": root / "ppstructurev3" / "PP-DocLayout_plus-L" / "inference.pdiparams",
        "ppstructurev3.PP-DocBlockLayout": root / "ppstructurev3" / "PP-DocBlockLayout" / "inference.pdiparams",
        "element_relations_nli.onnx": root / "element_relations_nli" / "onnx" / "model_quint8_avx2.onnx",
        "inpainting.lama.onnx": root / "inpainting" / "lama" / "model.onnx",
    }
    for name, path in checks.items():
        if path.exists():
            seen.append(name)
    return seen


def _page_features(contract: Any, normalized: dict | None = None) -> dict:
    blocks = list(getattr(contract, "blocks", []) or [])
    boxes = []
    roles = []
    for b in blocks:
        layout = getattr(b, "layout", None)
        bb = getattr(layout, "layout_bbox", None) or getattr(layout, "coverage_bbox", None) or getattr(layout, "source_bbox", None)
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            boxes.append([float(x) for x in bb])
        roles.append(str(getattr(b, "role", "") or ""))
    page = getattr(contract, "page_info", None)
    ps = getattr(page, "page_size", None)
    return {
        "block_count": len(blocks),
        "roles": roles[:80],
        "page_size": ps,
        "bbox_count": len(boxes),
        "page_intelligence": (normalized or {}).get("page_intelligence") if isinstance(normalized, dict) else {},
    }


def _try_pipeline_agent(name: str, payload: dict) -> dict | None:
    try:
        from pipeline_agents.registry import get_agent  # type: ignore
        agent = get_agent(name)
        for meth in ("advise", "run", "__call__"):
            fn = getattr(agent, meth, None)
            if callable(fn):
                out = fn(payload)
                if isinstance(out, dict):
                    return out
                if hasattr(out, "to_dict"):
                    return out.to_dict()
    except Exception:
        return None
    return None


def build_ai_layout_policy(contract: Any, normalized: dict | None = None) -> AILayoutPolicy:
    models = _available_models()
    notes: list[str] = []
    payload = {"task": "publication_layout_policy", "features": _page_features(contract, normalized), "models": models}

    # P7 is the most relevant. P3/P5 are accepted as complementary advisors.
    for agent_name in ("p7_publication_layout", "p5_render", "p3_layout"):
        out = _try_pipeline_agent(agent_name, payload)
        if not out:
            continue
        pol = AILayoutPolicy(provider=f"pipeline_agents.{agent_name}", models_seen=models, notes=[f"agent:{agent_name}"])
        # Defensive interpretation: only simple booleans/numbers, never raw geometry.
        if str(out.get("layout_mode") or out.get("mode") or "").lower() in {"preserve_line_breaks", "anchored"}:
            pol.merge_atomic_lines = False
            pol.paragraph_mode = "preserve_line_breaks"
        if out.get("prefer_vertical") is True:
            pol.prefer_horizontal_growth = False
        if isinstance(out.get("max_width_growth_pt"), (int, float)):
            pol.max_width_growth_pt = max(40.0, min(320.0, float(out["max_width_growth_pt"])))
        pol.confidence = 0.70
        return pol

    if models:
        notes.append("local_ai_models_detected:" + ",".join(models))
    if any("smolvlm" in m for m in models):
        notes.append("visual_layout_model_available_for_future_scoring")
    if any("PP-DocLayout" in m or "PP-DocBlockLayout" in m for m in models):
        notes.append("ppstructure_layout_models_available")
    if any("element_relations" in m for m in models):
        notes.append("nli_relation_model_available")

    # Default policy favours horizontal growth first, because the observed error
    # was height expansion while free right space existed.
    return AILayoutPolicy(provider="heuristic+local_model_inventory", models_seen=models, notes=notes)
