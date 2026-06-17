"""AI/model-aware layout advisory for pagereconstruct.

This module is intentionally non-invasive. It discovers the local models already
present in docs_parser/ai_models and turns them into layout *policy hints* for
the deterministic geometry solver.

Why not run a heavy VLM by default?
    - reconstruction must remain CPU-safe and deterministic;
    - missing optional runtimes must never break the pipeline;
    - the hard guarantees (text presence, clean background) must not depend on a
      generative model.

Still, the solver is not blind to the model inventory:
    - PP-DocLayout / PP-DocBlockLayout => stronger document-layout confidence;
    - visual_layout/smolvlm-500m       => visual-layout advisory available;
    - element_relations_nli            => relationship classifier available;
    - LAMA inpainting                  => background model available, but geometry
                                          still prefers deterministic cleanbg.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class LayoutModelAssets:
    root: str
    pp_doc_layout_plus_l: bool = False
    pp_doc_block_layout: bool = False
    pp_formula: bool = False
    pp_table_cell: bool = False
    visual_smolvlm_500m: bool = False
    element_relations_nli_onnx: bool = False
    lama_inpainting_onnx: bool = False
    embedded_fonts: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayoutPolicyHint:
    mode: str = "deterministic_model_aware"
    min_gap_pt: float = 2.0
    para_gap_pt: float = 3.0
    heading_gap_pt: float = 4.5
    max_width_growth_pt: float = 90.0
    prefer_width_growth_before_vertical_push: bool = True
    avoid_visual_obstacles: bool = True
    group_atomic_lines: bool = True
    confidence: float = 0.60
    assets: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _project_root_from_contract(contract: Any = None) -> Path:
    # The module lives in docs_parser/pagereconstruct. root is two levels above.
    return Path(__file__).resolve().parents[1]


def discover_layout_model_assets(project_root: str | Path | None = None) -> LayoutModelAssets:
    root = Path(project_root) if project_root else _project_root_from_contract()
    ai = root / "ai_models"
    pp = ai / "ppstructurev3"
    assets = LayoutModelAssets(root=str(root))
    assets.pp_doc_layout_plus_l = (pp / "PP-DocLayout_plus-L" / "inference.pdiparams").is_file()
    assets.pp_doc_block_layout = (pp / "PP-DocBlockLayout" / "inference.pdiparams").is_file()
    assets.pp_formula = (pp / "PP-FormulaNet_plus-L" / "inference.pdiparams").is_file()
    assets.pp_table_cell = (pp / "RT-DETR-L_wired_table_cell_det" / "inference.pdiparams").is_file() or (
        pp / "RT-DETR-L_wireless_table_cell_det" / "inference.pdiparams").is_file()
    assets.visual_smolvlm_500m = (ai / "visual_layout" / "smolvlm-500m" / "model.safetensors").is_file()
    assets.element_relations_nli_onnx = (
        ai / "element_relations_nli" / "onnx" / "model_quint8_avx2.onnx"
    ).is_file() or (
        ai / "element_relations_nli" / "onnx" / "onnx" / "model_quint8_avx2.onnx"
    ).is_file()
    assets.lama_inpainting_onnx = (ai / "inpainting" / "lama" / "model.onnx").is_file() or (
        ai / "inpainting" / "lama" / "lama_fp32.onnx").is_file()
    assets.embedded_fonts = (ai / "fonts" / "embedded").is_dir()
    return assets


def build_layout_policy_hint(contract: Any = None, normalized: dict | None = None,
                             project_root: str | Path | None = None) -> LayoutPolicyHint:
    assets = discover_layout_model_assets(project_root)
    hint = LayoutPolicyHint(assets=assets.to_dict())

    # Layout models present => we can be more confident that obstacles/regions
    # from PAGEPRINT are meaningful, so the deterministic solver may use slightly
    # stronger width expansion and obstacle avoidance.
    if assets.pp_doc_layout_plus_l or assets.pp_doc_block_layout:
        hint.confidence += 0.10
        hint.max_width_growth_pt = 120.0
        hint.avoid_visual_obstacles = True

    if assets.visual_smolvlm_500m:
        # A visual-layout model exists locally. We do not run it by default, but
        # the solver records this and uses a more conservative visual-page mode.
        hint.confidence += 0.05

    if assets.element_relations_nli_onnx:
        # Relationship model available: prefer grouping atomic lines into their
        # parent flow rather than treating every line as isolated.
        hint.group_atomic_lines = True
        hint.confidence += 0.05

    if assets.embedded_fonts:
        # Embedded fonts improve text measurement fidelity.
        hint.confidence += 0.05

    pi = (normalized or {}).get("page_intelligence") or {}
    layout_type = str(pi.get("layout_type") or pi.get("page_layout") or "").lower()
    page_role = str(((normalized or {}).get("page") or {}).get("page_role") or "").lower()
    if "image" in layout_type or page_role in {"cover", "image_dominant"}:
        hint.max_width_growth_pt = min(hint.max_width_growth_pt, 60.0)
        hint.prefer_width_growth_before_vertical_push = False
        hint.para_gap_pt = 2.0

    hint.confidence = min(0.95, hint.confidence)
    return hint
