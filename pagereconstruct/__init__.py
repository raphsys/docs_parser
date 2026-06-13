"""PAGERECONSTRUCT — compile a translated page into controlled render operations.

Pass 1 (this milestone): plan compilation only, no PDF rendering.
"""

from __future__ import annotations

from .errors import (
    PageReconstructError,
    PageReconstructInputError,
    PageRenderBackendError,
    PageRenderPlanError,
)
from .input_adapter import PageReconstructInputAdapter
from .plan_compiler import PageRenderPlanCompiler, choose_renderer, compile_page_render_plan
from .protected_region_index import ProtectedRegionIndex, build_protected_region_index
from .schema import PageRenderPlan, PreservedUnit, ProtectedRegion, TranslatedTextUnit

__all__ = [
    "PageReconstructError", "PageReconstructInputError", "PageRenderPlanError", "PageRenderBackendError",
    "PageReconstructInputAdapter", "PageRenderPlanCompiler", "compile_page_render_plan", "choose_renderer",
    "ProtectedRegionIndex", "build_protected_region_index",
    "PageRenderPlan", "TranslatedTextUnit", "PreservedUnit", "ProtectedRegion",
]
