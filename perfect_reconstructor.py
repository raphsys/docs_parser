"""Perfect-extraction aware experimental reconstructor.

This subclass keeps the existing DocumentReconstructor implementation and only
changes dispatch decisions for pages explicitly produced by
PerfectExtractionReconstructionAdapter.
"""

from __future__ import annotations

from typing import Any

from reconstructor import DocumentReconstructor


class PerfectAwareDocumentReconstructor(DocumentReconstructor):
    """Existing reconstructor with stricter consumption of perfect_extraction."""

    def _is_perfect_page(self, page_data: dict | None) -> bool:
        return isinstance(page_data, dict) and isinstance(page_data.get("perfect_extraction_page"), dict)

    def _looks_like_toc_page(self, page_data):
        if self._is_perfect_page(page_data):
            # Perfect TOC rows must be supplied explicitly. The historical TOC
            # synthesizer groups native blocks too aggressively for perfect
            # extraction, so do not let it infer rows from large blocks.
            return False
        return super()._looks_like_toc_page(page_data)

    def _contract_render_mode_for_block(self, block, page_data=None, block_type=None, contract_entry=None):
        if self._is_perfect_page(page_data) and isinstance(block, dict):
            render_policy = str(block.get("render_policy") or "").strip().lower()
            if render_policy == "source_overlay":
                return super()._contract_render_mode_for_block(
                    block,
                    page_data=page_data,
                    block_type=block_type,
                    contract_entry=contract_entry,
                )
            lines = [line for line in (block.get("lines") or []) if isinstance(line, dict) and line.get("bbox")]
            if len(lines) >= 1:
                return "line_preserve"
        return super()._contract_render_mode_for_block(
            block,
            page_data=page_data,
            block_type=block_type,
            contract_entry=contract_entry,
        )

    def _reconstruction_contract_for_block(self, block, page_data=None):
        contract = dict(super()._reconstruction_contract_for_block(block, page_data=page_data) or {})
        if self._is_perfect_page(page_data) and isinstance(block, dict):
            if str(block.get("render_policy") or "").strip().lower() != "source_overlay" and block.get("lines"):
                contract["render_mode"] = "line_preserve"
                contract["strict_non_reflow"] = True
                contract["font_size_policy"] = "preserve_extracted"
                contract["style_policy"] = "preserve_extracted"
                contract["allow_expansion"] = False
                contract["min_font_ratio"] = 1.0
                contract["shrink_max_pt"] = 0.0
        return contract
