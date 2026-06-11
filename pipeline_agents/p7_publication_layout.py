"""
P7 — Agent IA de mise en page finale publication-ready.

Cet agent travaille au niveau page complète, après extraction/traduction et
avant l'écriture finale. Il ne choisit pas une stratégie de bloc: il propose
des placements globaux en respectant les zones fixes (formules, images,
tableaux, overlays) et l'ordre de lecture.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .base import ModelRuntime, PipelineAgent, _extract_json

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a publication-ready document layout engine. Return JSON only.
You receive one PDF page represented as fixed regions and translated text blocks.
Your job is to propose final block placements close to the original page.
The input may include visual_faults. Fix these first.
The input may also include visual_model_placements produced by a visual document
model. Treat them as suggestions only: validate them against all hard rules.

Hard rules:
- preserve every fixed region: formulas, equations, figures, tables, images;
- never place text over fixed regions;
- never overlap text blocks;
- preserve reading order top-to-bottom and column order;
- never remove text;
- never shrink font or line spacing;
- keep placements close to source unless needed to avoid collisions;
- use the same page bounds and margins.
- if visual_faults mention glued words, give that text block more room or move it to a clean slot;
- prefer a visual_model_placement when it fixes a blocking visual fault and does
  not create any overlap or reading-order regression;
- if no safe fix exists inside the current page, say so in warnings instead of guessing.

Output schema:
{
  "confidence": 0.0,
  "placements": [
    {"block_id":"...", "bbox":[x0,y0,x1,y1], "reason":"..."}
  ],
  "reading_order": ["block_id", "..."],
  "warnings": ["..."]
}

Coordinates are PDF points, not pixels. Return only placements you want changed.
If no safe publication-ready layout exists, return no placement and explain why
in warnings.
"""


class P7PublicationLayoutAgent(PipelineAgent):
    """
    Agent P7 : composition globale de page.

    Entrée attendue:
    ```json
    {
      "page": {"width": 595, "height": 842, "margin": 6},
      "fixed_regions": [{"id": "...", "type": "formula", "bbox": [...]}],
      "visual_faults": [{"type":"text_fixed_region_collision", "block_id":"...", "bbox":[...]}],
      "text_blocks": [{
        "block_id": "...",
        "source_bbox": [...],
        "current_bbox": [...],
        "text": "...",
        "role": "body",
        "reading_order": 12
      }]
    }
    ```
    """

    stage = "p7_publication_layout"
    prompt_version = "v1"
    default_max_new_tokens = 260

    def __init__(self, runtime: ModelRuntime) -> None:
        super().__init__(runtime)
        self._backend = os.environ.get("PIPELINE_AGENT_P7_BACKEND", "llm").lower()
        try:
            self.default_max_new_tokens = max(80, min(700, int(os.environ.get("PIPELINE_AGENT_P7_MAX_NEW_TOKENS", "260"))))
        except (TypeError, ValueError):
            self.default_max_new_tokens = 260

    def is_available(self) -> bool:
        if self._backend != "llm":
            return False
        return self.runtime.is_available()

    def build_messages(self, input_data: dict) -> list[dict]:
        payload = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]

    def parse_response(self, raw: str, input_data: dict) -> dict:
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return {"confidence": 0.0, "placements": [], "reading_order": [], "warnings": ["invalid_json"]}

        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0

        page = dict((input_data or {}).get("page") or {})
        page_w = float(page.get("width") or 0.0)
        page_h = float(page.get("height") or 0.0)
        valid_ids = {
            str((block or {}).get("block_id") or "")
            for block in (input_data or {}).get("text_blocks") or []
            if isinstance(block, dict)
        }

        placements = []
        for item in parsed.get("placements") or []:
            if not isinstance(item, dict):
                continue
            block_id = str(item.get("block_id") or "").strip()
            bbox = item.get("bbox")
            if block_id not in valid_ids or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            if page_w > 0 and (x0 < -2.0 or x1 > page_w + 2.0):
                continue
            if page_h > 0 and (y0 < -2.0 or y1 > page_h + 2.0):
                continue
            placements.append(
                {
                    "block_id": block_id,
                    "bbox": [x0, y0, x1, y1],
                    "reason": str(item.get("reason") or "")[:100],
                }
            )

        reading_order = [
            str(block_id)
            for block_id in parsed.get("reading_order") or []
            if str(block_id) in valid_ids
        ]
        warnings = [str(w)[:140] for w in parsed.get("warnings") or [] if str(w).strip()]

        return {
            "confidence": confidence,
            "placements": placements,
            "reading_order": reading_order,
            "warnings": warnings,
        }
