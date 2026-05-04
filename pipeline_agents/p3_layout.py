"""
P3 — Agent de classification du mode de layout.

Détermine pour chaque bloc le mode de mise en page optimal :
- inline_reflow : prose qui s'enroule naturellement
- preserve_line_breaks : listes, étapes, labels, titres
- preserve_paragraphs : blocs multi-paragraphes avec espacement
- fixed_lines : tables, formules, diagrammes
- column_layout : texte en colonnes multiples
"""

from __future__ import annotations

import json
import logging
import os

from .base import ModelRuntime, PipelineAgent, _extract_json
from .heuristics import P3HeuristicEstimator

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a document layout expert. Return JSON only.
Output schema: {"layout_mode":"...","confidence":0.0,"notes":"..."}
- layout_mode: one of inline_reflow | preserve_line_breaks | preserve_paragraphs | fixed_lines | column_layout
- confidence: float 0.0–1.0
- notes: short explanation (max 30 words)

Guidelines:
- inline_reflow: continuous prose, line breaks are soft wraps (default for body text)
- preserve_line_breaks: each line is a distinct unit (lists, labeled steps, heading+body)
- preserve_paragraphs: multiple paragraphs separated by blank lines or indentation
- fixed_lines: absolute x/y positions matter (tables, code, math, diagrams)
- column_layout: content is split into 2+ adjacent columns

If unsure: {"layout_mode":"inline_reflow","confidence":0.3,"notes":"default"}"""


_FEW_SHOT: list[tuple[str, str]] = [
    (
        '{"role":"body","line_count":4,"avg_line_width":0.9,"avg_words_per_line":9,"has_indent":false,"lines":["Neural networks learn hierarchical","feature representations from data,","using gradient descent to update","weights through backpropagation."]}',
        '{"layout_mode":"inline_reflow","confidence":0.92,"notes":"long lines, prose structure"}',
    ),
    (
        '{"role":"body","line_count":4,"avg_line_width":0.4,"avg_words_per_line":3,"has_indent":false,"lines":["Step 1: Load data","Step 2: Preprocess","Step 3: Train model","Step 4: Evaluate"]}',
        '{"layout_mode":"preserve_line_breaks","confidence":0.95,"notes":"numbered steps, each line is independent"}',
    ),
    (
        '{"role":"body","line_count":3,"avg_line_width":0.3,"avg_words_per_line":2,"has_indent":false,"lines":["x = 2.5","y = 3.1","z = x + y"]}',
        '{"layout_mode":"fixed_lines","confidence":0.88,"notes":"variable assignments, positional content"}',
    ),
]

_MAX_FEW_SHOT = 3


class P3LayoutAgent(PipelineAgent):
    """
    Agent P3 : sélection du mode de layout pour un bloc.

    Entrée (``input_data``) :
    ```json
    {
      "role": "body",
      "line_count": 4,
      "avg_line_width": 0.85,
      "avg_words_per_line": 8.2,
      "has_indent": false,
      "lines": ["sample line 1", "sample line 2"]
    }
    ```

    Sortie :
    ```json
    {
      "layout_mode": "inline_reflow",
      "confidence": 0.9,
      "notes": "continuous prose"
    }
    ```
    """

    stage = "p3_layout"
    prompt_version = "v1"
    default_max_new_tokens = 80

    _VALID_MODES = {
        "inline_reflow",
        "preserve_line_breaks",
        "preserve_paragraphs",
        "fixed_lines",
        "column_layout",
    }

    def __init__(self, runtime: ModelRuntime) -> None:
        super().__init__(runtime)
        self._heuristic = P3HeuristicEstimator()
        self._backend = os.environ.get("PIPELINE_AGENT_P3_BACKEND", "heuristic").lower()

    def is_available(self) -> bool:
        if self._backend == "llm":
            return self.runtime.is_available()
        return True

    def run(self, input_data: dict, *, use_cache: bool = True) -> dict:
        if self._backend == "llm":
            return super().run(input_data, use_cache=use_cache)
        try:
            return self._heuristic.estimate(input_data)
        except Exception as exc:
            logger.debug("[p3_layout/heuristic] échec: %s", exc)
            return {"layout_mode": "inline_reflow", "confidence": 0.3, "notes": "default"}

    def build_messages(self, input_data: dict) -> list[dict]:
        block_json = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user_ex, asst_ex in _FEW_SHOT[:_MAX_FEW_SHOT]:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": asst_ex})
        messages.append({"role": "user", "content": block_json})
        return messages

    def parse_response(self, raw: str, input_data: dict) -> dict:
        default = {"layout_mode": "inline_reflow", "confidence": 0.3, "notes": "default"}
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return default
        mode = str(parsed.get("layout_mode") or "").lower()
        if mode not in self._VALID_MODES:
            mode = "inline_reflow"
        try:
            confidence = float(parsed.get("confidence") or 0.3)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.3
        notes = str(parsed.get("notes") or "")[:60]
        return {"layout_mode": mode, "confidence": confidence, "notes": notes}

    @classmethod
    def build_input_from_block(cls, block: dict) -> dict:
        """Construit l'input de l'agent depuis un bloc pipeline."""
        lines = block.get("lines") or []
        texts = [(ln.get("line_text") or ln.get("text") or "").strip() for ln in lines]
        texts = [t for t in texts if t]
        line_count = len(texts)
        avg_words = sum(len(t.split()) for t in texts) / max(1, line_count)
        page_w = max(1.0, float((block.get("bbox") or [0, 0, 100, 100])[2]))
        block_w = max(1.0, float((block.get("bbox") or [0, 0, 100, 100])[2]) - float((block.get("bbox") or [0, 0, 0, 0])[0]))
        avg_line_width = round(block_w / page_w, 3)
        has_indent = any(
            float((ln.get("bbox") or [0])[0]) > float((block.get("bbox") or [0])[0]) + 5
            for ln in lines[:3]
            if isinstance(ln.get("bbox"), list)
        )
        return {
            "role": str(block.get("role") or "body"),
            "line_count": line_count,
            "avg_line_width": avg_line_width,
            "avg_words_per_line": round(avg_words, 1),
            "has_indent": has_indent,
            "lines": texts[:8],
        }
