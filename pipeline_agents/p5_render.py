"""
P5 — Agent de sélection de stratégie de rendu.

Détermine la stratégie de reconstruction optimale pour un bloc traduit :
- prose_reflow : reflow du texte en continu dans la bbox
- label_stack : empilement ligne par ligne avec positionnement
- heading_reflow : titre avec centrage/alignement adapté
- caption_reflow : légende d'image/tableau
- bitmap_preserve : bloc non modifiable (images, formules)
- fixed_preserve : positionnement fixe absolu

C'est l'agent le plus impactant du pipeline : il pilote `reconstructor.py`.
"""

from __future__ import annotations

import json
from typing import Any

from .base import ModelRuntime, PipelineAgent, _extract_json


_SYSTEM_PROMPT = """\
You are a document rendering expert. Return JSON only.
Output schema: {"strategy":"...","confidence":0.0,"params":{...},"reason":"..."}
- strategy: prose_reflow | label_stack | heading_reflow | caption_reflow | bitmap_preserve | fixed_preserve
- confidence: float 0.0–1.0
- params: strategy-specific rendering parameters (optional)
- reason: short explanation (max 20 words)

Strategy selection rules:
- prose_reflow: body paragraphs, ≥2 lines, avg words/line ≥ 5, translated text flows naturally
- label_stack: short lines (≤4 words avg), column shapes, form labels, list items
- heading_reflow: role=heading or role=title, may need centering
- caption_reflow: role=caption, follows a figure or table
- bitmap_preserve: role=formula, role=image, untranslatable content
- fixed_preserve: role=table_cell, page numbers, anchored annotations

Params for prose_reflow: {"justify":true|false,"max_lines":null|int}
Params for label_stack: {"align":"left|center|right","line_spacing":1.0}
Params for heading_reflow: {"align":"left|center","bold":true|false}

If unsure: {"strategy":"prose_reflow","confidence":0.4,"params":{},"reason":"default"}"""


_FEW_SHOT: list[tuple[str, str]] = [
    (
        '{"role":"body","line_count":5,"avg_words_per_line":8.5,"is_column_shape":false,"has_translation":true,"text_sample":"Le réseau apprend des représentations hiérarchiques de caractéristiques à partir des données brutes."}',
        '{"strategy":"prose_reflow","confidence":0.93,"params":{"justify":true},"reason":"continuous prose, 5 lines, 8.5 words avg"}',
    ),
    (
        '{"role":"body","line_count":4,"avg_words_per_line":2.8,"is_column_shape":true,"has_translation":true,"text_sample":"Nom:\\nPrénom:\\nDate:\\nSignature:"}',
        '{"strategy":"label_stack","confidence":0.91,"params":{"align":"left","line_spacing":1.2},"reason":"short label lines, column shape"}',
    ),
    (
        '{"role":"heading","line_count":1,"avg_words_per_line":4,"is_column_shape":false,"has_translation":true,"text_sample":"Chapitre 3 — Résultats expérimentaux"}',
        '{"strategy":"heading_reflow","confidence":0.95,"params":{"align":"left","bold":true},"reason":"heading role, single line"}',
    ),
    (
        '{"role":"caption","line_count":2,"avg_words_per_line":5,"is_column_shape":false,"has_translation":true,"text_sample":"Figure 3.4\\nComparaison des architectures CNN"}',
        '{"strategy":"caption_reflow","confidence":0.90,"params":{},"reason":"caption role"}',
    ),
    (
        '{"role":"formula","line_count":3,"avg_words_per_line":2,"is_column_shape":false,"has_translation":false,"text_sample":"dE/dw = ..."}',
        '{"strategy":"bitmap_preserve","confidence":0.97,"params":{},"reason":"formula role, not translatable"}',
    ),
]

_MAX_FEW_SHOT = 4

_VALID_STRATEGIES = {
    "prose_reflow",
    "label_stack",
    "heading_reflow",
    "caption_reflow",
    "bitmap_preserve",
    "fixed_preserve",
}


class P5RenderAgent(PipelineAgent):
    """
    Agent P5 : sélection de la stratégie de rendu pour la reconstruction.

    Entrée (``input_data``) :
    ```json
    {
      "role": "body",
      "line_count": 4,
      "avg_words_per_line": 7.2,
      "is_column_shape": false,
      "has_translation": true,
      "text_sample": "Le réseau apprend..."
    }
    ```

    Sortie :
    ```json
    {
      "strategy": "prose_reflow",
      "confidence": 0.9,
      "params": {"justify": true},
      "reason": "continuous prose"
    }
    ```
    """

    stage = "p5_render"
    prompt_version = "v1"
    default_max_new_tokens = 120

    def build_messages(self, input_data: dict) -> list[dict]:
        block_json = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user_ex, asst_ex in _FEW_SHOT[:_MAX_FEW_SHOT]:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": asst_ex})
        messages.append({"role": "user", "content": block_json})
        return messages

    def parse_response(self, raw: str, input_data: dict) -> dict:
        default: dict = {
            "strategy": "prose_reflow",
            "confidence": 0.4,
            "params": {},
            "reason": "default",
        }
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return default

        strategy = str(parsed.get("strategy") or "").lower()
        if strategy not in _VALID_STRATEGIES:
            strategy = "prose_reflow"

        try:
            confidence = float(parsed.get("confidence") or 0.4)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.4

        params = parsed.get("params")
        if not isinstance(params, dict):
            params = {}

        reason = str(parsed.get("reason") or "")[:60]

        return {
            "strategy": strategy,
            "confidence": confidence,
            "params": params,
            "reason": reason,
        }

    @classmethod
    def build_input_from_block(cls, block: dict) -> dict:
        """Construit l'input de l'agent depuis un bloc pipeline."""
        lines = block.get("lines") or []
        line_count = len(lines)
        texts = [(ln.get("line_text") or ln.get("text") or "").strip() for ln in lines]
        texts = [t for t in texts if t]
        avg_words = sum(len(t.split()) for t in texts) / max(1, len(texts))

        # Détection colonne : bloc plus étroit que 40% de la page
        bbox = block.get("bbox") or [0, 0, 0, 0]
        block_w = max(1.0, float(bbox[2]) - float(bbox[0]))
        # On ne peut pas connaître la largeur de page ici, on utilise un proxy
        page_w = float(block.get("_page_width") or 600)
        is_column_shape = block_w / page_w < 0.40

        has_translation = bool(
            block.get("translated_text")
            or any((ln.get("translated_text") or "") for ln in lines)
            or (block.get("semantic_payload") or {}).get("translated_text")
        )

        # Échantillon de texte traduit ou source
        text_sample = (
            str(block.get("translated_text") or "")
            or " ".join(texts[:3])
        )[:150]

        return {
            "role": str(block.get("role") or "body"),
            "line_count": line_count,
            "avg_words_per_line": round(avg_words, 1),
            "is_column_shape": is_column_shape,
            "has_translation": has_translation,
            "text_sample": text_sample,
        }

    @classmethod
    def score_block(cls, block: dict) -> float:
        """Score [0,1] indiquant si le bloc mérite une sélection IA de stratégie."""
        role = str(block.get("role") or "body").lower()
        lines = block.get("lines") or []
        line_count = len(lines)
        # Blocs simples → pas besoin d'IA
        if role in {"formula", "image", "page_number"}:
            return 0.0
        if line_count == 0:
            return 0.0
        # Ambiguïté structurelle → appel IA justifié
        score = 0.0
        texts = [(ln.get("line_text") or ln.get("text") or "").strip() for ln in lines if ln]
        avg_words = sum(len(t.split()) for t in texts) / max(1, len(texts))
        if 2 <= avg_words <= 6:
            score += 0.4  # zone grise entre prose et label
        if role in {"body", "paragraph"} and line_count <= 2:
            score += 0.3  # bloc court — heading ou prose ?
        if block.get("semantic_groups") and role in {"body"}:
            score += 0.3  # groupes sémantiques présents
        return min(1.0, score)
