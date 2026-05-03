"""
P2 — Agent de structuration hiérarchique des blocs.

Analyse les relations entre blocs d'une page pour déterminer :
- La hiérarchie (bloc parent/enfant, sections/sous-sections)
- Le regroupement de blocs appartenant à la même unité logique
- La détection de blocs flottants (encadrés, call-outs, annotations)
"""

from __future__ import annotations

import json
from typing import Any

from .base import ModelRuntime, PipelineAgent, _extract_json


_SYSTEM_PROMPT = """\
You are a document structure expert. Return JSON only.
Output schema: {"groups":[...],"floating":[...],"hierarchy":[...]}
- groups: [[block_id,...]] — blocks that belong to the same logical unit
- floating: [block_id,...] — sidebar, callout, annotation blocks
- hierarchy: [{"parent":block_id,"children":[block_id,...]}]

Rules:
- Group blocks when they share the same caption, form a list, or are visually connected
- Mark as floating when a block is visually separated from the main text flow
- Build hierarchy when headings introduce body paragraphs

If unsure: {"groups":[],"floating":[],"hierarchy":[]}"""


_FEW_SHOT: list[tuple[str, str]] = [
    (
        '{"blocks":[{"id":"b0","role":"heading","text":"Introduction"},{"id":"b1","role":"body","text":"This chapter covers..."},{"id":"b2","role":"caption","text":"Figure 1.1 Overview"}]}',
        '{"groups":[],"floating":[],"hierarchy":[{"parent":"b0","children":["b1"]}]}',
    ),
    (
        '{"blocks":[{"id":"b0","role":"body","text":"Step 1: Initialize"},{"id":"b1","role":"body","text":"Step 2: Iterate"},{"id":"b2","role":"body","text":"Step 3: Converge"},{"id":"b3","role":"caption","text":"Figure 2.3 Algorithm steps"}]}',
        '{"groups":[["b0","b1","b2"]],"floating":[],"hierarchy":[]}',
    ),
]

_MAX_FEW_SHOT = 2
_MAX_BLOCKS = 12


class P2StructureAgent(PipelineAgent):
    """
    Agent P2 : structuration hiérarchique de la page.

    Entrée (``input_data``) :
    ```json
    {
      "blocks": [
        {"id": "b0", "role": "heading", "text": "...", "bbox": [...]}
      ]
    }
    ```

    Sortie :
    ```json
    {
      "groups": [["b0", "b1"]],
      "floating": ["b5"],
      "hierarchy": [{"parent": "b0", "children": ["b1", "b2"]}]
    }
    ```
    """

    stage = "p2_structure"
    prompt_version = "v1"
    default_max_new_tokens = 256

    def build_messages(self, input_data: dict) -> list[dict]:
        # Tronquer à MAX_BLOCKS blocs pour le prompt
        truncated = dict(input_data)
        blocks = (truncated.get("blocks") or [])[:_MAX_BLOCKS]
        # Simplifier chaque bloc pour le prompt
        simplified_blocks = []
        for b in blocks:
            simplified_blocks.append({
                "id": str(b.get("id") or b.get("block_id") or ""),
                "role": str(b.get("role") or "body"),
                "text": str(b.get("text") or b.get("line_text") or "")[:120],
                "bbox": b.get("bbox"),
            })
        truncated["blocks"] = simplified_blocks
        block_json = json.dumps(truncated, ensure_ascii=False, separators=(",", ":"))

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user_ex, asst_ex in _FEW_SHOT[:_MAX_FEW_SHOT]:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": asst_ex})
        messages.append({"role": "user", "content": block_json})
        return messages

    def parse_response(self, raw: str, input_data: dict) -> dict:
        empty: dict = {"groups": [], "floating": [], "hierarchy": []}
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return empty

        # groups — liste de listes d'ids
        raw_groups = parsed.get("groups") or []
        valid_groups: list[list[str]] = []
        for grp in raw_groups:
            if isinstance(grp, list) and len(grp) >= 2:
                ids = [str(x) for x in grp if str(x or "").strip()]
                if len(ids) >= 2:
                    valid_groups.append(ids)

        # floating
        raw_floating = parsed.get("floating") or []
        valid_floating = [str(x) for x in raw_floating if str(x or "").strip()]

        # hierarchy
        raw_hier = parsed.get("hierarchy") or []
        valid_hier: list[dict] = []
        for item in raw_hier:
            if not isinstance(item, dict):
                continue
            parent = str(item.get("parent") or "").strip()
            children = [str(c) for c in (item.get("children") or []) if str(c or "").strip()]
            if parent and children:
                valid_hier.append({"parent": parent, "children": children})

        return {"groups": valid_groups, "floating": valid_floating, "hierarchy": valid_hier}
