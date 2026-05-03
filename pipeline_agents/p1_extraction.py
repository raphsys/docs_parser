"""
P1 — Agent d'extraction et de segmentation sémantique.

Cet agent analyse les blocs de texte extraits par OCR et détermine :
- Les lignes qui sont des titres (heading), formules, légendes atomiques, etc.
- Les frontières sémantiques intra-ligne (sb) et inter-lignes (lb)
- Le mode de mise en page (inline_reflow, preserve_line_breaks, …)
- Les jointures de mots coupés par césure (hj)

Il délègue au corrector existant (llm_semantic_corrector) pour la compatibilité
ascendante, mais peut aussi être utilisé directement.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import ModelRuntime, PipelineAgent, _extract_json


# ---------------------------------------------------------------------------
# Prompt système
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a text-structure expert. Return JSON only.
Output schema: {"c":[...],"sb":[...],"lb":[...],"lm":null|{...},"hj":[...]}
- c = corrections: [{"li":[line indices],"a":"heading|formula|skip|atomic|keep"}]
- sb = intra-line semantic boundaries: [{"i":line_index,"s":["segment 1","segment 2",...]}]
- lb = semantic breaks after full lines: [line_index,...]
- lm = layout mode: {"line_flow":"inline_reflow|preserve_line_breaks|preserve_paragraphs|fixed_lines","breaks":[{"i":line_index,"after":"soft_wrap|new_line|paragraph_break|block_end"}]}
- hj = hyphen joins: [{"i":line_index,"w":"full_word"}]

Rules:
- heading: section/chapter title not part of a paragraph (even mixed-case)
- formula: math, code, technical symbol sequence
- skip: page number, separator, isolated axis label
- atomic: multi-line caption (Figure/Table/Fig.) — keep ALL lines together
- keep: normal prose
- sb: one line contains multiple semantic units
- lb: semantic sentence boundary BETWEEN two lines (even without punctuation)
- lm: inline_reflow for soft-wrapped prose; preserve_line_breaks for lists/labels; fixed_lines for tables/math
- hj: line ends with a hyphenated word fragment

If unsure: {"c":[],"sb":[],"lb":[],"lm":null,"hj":[]}"""

_FEW_SHOT: list[tuple[str, str]] = [
    (
        '{"role":"body","lines":[{"i":0,"t":"Too-high vs. too-low learning rate Setting the learning"},{"i":1,"t":"rate high or low is a trade-off between speed and accuracy."}]}',
        '{"c":[{"li":[0],"a":"heading"},{"li":[1],"a":"keep"}],"sb":[],"lb":[],"lm":{"line_flow":"preserve_line_breaks","breaks":[{"i":0,"after":"new_line"},{"i":1,"after":"block_end"}]},"hj":[]}',
    ),
    (
        '{"role":"body","lines":[{"i":0,"t":"Figure 4.17"},{"i":1,"t":"A learning rate larger than the"},{"i":2,"t":"ideal lr value: the optimizer overshoots."}]}',
        '{"c":[{"li":[0,1,2],"a":"atomic"}],"sb":[],"lb":[],"lm":null,"hj":[]}',
    ),
    (
        '{"role":"body","lines":[{"i":0,"t":"Concatenates together the"},{"i":1,"t":"depth of the three filters."}]}',
        '{"c":[],"sb":[],"lb":[],"lm":{"line_flow":"inline_reflow","breaks":[{"i":0,"after":"soft_wrap"},{"i":1,"after":"block_end"}]},"hj":[]}',
    ),
    (
        '{"role":"body","lines":[{"i":0,"t":"The algorithm is guaran-"},{"i":1,"t":"teed to converge eventually."}]}',
        '{"c":[],"sb":[],"lb":[],"lm":null,"hj":[{"i":0,"w":"guaranteed"}]}',
    ),
]

_MAX_FEW_SHOT = 4
_MAX_LINES = 12


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class P1ExtractionAgent(PipelineAgent):
    """
    Agent P1 : classification structurelle des blocs OCR.

    Entrée (``input_data``) :
    ```json
    {
      "role": "body",
      "lines": [{"i": 0, "t": "...", "bb": [x0, y0, x1, y1]}, ...],
      "hs": [...]   // optionnel : phrases heuristiques provisoires
    }
    ```

    Sortie :
    ```json
    {
      "c":  [{"li": [0], "a": "heading|formula|skip|atomic|keep"}],
      "sb": [{"i": 0, "s": ["seg1", "seg2"]}],
      "lb": [0, 1],
      "lm": {"line_flow": "inline_reflow", "breaks": [...]},
      "hj": [{"i": 0, "w": "guaranteed"}]
    }
    ```
    """

    stage = "p1_extraction"
    prompt_version = "v1"
    default_max_new_tokens = 200

    def build_messages(self, input_data: dict) -> list[dict]:
        block_json = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user_ex, asst_ex in _FEW_SHOT[:_MAX_FEW_SHOT]:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": asst_ex})
        messages.append({"role": "user", "content": block_json})
        return messages

    def parse_response(self, raw: str, input_data: dict) -> dict:
        empty: dict = {"c": [], "sb": [], "lb": [], "lm": None, "hj": []}
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return empty

        # --- corrections ---
        valid_c: list[dict] = []
        valid_actions = {"heading", "formula", "skip", "atomic", "keep"}
        for corr in (parsed.get("c") or []):
            if not isinstance(corr, dict):
                continue
            action = str(corr.get("a") or "keep")
            if action not in valid_actions:
                continue
            li = corr.get("li")
            if not isinstance(li, list) or not li:
                continue
            try:
                valid_c.append({"li": [int(x) for x in li], "a": action})
            except (ValueError, TypeError):
                continue
        # rejeter si toutes les lignes sont "skip"
        if valid_c and all(c["a"] == "skip" for c in valid_c):
            valid_c = []

        # --- intra-line boundaries ---
        valid_sb: list[dict] = []
        for split in (parsed.get("sb") or []):
            if not isinstance(split, dict):
                continue
            try:
                idx = int(split.get("i", -1))
            except (ValueError, TypeError):
                continue
            segs = [str(s or "").strip() for s in (split.get("s") or []) if str(s or "").strip()]
            if idx >= 0 and len(segs) >= 2:
                valid_sb.append({"i": idx, "s": segs})

        # --- inter-line boundaries ---
        valid_lb: list[int] = sorted({
            int(x) for x in (parsed.get("lb") or [])
            if str(x).lstrip("-").isdigit() and int(x) >= 0
        })

        # --- layout mode ---
        lm_raw = parsed.get("lm")
        valid_lm = None
        if isinstance(lm_raw, dict):
            line_flow = str(lm_raw.get("line_flow") or "").lower()
            if line_flow in {"inline_reflow", "preserve_line_breaks", "preserve_paragraphs", "fixed_lines"}:
                breaks: list[dict] = []
                for b in (lm_raw.get("breaks") or []):
                    if not isinstance(b, dict):
                        continue
                    try:
                        bi = int(b.get("i", -1))
                    except (TypeError, ValueError):
                        continue
                    after = str(b.get("after") or "").lower()
                    if bi >= 0 and after in {"soft_wrap", "new_line", "paragraph_break", "list_item", "fixed_line", "block_end"}:
                        breaks.append({"i": bi, "after": after})
                valid_lm = {"line_flow": line_flow, "breaks": breaks}

        # --- hyphen joins ---
        valid_hj: list[dict] = []
        for hj in (parsed.get("hj") or []):
            if not isinstance(hj, dict):
                continue
            try:
                hi = int(hj.get("i", -1))
            except (TypeError, ValueError):
                continue
            word = str(hj.get("w") or "").strip()
            if hi >= 0 and word:
                valid_hj.append({"i": hi, "w": word})

        return {"c": valid_c, "sb": valid_sb, "lb": valid_lb, "lm": valid_lm, "hj": valid_hj}

    @classmethod
    def score_block(cls, block: dict) -> float:
        """Score [0,1] indiquant si le bloc mérite un appel LLM."""
        lines = block.get("lines") or []
        sps = block.get("semantic_phrases") or []
        if not lines:
            return 0.0
        score = 0.0
        line_texts = [(ln.get("line_text") or ln.get("t") or ln.get("text") or "").strip() for ln in lines]
        # Césures
        score += 0.20 * sum(1 for t in line_texts if t.endswith("-"))
        # Lignes très courtes (labels/axes)
        score += 0.30 * sum(1 for t in line_texts if 0 < len(t) <= 3)
        # Phrases multi-lignes non terminées
        for sp in sps:
            if sp.get("sentence_end_reason") == "eof" and len(str(sp.get("text") or "").split()) >= 4:
                if not re.search(r'[.!?]["\')]*$', str(sp.get("text") or "")):
                    score += 0.30
        # Légendes
        if any(re.match(r'^(Figure|Fig\.|Table|Tab\.)\s+\d', t, re.I) for t in line_texts):
            score += 0.25
        return min(1.0, score)
