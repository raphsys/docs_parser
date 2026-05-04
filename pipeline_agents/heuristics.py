"""
Estimateurs heuristiques pour P1, P3, P5 et P6 — sans modèle LLM.

Chaque estimateur reproduit la logique de l'agent LLM correspondant via des
règles déterministes basées sur les features du bloc : rôle, comptage de mots,
ratios de longueur, patterns de texte, confiance d'inpainting, etc.

Avantages : < 1 ms par bloc, 0 dépendance externe, disponible même sans GPU.
Limites : ne gère pas les cas ambigus autant qu'un LLM fin-tuné.

Pour brancher un modèle discriminatif à la place d'un estimateur, sous-classer
la classe correspondante et implémenter `estimate(input_data) -> dict`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Utilitaires communs
# ---------------------------------------------------------------------------

_LIST_START = re.compile(r"^(?:\d+[\.\)]\s|[•·▪▫◦○●\-–]\s|Step\s+\d+[:\.])")
_CODE_MATH = re.compile(r"[=\+\-\*/\\^{}()<>\[\]]{2,}|def\s+\w+|import\s+\w+|^\s*\w+\s*=\s*\w+")
_CAPTION_START = re.compile(
    r"^(?:Figure|Fig\.|Figures|Table|Tab\.|Tableau|Listing|Algorithm|Algo\.)\s*\d+",
    re.IGNORECASE,
)
_HEADING_CHAPTER = re.compile(
    r"^(?:Chapter|Section|Part|Unit|Module)\s+\d+|"
    r"^(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.\s+\S",
    re.IGNORECASE,
)
_SECTION_NUM = re.compile(r"^\d+(?:\.\d+)*\s+\S")
_HYPHEN_BREAK = re.compile(r"\b(\w{3,})-$")


# ---------------------------------------------------------------------------
# P1 — Extraction et segmentation structurelle
# ---------------------------------------------------------------------------

class P1HeuristicEstimator:
    """
    Estime les corrections P1 sans LLM.

    Détecte :
    - Légendes atomiques (Figure/Table multi-ligne)
    - Titres de chapitre/section
    - Formules et lignes techniques
    - Lignes à ignorer (numéros de page, séparateurs)
    - Jointures de mots coupés par césure

    Sortie : même schéma que P1ExtractionAgent.parse_response()
    """

    _FORMULA_CHARS = frozenset("=∑∫√±×÷∞∂∇∆∈∉⊂⊃∪∩∧∨¬∀∃→⟹↔≤≥≠≈∝|‖·{}\\")
    _SKIP_ONLY_DIGITS = re.compile(r"^[\d\s\-–—·.]+$")

    def is_available(self) -> bool:
        return True

    def estimate(self, input_data: dict) -> dict:
        lines = input_data.get("lines") or []
        role = str(input_data.get("role") or "body").lower()
        empty: dict = {"c": [], "sb": [], "lb": [], "lm": None, "hj": []}

        if not lines:
            return empty

        texts = [
            str(ln.get("t") or ln.get("line_text") or ln.get("text") or "").strip()
            for ln in lines
        ]

        result: dict = {"c": [], "sb": [], "lb": [], "lm": None, "hj": []}

        # Césures
        for i, text in enumerate(texts[:-1]):
            m = _HYPHEN_BREAK.search(text)
            if m and i + 1 < len(texts):
                nxt = texts[i + 1]
                # Continuation probable si la ligne suivante commence en minuscule
                m2 = re.match(r"^([a-zà-ÿ]\w*)", nxt)
                if m2:
                    full = m.group(1) + m2.group(1)
                    result["hj"].append({"i": i, "w": full})

        # Légende atomique (Figure N … multi-ligne)
        if len(texts) >= 2 and _CAPTION_START.match(texts[0]):
            result["c"] = [{"li": list(range(len(texts))), "a": "atomic"}]
            result["lm"] = self._layout_mode(texts, role)
            return result

        # Corrections ligne par ligne
        corrections: list[dict] = []
        for i, text in enumerate(texts):
            action = self._classify(text, i, texts, role)
            if action and action != "keep":
                corrections.append({"li": [i], "a": action})

        result["c"] = corrections
        result["lm"] = self._layout_mode(texts, role)
        return result

    def _classify(self, text: str, idx: int, all_texts: list[str], role: str) -> str | None:
        if not text:
            return "skip"

        # Numéro de page / séparateur
        if len(text) <= 3 or (self._SKIP_ONLY_DIGITS.match(text) and len(text) <= 8):
            return "skip"

        # Formule
        if role == "formula":
            return None
        formula_ratio = sum(1 for c in text if c in self._FORMULA_CHARS) / max(1, len(text))
        if formula_ratio > 0.28:
            return "formula"

        words = text.split()

        # Titre
        if len(words) <= 8:
            if _HEADING_CHAPTER.match(text) or _SECTION_NUM.match(text):
                return "heading"
            if text.isupper() and len(words) >= 2 and not any(c in text for c in "=+*/\\"):
                return "heading"
            # Ligne initiale en Title Case courte (≤ 6 mots) sur un bloc body
            if idx == 0 and role == "body" and len(words) <= 6 and text.istitle():
                return "heading"

        return None

    def _layout_mode(self, texts: list[str], role: str) -> dict | None:
        if not texts:
            return None

        if role in {"formula", "code"}:
            breaks = [{"i": i, "after": "fixed_line"} for i in range(len(texts))]
            return {"line_flow": "fixed_lines", "breaks": breaks}

        list_n = sum(1 for t in texts if _LIST_START.match(t))
        code_n = sum(1 for t in texts if _CODE_MATH.search(t))
        word_counts = [len(t.split()) for t in texts if t]
        avg_w = sum(word_counts) / max(1, len(word_counts))

        if code_n / max(1, len(texts)) > 0.4:
            breaks = [{"i": i, "after": "fixed_line"} for i in range(len(texts))]
            return {"line_flow": "fixed_lines", "breaks": breaks}

        if list_n / max(1, len(texts)) > 0.35 or avg_w <= 3:
            breaks = [{"i": i, "after": "new_line"} for i in range(len(texts) - 1)]
            breaks.append({"i": len(texts) - 1, "after": "block_end"})
            return {"line_flow": "preserve_line_breaks", "breaks": breaks}

        if avg_w >= 5:
            breaks = [{"i": i, "after": "soft_wrap"} for i in range(len(texts) - 1)]
            breaks.append({"i": len(texts) - 1, "after": "block_end"})
            return {"line_flow": "inline_reflow", "breaks": breaks}

        return None


# ---------------------------------------------------------------------------
# P3 — Mode de layout
# ---------------------------------------------------------------------------

class P3HeuristicEstimator:
    """
    Détermine le mode de layout d'un bloc sans LLM.

    Règles de décision basées sur : rôle, avg_words_per_line, avg_line_width,
    has_indent, patterns liste/code dans les lignes.

    Sortie : même schéma que P3LayoutAgent.parse_response()
    """

    _VALID_MODES = {
        "inline_reflow",
        "preserve_line_breaks",
        "preserve_paragraphs",
        "fixed_lines",
        "column_layout",
    }

    def is_available(self) -> bool:
        return True

    def estimate(self, input_data: dict) -> dict:
        role = str(input_data.get("role") or "body").lower()
        avg_words = float(input_data.get("avg_words_per_line") or 5.0)
        has_indent = bool(input_data.get("has_indent") or False)
        lines = [str(ln) for ln in (input_data.get("lines") or [])]

        # Rôles avec disposition fixe
        if role in {"formula", "code", "table", "image"}:
            return {"layout_mode": "fixed_lines", "confidence": 0.95, "notes": f"role={role}"}

        list_n = sum(1 for ln in lines if _LIST_START.match(ln))
        code_n = sum(1 for ln in lines if _CODE_MATH.search(ln))
        total = max(1, len(lines))

        if code_n / total > 0.40:
            return {"layout_mode": "fixed_lines", "confidence": 0.88, "notes": "code/math patterns"}

        if list_n / total > 0.40:
            return {"layout_mode": "preserve_line_breaks", "confidence": 0.90, "notes": "list/step patterns"}

        if avg_words >= 6.5:
            conf = round(min(0.95, 0.72 + (avg_words - 6.5) * 0.025), 2)
            return {"layout_mode": "inline_reflow", "confidence": conf, "notes": f"avg {avg_words:.1f} w/line"}

        if avg_words <= 3.0:
            return {"layout_mode": "preserve_line_breaks", "confidence": 0.85, "notes": f"short lines avg {avg_words:.1f}w"}

        if has_indent:
            return {"layout_mode": "preserve_paragraphs", "confidence": 0.75, "notes": "indented paragraphs"}

        return {"layout_mode": "inline_reflow", "confidence": 0.60, "notes": "default"}


# ---------------------------------------------------------------------------
# P5 — Stratégie de rendu
# ---------------------------------------------------------------------------

class P5HeuristicEstimator:
    """
    Sélectionne la stratégie de rendu sans LLM.

    Priorité : rôle du bloc > avg_words_per_line + is_column_shape.

    Sortie : même schéma que P5RenderAgent.parse_response()
    """

    def is_available(self) -> bool:
        return True

    def estimate(self, input_data: dict) -> dict:
        role = str(input_data.get("role") or "body").lower()
        line_count = int(input_data.get("line_count") or 1)
        avg_words = float(input_data.get("avg_words_per_line") or 5.0)
        is_column = bool(input_data.get("is_column_shape") or False)

        # Rôles à stratégie fixe
        if role in {"formula", "image", "figure"}:
            return {"strategy": "bitmap_preserve", "confidence": 0.97, "params": {}, "reason": f"role={role}"}

        if role in {"page_number", "table_cell", "header", "footer"}:
            return {"strategy": "fixed_preserve", "confidence": 0.95, "params": {}, "reason": f"role={role}"}

        if role in {"heading", "title", "section"}:
            return {
                "strategy": "heading_reflow",
                "confidence": 0.92,
                "params": {"align": "left", "bold": True},
                "reason": "heading role",
            }

        if role in {"caption", "footnote"}:
            return {"strategy": "caption_reflow", "confidence": 0.90, "params": {}, "reason": "caption role"}

        # Décision par contenu
        if is_column or avg_words <= 4:
            return {
                "strategy": "label_stack",
                "confidence": 0.82,
                "params": {"align": "left", "line_spacing": 1.0},
                "reason": f"short lines avg {avg_words:.1f}w" + (" column" if is_column else ""),
            }

        if avg_words >= 5 and line_count >= 2:
            justify = line_count >= 3
            return {
                "strategy": "prose_reflow",
                "confidence": 0.85,
                "params": {"justify": justify},
                "reason": f"prose {line_count}L avg {avg_words:.1f}w",
            }

        # Ligne unique
        if line_count == 1:
            if avg_words <= 5:
                return {"strategy": "label_stack", "confidence": 0.70, "params": {"align": "left", "line_spacing": 1.0}, "reason": "single short line"}
            return {"strategy": "prose_reflow", "confidence": 0.65, "params": {"justify": False}, "reason": "single long line"}

        return {"strategy": "prose_reflow", "confidence": 0.50, "params": {}, "reason": "default"}


# ---------------------------------------------------------------------------
# P6 — Audit background
# ---------------------------------------------------------------------------

class P6HeuristicEstimator:
    """
    Évalue la qualité du background après inpainting sans LLM.

    Règles basées sur : avg_confidence, coverage_ratio, blocks_removed.
    Si avg_confidence < 0.60, les régions sont signalées pour re-inpainting.

    Sortie : même schéma que P6BackgroundAgent.parse_response()
    """

    def is_available(self) -> bool:
        return True

    def estimate(self, input_data: dict) -> dict:
        avg_conf = float(input_data.get("avg_confidence") or 0.90)
        coverage = float(input_data.get("coverage_ratio") or 0.0)
        blocks_removed = int(input_data.get("blocks_removed") or 0)
        regions = list(input_data.get("inpaint_regions") or [])

        quality = avg_conf

        # Pénalité : beaucoup de blocs retirés avec faible confiance
        if blocks_removed > 5 and avg_conf < 0.70:
            quality = max(quality - 0.10, 0.0)

        # Pénalité légère si coverage très élevé
        if coverage > 0.85:
            quality = max(quality - 0.05, 0.0)

        artifacts: list[dict] = []
        reprocess: list[list[float]] = []

        if avg_conf < 0.60:
            severity = "high" if avg_conf < 0.45 else "medium"
            for reg in regions:
                if isinstance(reg, list) and len(reg) == 4:
                    try:
                        freg = [float(x) for x in reg]
                    except (TypeError, ValueError):
                        continue
                    artifacts.append({"type": "text_residue", "severity": severity, "region": freg})
                    reprocess.append(freg)

        ok = quality >= 0.85 and not any(a["severity"] == "high" for a in artifacts)

        return {
            "quality": round(max(0.0, min(1.0, quality)), 3),
            "artifacts": artifacts,
            "reprocess": reprocess,
            "ok": ok,
        }
