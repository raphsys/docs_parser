"""
P4 — Agent de validation et post-édition de la traduction.

Analyse la qualité d'un bloc traduit et propose :
- Un score de qualité [0,1]
- Les problèmes détectés (terminologie, cohérence, sens)
- Une version corrigée si nécessaire (post-édition légère)
- Le marquage des segments non traduits

Backends disponibles (variable PIPELINE_AGENT_P4_BACKEND) :
  "heuristic" (défaut) — HeuristicQEEstimator, sans modèle, < 1 ms/bloc
  "llm"                — phi35 / qwen via ModelRuntime (lent, ~2-5 s/bloc)
"""

from __future__ import annotations

import json
import logging
import os

from .base import ModelRuntime, PipelineAgent, _extract_json
from .p4_qe_estimator import HeuristicQEEstimator

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a translation quality expert. Return JSON only.
Output schema: {"score":0.0,"issues":[...],"post_edit":"..."|null,"untranslated":[...]}
- score: float 0.0–1.0 (1.0 = perfect translation)
- issues: [{"type":"terminology|fluency|accuracy|omission|addition","desc":"...","severity":"minor|major|critical"}]
- post_edit: corrected translation string (only if score < 0.7), or null
- untranslated: [segment strings that were NOT translated and should be]

Rules:
- Do NOT correct technical terms, code, formulas, proper nouns
- Do NOT post-edit if score >= 0.7
- Mark as untranslated only segments that are clearly source-language text in a translated block
- For post_edit: keep the same register (formal/informal) as the source

If unsure: {"score":0.8,"issues":[],"post_edit":null,"untranslated":[]}"""


_FEW_SHOT: list[tuple[str, str]] = [
    (
        '{"source_lang":"en","target_lang":"fr","source":"The gradient descent algorithm minimizes the loss function.","translation":"L\'algorithme de descente de gradient minimise la fonction de perte."}',
        '{"score":0.95,"issues":[],"post_edit":null,"untranslated":[]}',
    ),
    (
        '{"source_lang":"en","target_lang":"fr","source":"Backpropagation computes gradients efficiently.","translation":"Backpropagation computes gradients efficiently."}',
        '{"score":0.1,"issues":[{"type":"omission","desc":"text not translated","severity":"critical"}],"post_edit":"La rétropropagation calcule les gradients efficacement.","untranslated":["Backpropagation computes gradients efficiently."]}',
    ),
    (
        '{"source_lang":"en","target_lang":"fr","source":"Use learning_rate=0.001 for stability.","translation":"Utilisez learning_rate=0.001 pour la stabilité."}',
        '{"score":0.9,"issues":[],"post_edit":null,"untranslated":[]}',
    ),
]

_MAX_FEW_SHOT = 3
_MIN_SCORE_FOR_POST_EDIT = 0.7


class P4TranslationAgent(PipelineAgent):
    """
    Agent P4 : validation qualité et post-édition de traduction.

    Par défaut utilise HeuristicQEEstimator (sans modèle, < 1 ms/bloc).
    Passer PIPELINE_AGENT_P4_BACKEND=llm pour activer le backend LLM.

    Entrée (``input_data``) :
    ```json
    {
      "source_lang": "en",
      "target_lang": "fr",
      "source": "The network learns features.",
      "translation": "Le réseau apprend des caractéristiques."
    }
    ```

    Sortie :
    ```json
    {
      "score": 0.9,
      "issues": [],
      "post_edit": null,
      "untranslated": []
    }
    ```
    """

    stage = "p4_translation"
    prompt_version = "v1"
    default_max_new_tokens = 200

    _VALID_ISSUE_TYPES = {"terminology", "fluency", "accuracy", "omission", "addition"}
    _VALID_SEVERITIES = {"minor", "major", "critical"}

    def __init__(self, runtime: ModelRuntime) -> None:
        super().__init__(runtime)
        self._heuristic_qe = HeuristicQEEstimator()
        self._backend = os.environ.get("PIPELINE_AGENT_P4_BACKEND", "heuristic").lower()

    # ------------------------------------------------------------------
    # Surcharge du chemin principal

    def is_available(self) -> bool:
        if self._backend == "llm":
            return self.runtime.is_available()
        return True  # heuristique toujours disponible

    def run(self, input_data: dict, *, use_cache: bool = True) -> dict:
        if self._backend == "llm":
            return super().run(input_data, use_cache=use_cache)
        return self._run_heuristic(input_data)

    def _run_heuristic(self, input_data: dict) -> dict:
        source = str(input_data.get("source") or "").strip()
        translation = str(input_data.get("translation") or "").strip()
        source_lang = str(input_data.get("source_lang") or "en").lower()
        target_lang = str(input_data.get("target_lang") or "fr").lower()
        try:
            return self._heuristic_qe.score(source, translation, source_lang, target_lang)
        except Exception as exc:
            logger.debug("[p4_translation/heuristic] échec: %s", exc)
            return {"score": 0.8, "issues": [], "post_edit": None, "untranslated": []}

    # ------------------------------------------------------------------
    # Chemin LLM (activé via PIPELINE_AGENT_P4_BACKEND=llm)

    def build_messages(self, input_data: dict) -> list[dict]:
        block_json = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user_ex, asst_ex in _FEW_SHOT[:_MAX_FEW_SHOT]:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": asst_ex})
        messages.append({"role": "user", "content": block_json})
        return messages

    def parse_response(self, raw: str, input_data: dict) -> dict:
        default: dict = {"score": 0.8, "issues": [], "post_edit": None, "untranslated": []}
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return default

        try:
            score = float(parsed.get("score") or 0.8)
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.8

        raw_issues = parsed.get("issues") or []
        valid_issues: list[dict] = []
        for issue in raw_issues:
            if not isinstance(issue, dict):
                continue
            itype = str(issue.get("type") or "").lower()
            severity = str(issue.get("severity") or "minor").lower()
            desc = str(issue.get("desc") or "")[:100]
            if itype in self._VALID_ISSUE_TYPES and severity in self._VALID_SEVERITIES:
                valid_issues.append({"type": itype, "desc": desc, "severity": severity})

        post_edit = None
        if score < _MIN_SCORE_FOR_POST_EDIT:
            raw_pe = parsed.get("post_edit")
            if isinstance(raw_pe, str) and raw_pe.strip():
                post_edit = raw_pe.strip()

        untranslated = [
            str(s) for s in (parsed.get("untranslated") or [])
            if str(s or "").strip()
        ]

        return {
            "score": score,
            "issues": valid_issues,
            "post_edit": post_edit,
            "untranslated": untranslated,
        }

    @classmethod
    def needs_validation(cls, block: dict, threshold: float = 0.0) -> bool:
        """Retourne True si le bloc mérite une validation de traduction."""
        has_translation = bool(
            block.get("translated_text")
            or any(
                (ln.get("translated_text") or "") for ln in (block.get("lines") or [])
            )
        )
        return has_translation

    @classmethod
    def build_input_from_block(
        cls,
        block: dict,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> dict:
        """Construit l'input de l'agent depuis un bloc pipeline."""
        source = " ".join(
            (ln.get("line_text") or ln.get("text") or "").strip()
            for ln in (block.get("lines") or [])
        ).strip()
        translation = str(block.get("translated_text") or "").strip()
        if not translation:
            translation = " ".join(
                (ln.get("translated_text") or "").strip()
                for ln in (block.get("lines") or [])
            ).strip()
        return {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source": source[:400],
            "translation": translation[:400],
        }
