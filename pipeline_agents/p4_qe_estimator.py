"""
P4 QE — Estimateurs de qualité de traduction sans génération LLM.

Remplace le backend LLM de P4TranslationAgent par des heuristiques légères
(HeuristicQEEstimator) ou, à terme, par un modèle discriminatif spécialisé
(CometKiwi, ChrF, …) sans avoir besoin d'un LLM génératif.

Schéma de sortie commun (même que P4TranslationAgent) :
    {"score": float, "issues": list, "post_edit": None|str, "untranslated": list}

Pour brancher un estimateur custom :
    class MonEstimateur(QEEstimatorBase):
        def score(self, source, translation, source_lang, target_lang) -> dict: ...
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

class QEEstimatorBase(ABC):
    """Interface commune pour tout estimateur de qualité de traduction."""

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def score(
        self,
        source: str,
        translation: str,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> dict:
        """
        Retourne {"score": float, "issues": list, "post_edit": None, "untranslated": list}.
        Ne lève jamais d'exception.
        """


# ---------------------------------------------------------------------------
# Estimateur heuristique
# ---------------------------------------------------------------------------

class HeuristicQEEstimator(QEEstimatorBase):
    """
    Estimation de qualité basée sur :
      1. Ratio de longueur source/traduction
      2. Détection de segments non traduits (stopwords de la langue source)
      3. Confiance de la langue cible (marqueurs lexicaux et orthographiques)
      4. Identité source == traduction

    Avantages : sans modèle, < 1 ms par bloc, 0 dépendance externe.
    Limites : ne détecte pas les erreurs de sens subtiles ni la fluidité.
    """

    # Stopwords représentatifs par langue (mots grammaticaux très fréquents)
    _STOPWORDS: dict[str, frozenset] = {
        "en": frozenset({
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "this", "that", "these", "those", "it", "he", "she", "we", "they",
            "i", "you", "his", "her", "its", "our", "their", "my", "your",
            "not", "so", "as", "if", "then", "than", "when", "which", "what",
            "who", "how", "all", "each", "both", "any", "some", "also", "just",
        }),
        "fr": frozenset({
            "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "dans",
            "en", "au", "aux", "pour", "de", "du", "par", "avec", "est", "pas",
            "sont", "ce", "cet", "cette", "ces", "se", "ne", "qui", "que", "dont",
            "sa", "son", "ses", "leur", "leurs", "nous", "vous", "ils",
            "elles", "je", "tu", "il", "elle", "on", "me", "te", "lui", "y",
            "si", "tout", "tous", "toute", "toutes", "plus", "bien",
            "aussi", "comme", "alors", "donc", "car", "ni", "sans",
        }),
        "de": frozenset({
            "der", "die", "das", "ein", "eine", "und", "oder", "aber", "in",
            "auf", "an", "zu", "mit", "von", "für", "ist", "sind", "war",
            "waren", "sein", "haben", "hat", "hatte", "ich", "du", "er", "sie",
            "es", "wir", "ihr", "nicht", "auch", "so", "als", "wenn", "dann",
            "noch", "nur", "dem", "den", "des",
        }),
        "es": frozenset({
            "el", "la", "los", "las", "un", "una", "y", "o", "pero", "en",
            "de", "del", "al", "para", "con", "es", "son", "por", "que",
            "yo", "no", "se", "me", "te", "le", "nos", "lo", "como",
        }),
        "pt": frozenset({
            "o", "a", "os", "as", "um", "uma", "e", "ou", "mas", "em",
            "de", "do", "da", "dos", "das", "para", "com", "por", "que",
            "não", "se", "como",
        }),
        "it": frozenset({
            "il", "la", "i", "le", "un", "una", "e", "o", "ma", "in",
            "di", "del", "della", "dei", "per", "con", "non", "si", "che",
            "come", "questo", "questa", "sono",
        }),
    }

    # Caractères accentués typiques — révélateurs de langue non-ASCII
    _TARGET_ACCENTS: dict[str, frozenset] = {
        "fr": frozenset("éèêëàâùûüîïçœæÉÈÊËÀÂÙÛÜÎÏÇŒÆ"),
        "de": frozenset("äöüßÄÖÜ"),
        "es": frozenset("áéíóúüñÁÉÍÓÚÜÑ"),
        "pt": frozenset("ãõâêîôûáéíóúçÃÕÂÊÎÔÛÁÉÍÓÚÇ"),
        "it": frozenset("àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"),
        "ru": frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"),
    }

    # Termes cross-linguistiques légitimes (ne pas pénaliser s'ils restent en anglais)
    _CROSSLINGUAL_RE = re.compile(
        r"""
        \b(?:
            [A-Z]{2,}                              # Acronymes : GPU, API, PDF
          | \d+(?:[.,]\d+)*                        # Nombres
          | [A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+  # snake_case : learning_rate
          | [A-Z][a-z]+(?:[A-Z][a-z]+)+            # CamelCase : BackPropagation
          | https?://\S+                            # URLs
        )\b
        """,
        re.VERBOSE,
    )

    # -----------------------------------------------------------------------

    def score(
        self,
        source: str,
        translation: str,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> dict:
        source = (source or "").strip()
        translation = (translation or "").strip()

        # Source vide → on ne peut pas juger
        if not source:
            return {"score": 1.0, "issues": [], "post_edit": None, "untranslated": []}

        # Traduction vide → échec total
        if not translation:
            return {
                "score": 0.0,
                "issues": [{"type": "omission", "desc": "translation is empty", "severity": "critical"}],
                "post_edit": None,
                "untranslated": [source],
            }

        score = 1.0
        issues: list[dict] = []
        untranslated: list[str] = []

        # 1. Ratio de longueur (caractères)
        char_ratio = len(translation) / max(1, len(source))
        if char_ratio < 0.25:
            score -= 0.5
            issues.append({
                "type": "omission",
                "desc": f"translation too short (char ratio={char_ratio:.2f})",
                "severity": "critical",
            })
        elif char_ratio < 0.40:
            score -= 0.20
            issues.append({
                "type": "omission",
                "desc": f"translation possibly truncated (char ratio={char_ratio:.2f})",
                "severity": "major",
            })
        elif char_ratio > 4.0:
            score -= 0.25
            issues.append({
                "type": "addition",
                "desc": f"translation much longer than source (char ratio={char_ratio:.2f})",
                "severity": "major",
            })

        # 2. Identité exacte source == traduction
        if source.lower() == translation.lower():
            score = min(score, 0.15)
            issues.append({
                "type": "omission",
                "desc": "translation identical to source (not translated)",
                "severity": "critical",
            })
            untranslated.append(source)

        # 3. Détection de segments non traduits via stopwords de la langue source
        else:
            src_sw = self._STOPWORDS.get(source_lang.lower(), frozenset())
            if src_sw:
                src_words = re.findall(r"\b\w+\b", source.lower())
                src_sw_present = [w for w in src_words if w in src_sw]
                if src_sw_present:
                    trans_lower = translation.lower()
                    in_trans = [
                        w for w in src_sw_present
                        if re.search(rf"\b{re.escape(w)}\b", trans_lower)
                    ]
                    ratio = len(in_trans) / len(src_sw_present)
                    if ratio >= 0.60:
                        score -= 0.45
                        issues.append({
                            "type": "omission",
                            "desc": (
                                f"text likely not translated — "
                                f"{ratio:.0%} of source stopwords found verbatim in translation"
                            ),
                            "severity": "critical",
                        })
                        untranslated.append(source)
                    elif ratio >= 0.30:
                        score -= 0.15
                        issues.append({
                            "type": "accuracy",
                            "desc": (
                                f"partial translation suspected — "
                                f"{ratio:.0%} of source stopwords found in translation"
                            ),
                            "severity": "major",
                        })

        # 4. Vérification langue cible (marqueurs lexicaux et orthographiques)
        if target_lang.lower() in self._TARGET_ACCENTS and len(translation) > 20:
            accent_chars = self._TARGET_ACCENTS[target_lang.lower()]
            tgt_sw = self._STOPWORDS.get(target_lang.lower(), frozenset())
            trans_lower = translation.lower()

            has_accent = any(c in accent_chars for c in translation)
            has_tgt_stopword = any(
                re.search(rf"\b{re.escape(w)}\b", trans_lower) for w in tgt_sw
            )

            if not has_accent and not has_tgt_stopword:
                # Vérifier si le texte ressemble à la langue source OU à l'anglais
                # (l'anglais est souvent utilisé à tort comme langue cible)
                trans_words = re.findall(r"\b\w+\b", trans_lower)
                # Langues à tester : source + anglais (si source != anglais)
                wrong_lang_candidates = [source_lang.lower()]
                if source_lang.lower() != "en":
                    wrong_lang_candidates.append("en")

                for cand_lang in wrong_lang_candidates:
                    cand_sw = self._STOPWORDS.get(cand_lang, frozenset())
                    cand_count = sum(1 for w in trans_words if w in cand_sw)
                    if trans_words and cand_count / len(trans_words) > 0.12:
                        score -= 0.30
                        issues.append({
                            "type": "accuracy",
                            "desc": (
                                f"translation appears to be in {cand_lang!r} "
                                f"instead of target ({target_lang})"
                            ),
                            "severity": "critical",
                        })
                        break

        return {
            "score": round(max(0.0, min(1.0, score)), 4),
            "issues": issues,
            "post_edit": None,
            "untranslated": list(dict.fromkeys(untranslated)),
        }
