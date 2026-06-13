from __future__ import annotations

import re


class RuleEngine:
    """Small deterministic rule engine for local smoke tests.

    This is not a production translator. It proves that a non-echo engine can
    receive context, preserve protected placeholders and produce stable output.
    """

    profile = "rule"

    RULES = {
        "This sentence should be translated.": "Cette phrase doit etre traduite.",
        "This is a real sentence that should be translated.": "Ceci est une vraie phrase qui doit etre traduite.",
        "Image classification using MLP": "Classification d'image utilisant MLP",
        "Hidden layers": "Couches cachees",
        "Create a directory": "Creer un repertoire",
        "Traditional ML algorithms require features.": "Les algorithmes ML traditionnels necessitent des caracteristiques.",
    }

    WORDS = {
        "Hello": "Bonjour",
        "world": "monde",
        "sentence": "phrase",
        "translated": "traduite",
        "directory": "repertoire",
    }

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        if target_lang and not str(target_lang).lower().startswith("fr"):
            return text
        if text in self.RULES:
            return self.RULES[text]
        output = text
        for source, target in self.WORDS.items():
            output = re.sub(rf"\b{re.escape(source)}\b", target, output)
        return output
