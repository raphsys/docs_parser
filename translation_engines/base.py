from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationResult:
    translated_text: str
    raw_output: str | None = None
    metadata: dict | None = None


class TranslationEngine:
    profile: str = "external_model"
    supports_batch: bool = False

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        raise NotImplementedError

    def translate_batch(self, requests: list[dict]) -> list[dict]:
        return [
            {
                "translated_text": self.translate(
                    req.get("text") or "",
                    req.get("source_lang") or "auto",
                    req.get("target_lang") or "fr",
                    req.get("context") or {},
                ),
                "raw_output": None,
                "metadata": {},
            }
            for req in requests
        ]

    def healthcheck(self) -> dict:
        return {"status": "unknown", "engine": self.profile}
