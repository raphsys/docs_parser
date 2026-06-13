from __future__ import annotations


class LocalModelEngine:
    """Adapter placeholder for a local translation model.

    Pass a callable model with signature `(text, source_lang, target_lang,
    context) -> str`. Keeping this adapter explicit prevents trial code from
    importing heavyweight ML dependencies by default.
    """

    profile = "local_model"

    def __init__(self, model=None):
        self.model = model

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        if self.model is None:
            raise RuntimeError("LocalModelEngine requires a model callable before use")
        return self.model(text, source_lang, target_lang, context)
