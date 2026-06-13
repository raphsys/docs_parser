from __future__ import annotations


class ExternalModelEngine:
    """Adapter placeholder for external translation providers.

    The client object must expose `translate(text, source_lang, target_lang,
    context)`. No network provider is imported here so tests remain offline.
    """

    profile = "external_model"

    def __init__(self, client=None):
        self.client = client

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        if self.client is None:
            raise RuntimeError("ExternalModelEngine requires a provider client before use")
        return self.client.translate(text, source_lang=source_lang, target_lang=target_lang, context=context)
