from __future__ import annotations


class MockEngine:
    """Deterministic engine used to validate protection, projection and QA plumbing."""

    profile = "mock"

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        return f"{str(target_lang or 'xx').upper()}::{text}"


class PrefixEngine:
    """Visible no-op translation engine for smoke trials."""

    profile = "mock"

    def __init__(self, prefix: str = "FR::"):
        self.prefix = prefix

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        return f"{self.prefix}{text}"


class EchoEngine:
    """Identity engine for QA tests that need an unchanged output."""

    profile = "echo"

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        return text
