from __future__ import annotations

import os

from .ct2_engine import CTranslate2Engine
from .external_model_engine import ExternalModelEngine
from .local_model_engine import LocalModelEngine
from .mock_engine import EchoEngine, MockEngine, PrefixEngine
from .rule_engine import RuleEngine


def create_translation_engine(engine_name: str | None = None, **kwargs):
    """Build a translation engine by name.

    Extra keyword arguments (inventory_path, model_name, device, compute_type,
    batch_size, max_input_tokens, source_lang, target_lang) are forwarded to the
    CTranslate2 engine and ignored by the simpler engines.
    """
    name = str(engine_name or os.environ.get("TRANSLATION_ENGINE") or "mock").strip().lower()
    if name == "mock":
        return MockEngine()
    if name == "echo":
        return EchoEngine()
    if name in {"prefix", "prefix_mock"}:
        return PrefixEngine()
    if name in {"rule", "rules"}:
        return RuleEngine()
    if name in {"ct2", "ctranslate2"}:
        return CTranslate2Engine(**{key: value for key, value in kwargs.items() if value is not None})
    if name in {"local", "local_model"}:
        return LocalModelEngine()
    if name in {"external", "external_model"}:
        return ExternalModelEngine()
    raise ValueError(f"unknown translation engine: {engine_name}")
