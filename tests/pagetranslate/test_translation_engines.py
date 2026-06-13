import pytest

from translation_engines import create_translation_engine
from translation_engines.external_model_engine import ExternalModelEngine
from translation_engines.local_model_engine import LocalModelEngine


def test_create_mock_engine():
    engine = create_translation_engine("mock")
    assert engine.translate("Hello", "en", "fr", {}) == "FR::Hello"


def test_rule_engine_translates_known_segment():
    engine = create_translation_engine("rule")
    assert engine.translate("Hidden layers", "en", "fr", {}) == "Couches cachees"


def test_local_model_engine_requires_callable():
    with pytest.raises(RuntimeError):
        LocalModelEngine().translate("Hello", "en", "fr", {})


def test_external_model_engine_requires_client():
    with pytest.raises(RuntimeError):
        ExternalModelEngine().translate("Hello", "en", "fr", {})
