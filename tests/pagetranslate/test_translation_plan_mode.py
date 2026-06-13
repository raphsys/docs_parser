import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pagetranslate import build_page_translation
from pagetranslate.protection import protect_text, restore_text


class FakeTranslator:
    def translate_text(self, text, **kwargs):
        return f"FR::{text}"


def _input_data():
    return {
        "schema_version": "pageprint.input.v1",
        "input_id": "input-test",
        "document": {"language": {"source_lang": "en", "target_lang": "fr"}},
        "page": {"page_role": "body"},
        "translation_context": {},
        "units": [
            {
                "unit_id": "u1",
                "level": "phrase",
                "content": {"text": "Hello MLP world."},
                "geometry": {"bbox": [0, 0, 100, 20], "reading_order_index": 1},
                "understanding": {"role": "body_paragraph", "object_type": "natural_text", "semantic_kind": "prose"},
                "policy": {"translatable": True, "translation_strategy": "layout_constrained"},
                "visual": {"style": {}},
                "children_ids": [],
            }
        ],
        "views": {
            "translation_plan": [
                {
                    "translation_unit_id": "tp_0001",
                    "unit_id": "seg_1",
                    "level": "semantic_phrase",
                    "source_unit_ids": ["u1"],
                    "source_text": "Hello MLP world.",
                    "role": "body_paragraph",
                    "object_type": "natural_text",
                    "semantic_kind": "prose",
                    "translation_mode": "translate",
                    "translation_strategy": "layout_constrained",
                    "protected_tokens": ["MLP"],
                    "render_target": {"reconstruction_unit_id": "ru_0001", "bbox": [0, 0, 100, 20], "style_source_unit_id": "u1"},
                    "qa_requirements": {"preserve_protected_tokens": True},
                }
            ]
        },
    }


def test_reads_translation_plan_in_normal_mode():
    result = build_page_translation(_input_data(), translator=FakeTranslator(), target_lang="fr")
    assert result["debug"]["selection_mode"] == "translation_plan"
    assert len(result["translation_units"]) == 1
    assert result["translation_units"][0]["role"] == "body_paragraph"
    assert result["translated_input_data"]["views"]["reconstruction_units"][0]["role"] == "body_paragraph"


def test_placeholder_restore_accepts_new_and_legacy_variants():
    protected, protections = protect_text("Keep 42 kg.")
    assert "⟦PT0001⟧" in protected
    assert restore_text("FR ⟦ PT0001 ⟧.", protections) == "FR 42 kg."
    assert restore_text("FR [[[ PT0001 ]]].", protections) == "FR 42 kg."
