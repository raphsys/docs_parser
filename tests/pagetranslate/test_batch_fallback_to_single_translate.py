from pagetranslate import build_page_translation


class SingleEngine:
    profile = "single"

    def translate(self, text, source_lang, target_lang, context):
        return f"FR::{text}"


def test_batch_fallback_to_single_translate():
    result = build_page_translation(
        {
            "schema_version": "pageprint.input.v1",
            "input_id": "single-test",
            "document": {"language": {"source_lang": "en", "target_lang": "fr"}},
            "page": {"page_role": "body"},
            "translation_context": {},
            "units": [{
                "unit_id": "u1",
                "level": "phrase",
                "content": {"text": "Hello world."},
                "geometry": {"bbox": [0, 0, 100, 20], "reading_order_index": 1},
                "understanding": {"role": "body_paragraph", "object_type": "natural_text", "semantic_kind": "prose"},
                "policy": {"translatable": True, "translation_strategy": "layout_constrained"},
                "visual": {"style": {}},
                "children_ids": [],
            }],
            "views": {
                "translation_plan": [{
                    "translation_unit_id": "tp_0001",
                    "unit_id": "seg_1",
                    "level": "semantic_phrase",
                    "source_unit_ids": ["u1"],
                    "source_text": "Hello world.",
                    "role": "body_paragraph",
                    "object_type": "natural_text",
                    "semantic_kind": "prose",
                    "translation_mode": "translate",
                    "translation_strategy": "layout_constrained",
                    "render_target": {"reconstruction_unit_id": "ru_0001", "bbox": [0, 0, 100, 20], "style_source_unit_id": "u1"},
                    "qa_requirements": {},
                }]
            },
        },
        translator=SingleEngine(),
        target_lang="fr",
    )
    assert result["translation_units"][0]["translated_text"].startswith("FR::")
    assert result["translation_runtime_status"] == "ok"

