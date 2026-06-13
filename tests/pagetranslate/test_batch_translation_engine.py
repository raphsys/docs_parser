from pagetranslate import build_page_translation


class BatchEngine:
    profile = "batch"
    supports_batch = True

    def translate_batch(self, requests):
        return [
            {
                "translated_text": f"FR::{request['text']}",
                "raw_output": f"RAW::{request['text']}",
                "metadata": {"latency_ms": 1.0, "input_token_count": len(request["text"].split()), "output_token_count": len(request["text"].split()) + 1, "truncated": False},
            }
            for request in requests
        ]


def _input_data():
    return {
        "schema_version": "pageprint.input.v1",
        "input_id": "batch-test",
        "document": {"language": {"source_lang": "en", "target_lang": "fr"}},
        "page": {"page_role": "body"},
        "translation_context": {},
        "units": [{
            "unit_id": "u1",
            "level": "phrase",
            "content": {"text": "Hello MLP world."},
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
                "source_text": "Hello MLP world.",
                "role": "body_paragraph",
                "object_type": "natural_text",
                "semantic_kind": "prose",
                "translation_mode": "translate",
                "translation_strategy": "layout_constrained",
                "protected_tokens": ["MLP"],
                "render_target": {"reconstruction_unit_id": "ru_0001", "bbox": [0, 0, 100, 20], "style_source_unit_id": "u1"},
                "qa_requirements": {"preserve_protected_tokens": True},
            }]
        },
    }


def test_batch_translation_engine_uses_batch_path():
    result = build_page_translation(_input_data(), translator=BatchEngine(), target_lang="fr")
    assert result["translation_units"][0]["translated_text"].startswith("FR::")
    assert result["translation_units"][0]["engine_trace"]["engine"] == "batch"
    assert result["translation_units"][0]["engine_trace"]["raw_engine_output"].startswith("RAW::")
    assert result["translation_runtime_status"] == "ok"

