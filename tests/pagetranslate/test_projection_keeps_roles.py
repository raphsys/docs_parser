import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pagetranslate import build_page_translation


def test_projection_keeps_roles_and_render_target():
    input_data = {
        "schema_version": "pageprint.input.v1",
        "input_id": "input-test",
        "document": {"language": {"source_lang": "en", "target_lang": "fr"}},
        "page": {"page_role": "toc"},
        "translation_context": {},
        "units": [{
            "unit_id": "line1",
            "level": "line",
            "content": {"text": "Image classification using MLP"},
            "geometry": {"bbox": [0, 0, 200, 20], "reading_order_index": 1},
            "understanding": {"role": "toc_entry", "object_type": "toc_entry", "semantic_kind": "toc_entry"},
            "policy": {"translatable": True, "translation_strategy": "layout_constrained"},
            "visual": {"style": {}},
            "children_ids": [],
        }],
        "views": {"translation_plan": [{
            "translation_unit_id": "tp_0001",
            "unit_id": "seg_0001",
            "level": "semantic_phrase",
            "source_unit_ids": ["line1"],
            "logical_unit_id": "toc_entry_0001",
            "source_text": "Image classification using MLP",
            "role": "toc_entry_title",
            "object_type": "natural_text",
            "semantic_kind": "toc_entry_title",
            "translation_mode": "translate",
            "translation_strategy": "layout_constrained",
            "protected_tokens": ["MLP"],
            "context": {"page_role": "toc"},
            "render_target": {"reconstruction_unit_id": "ru_0001", "bbox": [0, 0, 200, 20], "style_source_unit_id": "line1"},
            "qa_requirements": {"preserve_protected_tokens": True},
        }]},
    }
    result = build_page_translation(input_data, dry_run=True)
    reconstruction_unit = result["translated_input_data"]["views"]["reconstruction_units"][0]
    assert reconstruction_unit["role"] == "toc_entry_title"
    assert reconstruction_unit["object_type"] == "natural_text"
    assert reconstruction_unit["semantic_kind"] == "toc_entry_title"
    assert reconstruction_unit["logical_unit_id"] == "toc_entry_0001"
    assert reconstruction_unit["render_target"]
