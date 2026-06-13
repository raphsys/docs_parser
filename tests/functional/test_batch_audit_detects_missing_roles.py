from tools.run_functional_audit import audit_pages


def test_batch_audit_detects_missing_roles():
    page = {
        "units": [{"unit_id": "u1", "level": "phrase", "content": {"text": "Hello"}, "understanding": {"role": "body_paragraph"}}],
        "views": {"translation_plan": [{"translation_unit_id": "tp1", "source_unit_ids": ["u1"], "source_text": "Hello", "object_type": "natural_text", "semantic_kind": "prose", "translation_mode": "translate", "render_target": {"reconstruction_unit_id": "ru1"}}]},
    }
    result = audit_pages([page])
    assert result["functional_status"] == "ko"
    assert result["metrics"]["role_none_translation_items"] == 1
