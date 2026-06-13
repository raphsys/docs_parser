from tools.run_functional_audit import audit_pages


def test_batch_audit_detects_empty_plan():
    page = {
        "units": [{"unit_id": "u1", "level": "phrase", "content": {"text": "Hello world sentence."}, "understanding": {"role": "body_paragraph"}, "policy": {"translatable": True}}],
        "views": {"translation_plan": []},
    }
    result = audit_pages([page])
    assert result["functional_status"] == "ko"
    assert "pages_using_fallback_gt_0" in result["errors"]
