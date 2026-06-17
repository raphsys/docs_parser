from pubready.stages.render_ops_audit import audit_page


def test_textop_requires_composition_id():
    result = audit_page({
        "layers": {"translated_text": [{"id": "ru1", "translated_text": "Bonjour", "role": "body_paragraph"}]},
        "render_ops": [{"op_type": "background"}, {"op_type": "text", "unit_id": "ru1", "lines": []}],
    }, {})

    assert result.status == "ko"
    assert "textop_missing_composition_id" in result.hard_blockers


def test_no_textop_from_raw_reconstruction_unit():
    result = audit_page({
        "layers": {"translated_text": [{"id": "ru1", "translated_text": "Bonjour", "role": "body_paragraph"}]},
        "render_ops": [{
            "op_type": "text",
            "unit_id": "ru1",
            "composition_id": "comp1",
            "source": "raw_reconstruction_unit",
            "lines": [],
        }],
    }, {})

    assert result.status == "ko"
    assert "textop_from_raw_reconstruction_unit" in result.hard_blockers
