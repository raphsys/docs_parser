from pubready.stages.intrablock_audit import audit_reading_order


def test_intrablock_composition_preserves_sentence_order():
    result = audit_reading_order([{
        "block_id": "b1",
        "source_reading_order": ["su1", "su2"],
        "render_order": ["su2", "su1"],
    }])

    assert result["status"] == "ko"
    assert "reading_order_changed" in result["hard_blockers"]
