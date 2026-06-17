from pubready.stages.contract_audit import audit_source_unit_states


def test_source_unit_cannot_be_preserved_and_translated():
    result = audit_source_unit_states({
        "source_unit_states": [{
            "source_unit_id": "su1",
            "state": "translated_and_rendered",
            "textop_ids": ["txt1"],
            "preservationop_ids": ["pres1"],
        }]
    })

    assert result["status"] == "ko"
    assert "source_unit_both_preserved_and_translated" in result["hard_blockers"]
