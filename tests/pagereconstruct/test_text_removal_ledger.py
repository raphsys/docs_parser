from pagereconstruct.text_removal_ledger import audit_text_removal_ledger, build_ledger


def test_source_text_visible_under_translation_blocks_publication():
    ledger = [{
        "entry_id": "tre1",
        "source_unit_ids": ["su1"],
        "expected_action": "clean_background_removed",
        "clean_background_verified": True,
        "final_render_verified": True,
        "residual_source_text_score": 0.91,
        "status": "ko",
    }]

    result = audit_text_removal_ledger(["su1"], ledger)

    assert result["status"] == "ko"
    assert "source_text_visible_under_translation" in result["hard_blockers"]


def test_text_removal_ledger_covers_translated_and_protected_units():
    plan = {
        "layers": {
            "background": [{"mode": "clean_background", "clean_background_verified": True, "text_removed": True}],
            "translated_text": [{
                "id": "ru1",
                "source_unit_ids": ["su1"],
                "source_text": "Hello",
                "translated_text": "Bonjour",
                "role": "body_paragraph",
                "coverage_bbox": [0, 0, 20, 10],
            }],
            "preserved_overlays": [{
                "id": "pres1",
                "source_unit_ids": ["su2"],
                "text": "12",
                "reason": "page_number",
                "bbox": [90, 0, 100, 10],
            }],
        }
    }

    ledger = build_ledger(plan, background_mode="clean_background", clean_background_verified=True)
    result = audit_text_removal_ledger(["su1", "su2"], [e.to_dict() for e in ledger])

    assert result["status"] == "ok"
    assert {sid for e in ledger for sid in e.source_unit_ids} == {"su1", "su2"}
