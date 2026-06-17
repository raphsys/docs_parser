from pubready.page_auditor import evaluate_page


def _minimal_normalized():
    return {
        "page": {"page_index": 1},
        "units": [{"unit_id": "su1", "text": "Source", "geometry": {"bbox": [10, 10, 80, 30]}}],
        "translated_units": [{"translation_unit_id": "tu1", "source_unit_ids": ["su1"], "translated_text": "Cible"}],
    }


def _plan(background, *, visual=None):
    plan = {
        "page": {"page_index": 1, "width_pt": 100, "height_pt": 100},
        "layers": {
            "background": [background],
            "translated_text": [{
                "id": "ru1",
                "source_unit_ids": ["su1"],
                "source_text": "Source",
                "translated_text": "Cible",
                "role": "body_paragraph",
                "bbox": [10, 10, 80, 30],
                "coverage_bbox": [10, 10, 80, 30],
            }],
            "patches": [],
        },
        "final_contract": {
            "layer_order": ["background_clean", "translated_text"],
            "blocks": [{
                "layout": {"layout_bbox": [10, 10, 80, 30]},
                "style": {"font_size_pt": 10},
                "render": {"renderer_name": "paragraph"},
                "source_unit_ids": ["su1"],
            }],
            "source_unit_states": [{
                "source_unit_id": "su1",
                "state": "translated_and_rendered",
                "text_removal_entry_id": "tre1",
                "textop_ids": ["txt1"],
                "preservationop_ids": [],
            }],
        },
        "text_removal_ledger": [{
            "entry_id": "tre1",
            "source_unit_ids": ["su1"],
            "expected_action": "clean_background_removed",
            "clean_background_verified": True,
            "final_render_verified": True,
            "residual_source_text_score": 0.0,
            "status": "ok",
            "findings": [],
        }],
        "render_ops": [{
            "op_type": "background",
            "path": background.get("path"),
            "mode": "publication",
            "is_clean": background.get("mode") == "clean_background",
        }, {
            "op_type": "text",
            "unit_id": "ru1",
            "composition_id": "comp1",
            "block_id": "ru1",
            "line_id": "l1",
            "run_id": "r1",
            "source_unit_ids": ["su1"],
            "translation_unit_id": "tu1",
            "lines": [{"text": "Cible", "x": 10, "y_top": 10}],
        }],
        "visual_image_audit": visual or {"image_qa_executed": True, "score": 1.0},
    }
    return plan


def test_publication_forbids_source_image_background():
    report = evaluate_page(
        _plan({"mode": "source_background", "path": "/tmp/source.png", "source_text_leak_risk": "high"}),
        _minimal_normalized(),
        mode="publication",
    )

    assert report.status == "ko"
    assert "source_background_forbidden" in report.hard_blockers
    assert not report.publication_ready


def test_source_text_visible_under_translation_blocks_publication():
    report = evaluate_page(
        _plan(
            {"mode": "clean_background", "path": "/tmp/clean.png", "clean_background_verified": True, "source_text_leak_risk": "low"},
            visual={"image_qa_executed": True, "old_text_visible": True, "score": 0.4},
        ),
        _minimal_normalized(),
        mode="publication",
    )

    assert report.status == "ko"
    assert "source_text_leak" in report.hard_blockers
    assert report.publication_ready_score <= 0.50
