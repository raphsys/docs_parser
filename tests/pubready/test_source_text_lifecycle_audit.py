from pagereconstruct.source_text_lifecycle_ledger import audit_source_text_lifecycle
from pubready.stages import render_ops_audit


def _normalized():
    return {
        "units": [
            {
                "unit_id": "b1",
                "level": "block",
                "content": {"text": "Hello world"},
                "geometry": {"bbox": [10, 10, 100, 30]},
            },
            {
                "unit_id": "l1",
                "level": "line",
                "parent_id": "b1",
                "content": {"text": "Hello world"},
                "geometry": {"bbox": [10, 10, 100, 30]},
            },
        ],
        "translated_units": [
            {
                "unit_id": "ru1",
                "reconstruction_unit_id": "ru1",
                "translation_unit_id": "tu1",
                "source_unit_ids": ["b1"],
                "translated_text": "Bonjour le monde",
                "bbox": [10, 10, 100, 30],
            }
        ],
        "preservation_plan": [],
        "exclusion_plan": [],
    }


def _plan_with_textop():
    return {
        "render_ops": [
            {"op_type": "background", "path": "clean.png"},
            {
                "op_type": "text",
                "unit_id": "b1",
                "source_unit_ids": ["b1"],
                "translation_unit_id": "tu1",
                "composition_id": "c1",
                "block_id": "b1",
                "line_id": "ln1",
                "run_id": "r1",
                "lines": [{"text": "Bonjour le monde"}],
            },
        ],
        "layers": {"translated_text": [{"source_unit_ids": ["b1"], "translated_text": "Bonjour le monde"}]},
    }


def test_source_text_lifecycle_blocks_missing_render_op():
    plan = _plan_with_textop()
    plan["render_ops"] = [op for op in plan["render_ops"] if op["op_type"] != "text"]

    audit = audit_source_text_lifecycle(plan, _normalized())

    assert audit["status"] == "ko"
    assert "source_text_missing_render_op" in audit["hard_blockers"]


def test_source_text_lifecycle_covers_children_by_rendered_parent():
    audit = audit_source_text_lifecycle(_plan_with_textop(), _normalized())

    assert audit["status"] == "ok"
    assert audit["missing_count"] == 0
    assert {e["source_unit_id"] for e in audit["ledger"]} == {"b1", "l1"}


def test_render_ops_audit_consumes_source_text_lifecycle():
    plan = _plan_with_textop()
    plan["render_ops"] = [op for op in plan["render_ops"] if op["op_type"] != "text"]

    stage = render_ops_audit.audit_page(plan, _normalized())

    assert stage.status == "ko"
    assert "source_text_missing_render_op" in stage.hard_blockers
