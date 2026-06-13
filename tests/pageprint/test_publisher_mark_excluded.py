from pageprint.role_resolver import resolve_unit_role
from pageprint.semantic_builder import _build_translation_segments_from_logical_structures


def test_estadisticos_footer_detected_as_publisher_mark():
    unit = {"level": "line", "content": {"text": "Estadísticos e-Books & Papers"}, "understanding": {}}
    role, reason, conf = resolve_unit_role(unit, page_intelligence={}, document_context={})
    assert role == "publisher_mark"


def test_publisher_footer_excluded_from_translation_segments():
    logical = {
        "body_paragraphs": [
            {"logical_unit_id": "p1", "source_unit_ids": ["u1"], "text": "A normal translatable paragraph about data."},
            {"logical_unit_id": "p2", "source_unit_ids": ["u2"], "text": "Estadísticos e-Books & Papers"},
        ]
    }
    segments = _build_translation_segments_from_logical_structures(logical)
    texts = [s["source_text"] for s in segments]
    assert "A normal translatable paragraph about data." in texts
    assert not any("Estadísticos" in t for t in texts)
