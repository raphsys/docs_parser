from pageprint.evidence.collector import collect_claims
from pageprint.structure_builders.publisher_mark_builder import build_artifacts


def _unit(unit_id, level, text, bbox):
    return {
        "unit_id": unit_id,
        "level": level,
        "content": {"text": text},
        "geometry": {"bbox": bbox},
        "understanding": {},
        "extraction": {"source": "native", "confidence": 0.7},
    }


def _claim_types(bundle, unit_id):
    return {claim["claim_type"] for claim in bundle["claims_by_unit"].get(unit_id, [])}


def test_heading_spans_are_not_page_references():
    units = [
        _unit("heading", "phrase", "CHAPTER 7", [180, 27, 216, 35]),
        _unit("heading_c", "span", "C", [180, 27, 186, 35]),
        _unit("heading_7", "span", "7", [209, 27, 216, 35]),
    ]

    bundle = collect_claims(
        units,
        [],
        {"page_role": "body", "page_geometry": {"height": 792, "render_dpi": 72}},
    )

    assert "page_reference" not in _claim_types(bundle, "heading_c")
    assert "page_reference" not in _claim_types(bundle, "heading_7")


def test_standalone_margin_page_number_remains_a_page_reference():
    units = [_unit("page_no", "line", "328", [57, 27, 71, 36])]

    bundle = collect_claims(
        units,
        [],
        {"page_role": "body", "page_geometry": {"height": 792, "render_dpi": 72}},
    )

    assert "page_reference" in _claim_types(bundle, "page_no")


def test_toc_phrase_page_reference_does_not_require_margin():
    units = [_unit("toc_ref", "phrase", "93", [500, 400, 520, 412])]

    bundle = collect_claims(
        units,
        [],
        {"page_role": "toc", "page_geometry": {"height": 792, "render_dpi": 72}},
    )

    assert "page_reference" in _claim_types(bundle, "toc_ref")


def test_heading_token_spans_are_not_page_number_artifacts():
    units = [
        _unit("heading", "line", "CHAPTER 7", [180, 27, 216, 35]),
        {
            **_unit("heading_c", "span", "C", [180, 27, 186, 35]),
            "parent_id": "heading",
        },
        {
            **_unit("heading_7", "span", "7", [209, 27, 216, 35]),
            "parent_id": "heading",
        },
    ]

    artifacts = build_artifacts(
        units,
        page_intelligence={"page_geometry": {"height": 792, "render_dpi": 72}},
    )

    assert artifacts["page_numbers"] == []
