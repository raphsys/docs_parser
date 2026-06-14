from pageprint.role_resolver import resolve_unit_role


def _unit(text, *, bold=False, size=None, level="line"):
    return {"level": level, "content": {"text": text},
            "visual": {"style": {"font_size_pt": size, "flags": {"bold": bold}}}, "understanding": {}}


def test_numbered_short_line_is_heading():
    role, _r, _c = resolve_unit_role(_unit("5.7 Challenges and Future Research"), page_intelligence={}, document_context={})
    assert role == "section_heading"


def test_bold_short_line_is_heading():
    role, _r, _c = resolve_unit_role(_unit("Challenges and Future Research", bold=True), page_intelligence={}, document_context={})
    assert role == "section_heading"


def test_long_sentence_is_body_not_heading():
    role, _r, _c = resolve_unit_role(
        _unit("This is a normal flowing sentence that clearly belongs to a body paragraph and ends with a period."),
        page_intelligence={}, document_context={})
    assert role == "body_paragraph"


def test_short_unstyled_line_not_forced_heading():
    # short last line of a paragraph, no style evidence -> not a heading
    role, _r, _c = resolve_unit_role(_unit("of the network"), page_intelligence={}, document_context={})
    assert role != "section_heading"
