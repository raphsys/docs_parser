from pageprint.role_resolver import resolve_unit_role
from pageprint.structure_builders.toc_builder import _looks_like_toc_row


def test_table_dominant_layout_does_not_force_table_cell():
    unit = {"level": "line", "content": {"text": "This is a normal editorial sentence in a book."},
            "understanding": {}}
    role, _r, _c = resolve_unit_role(unit, page_intelligence={"layout_type": "table_dominant"}, document_context={})
    assert role != "table_body_cell"
    assert role == "body_paragraph"


def test_index_entry_not_toc_row():
    assert _looks_like_toc_row("smallserial data type, 27,") is False
    assert _looks_like_toc_row("snake case, 10, 94,") is False


def test_real_toc_row_is_toc():
    assert _looks_like_toc_row("1.2 Introduction to Databases . . . . . 45") is True
    assert _looks_like_toc_row("Chapter Three      112") is True
