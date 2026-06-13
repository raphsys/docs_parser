from tests.pageprint.test_body_paragraph_builder import _body_page


def test_body_page_uses_logical_body_paragraphs():
    assert _body_page()["semantic_system"]["segment_source"] == "logical_body_paragraphs"
