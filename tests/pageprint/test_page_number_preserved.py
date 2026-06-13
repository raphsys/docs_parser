from tests.pageprint.test_publisher_mark_exclusion import _artifact_page


def test_margin_page_number_is_preserved():
    assert any(item["text"] == "42" for item in _artifact_page("42")["views"]["preservation_plan"])
