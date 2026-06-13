from tests.pageprint.test_caption_split import _caption_page


def test_caption_number_in_preservation_plan():
    assert any(item["text"] == "2.1" for item in _caption_page()["views"]["preservation_plan"])
