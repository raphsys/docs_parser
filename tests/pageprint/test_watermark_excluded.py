from tests.pageprint.test_publisher_mark_exclusion import _artifact_page


def test_watermark_not_translated():
    assert not _artifact_page("DRAFT", [150, 300, 450, 360])["views"]["translation_plan"]
