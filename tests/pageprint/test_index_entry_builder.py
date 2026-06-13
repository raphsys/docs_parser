from tests.pageprint.test_index_page_detection import _index_page


def test_index_entry_has_head_and_refs():
    entry = _index_page()["logical_structures"]["index_entries"][0]
    assert entry["head_term"] == "PostGIS"
    assert entry["page_refs"] == ["242"]
