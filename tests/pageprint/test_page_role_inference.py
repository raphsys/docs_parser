from pageprint.role_resolver import infer_page_role


def test_infers_index_from_dominant_index_entries():
    role = infer_page_role({"index_head_term": 25, "body_paragraph": 2}, current="body")
    assert role == "index"


def test_infers_toc_from_logical_structures():
    logical = {"toc_entries": [{"title_text": f"E{i}"} for i in range(20)]}
    assert infer_page_role({}, logical, current="body") == "toc"


def test_infers_table_page_from_cells():
    logical = {"tables": [{"cells": [{"text": "c"} for _ in range(30)]}]}
    assert infer_page_role({}, logical, current="body") == "table_page"


def test_does_not_override_explicit_role():
    # An already specific page_role is preserved.
    assert infer_page_role({"index_head_term": 25}, current="author_bio_page") == "author_bio_page"


def test_body_page_stays_body_without_dominance():
    assert infer_page_role({"body_paragraph": 12, "list_item": 3}, current="body") == "body"
