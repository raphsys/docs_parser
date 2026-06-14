from pagereconstruct.renderer_dispatcher import dispatch


def test_dispatch_by_renderer_name():
    assert dispatch("paragraph", None).renderer_name == "paragraph"
    assert dispatch("heading", None).renderer_name == "heading"
    assert dispatch("code", None).renderer_name == "code"
    assert dispatch("formula", None).renderer_name == "formula"


def test_dispatch_by_role():
    assert dispatch(None, "section_heading").renderer_name == "heading"
    assert dispatch(None, "table_body_cell").renderer_name == "table_cell"
    assert dispatch(None, "body_paragraph").renderer_name == "paragraph"


def test_unknown_role_is_review_not_paragraph():
    r = dispatch(None, "totally_unknown_role")
    assert r.renderer_name == "anchored_label_review"
    assert r.renderer_name != "paragraph"
