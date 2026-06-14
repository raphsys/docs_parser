from pagereconstruct.renderer_dispatcher import dispatch


def test_index_roles_use_index_renderer():
    assert dispatch(None, "index_entry").renderer_name == "index"
    assert dispatch(None, "index_subentry").renderer_name == "index"
    assert dispatch("index", None).renderer_name == "index"


def test_bibliography_uses_bibliography_renderer():
    assert dispatch(None, "bibliography_entry").renderer_name == "bibliography"


def test_index_not_anchored_label():
    assert dispatch(None, "index_entry").renderer_name != "anchored_label"
