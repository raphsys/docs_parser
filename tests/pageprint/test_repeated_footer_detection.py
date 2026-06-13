from pipelines.document_context import build_document_context


def test_repeated_footer_detection():
    pages = [
        {"dimensions": {"height": 800}, "blocks": [{"bbox": [40, 760, 560, 790], "text": "Technical Book"}]},
        {"dimensions": {"height": 800}, "blocks": [{"bbox": [40, 760, 560, 790], "text": "Technical Book"}]},
    ]
    context = build_document_context(pages)
    assert context["repeated_footers"] == [{"text": "Technical Book", "count": 2}]
