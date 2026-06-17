from pagereconstruct.page_level_contracts import (
    FigureCaptionContract,
    InlineLinkContract,
    PageNumberContract,
    audit_page_level_contracts,
)


def test_page_number_not_translated():
    contract = PageNumberContract(page_number="284", bbox=[500, 20, 530, 35], translate=True)

    result = audit_page_level_contracts(page_numbers=[contract])

    assert result["status"] == "ko"
    assert "page_number_translated" in result["hard_blockers"]


def test_page_number_not_duplicated():
    result = audit_page_level_contracts(page_numbers=[
        PageNumberContract(page_number="284", bbox=[500, 20, 530, 35]),
        PageNumberContract(page_number="284", bbox=[500, 760, 530, 775]),
    ])

    assert result["status"] == "ko"
    assert "duplicate_page_number" in result["hard_blockers"]


def test_figure_caption_attached_to_figure():
    caption = FigureCaptionContract(
        caption_id="cap1",
        figure_id="fig1",
        caption_number="Figure 1.",
        caption_text="Architecture.",
        number_bbox=[100, 400, 145, 415],
        text_bbox=[150, 400, 280, 415],
        combined_bbox=[100, 400, 280, 415],
        anchor="",
    )

    result = audit_page_level_contracts(figure_captions=[caption])

    assert result["status"] == "ko"
    assert "figure_caption_unanchored" in result["hard_blockers"]


def test_inline_url_stays_inline_blue_run():
    link = InlineLinkContract(
        source_text="https://example.test",
        translated_text="https://example.test",
        url="https://example.test",
        style={"color": "#000000"},
        run_policy="block",
    )

    result = audit_page_level_contracts(inline_links=[link])

    assert result["status"] == "ko"
    assert "inline_url_not_inline_blue_run" in result["hard_blockers"]
