from pagereconstruct import compile_page_render_plan, validate
from pagereconstruct.visual_qa import assess as visual_assess
from tests.pagereconstruct._fixtures import translated_input_data


def test_visual_qa_returns_six_scores():
    vqa = visual_assess(compile_page_render_plan(translated_input_data()).to_dict())
    for k in ("text_presence", "non_text_presence", "overlap", "position", "typography"):
        assert 0.0 <= vqa["scores"][k] <= 1.0
    assert 0.0 <= vqa["publication_ready_score"] <= 1.0


def test_leak_high_caps_publication_ready():
    data = translated_input_data()  # source_background -> leak high
    vqa = visual_assess(compile_page_render_plan(data).to_dict())
    assert vqa["publication_ready_score"] <= 0.70


def test_validator_exposes_publication_ready_flag():
    result = validate(compile_page_render_plan(translated_input_data()).to_dict())
    assert "publication_ready" in result
    assert result["publication_ready"] is False  # leak high blocks it
