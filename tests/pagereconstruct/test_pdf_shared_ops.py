import pytest

from pagereconstruct import compile_page_render_plan
from pagereconstruct.backends import pdf_vector
from tests.pagereconstruct._fixtures import translated_input_data


def test_plan_background_is_under_layers():
    plan = compile_page_render_plan(translated_input_data()).to_dict()
    assert "background" in plan["layers"]


@pytest.mark.skipif(not pdf_vector.is_available(), reason="PyMuPDF unavailable")
def test_pdf_uses_renderer_layout_and_contains_text(tmp_path):
    plan = compile_page_render_plan(translated_input_data()).to_dict()
    out = tmp_path / "p.pdf"
    res = pdf_vector.render(plan, str(out))
    assert res["ok"]
    import fitz
    doc = fitz.open(str(out))
    text = doc[0].get_text()
    doc.close()
    assert "Bonjour" in text  # translated text rendered in the PDF
