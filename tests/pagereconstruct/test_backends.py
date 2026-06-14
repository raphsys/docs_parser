import pytest

from pagereconstruct import compile_page_render_plan
from pagereconstruct.backends import pdf_vector, raster_debug, select_backend
from tests.pagereconstruct._fixtures import translated_input_data


def test_select_backend_prefers_pdf_when_available():
    b = select_backend("pdf_vector")
    assert b is (pdf_vector if pdf_vector.is_available() else raster_debug)


@pytest.mark.skipif(not pdf_vector.is_available(), reason="PyMuPDF unavailable")
def test_pdf_vector_renders_simple_page(tmp_path):
    plan = compile_page_render_plan(translated_input_data()).to_dict()
    out = tmp_path / "page.pdf"
    res = pdf_vector.render(plan, str(out))
    assert res["ok"] and out.is_file() and out.stat().st_size > 0


def test_backend_reads_background_from_plan():
    plan = compile_page_render_plan(translated_input_data()).to_dict()
    assert plan["layers"]["background"][0]["mode"] in {"clean_background", "source_background", "blank_degraded"}
