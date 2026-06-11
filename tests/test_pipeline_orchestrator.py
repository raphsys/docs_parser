"""Tests du PipelineOrchestrator (clone objectif).

Construit un petit PDF en mémoire avec PyMuPDF, exécute le pipeline complet
(SourceLoader → PageRenderer → RawExtractors → PageUnderstanding → PAGEPRINT)
et vérifie que chaque page produit un INPUT_DATA valide.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fitz = pytest.importorskip("fitz")

from pageprint import PAGEPRINT_SCHEMA_VERSION
from pipelines import PipelineOrchestrator
from pipelines.orchestrator import _parse_pages


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Heading of page {i + 1}", fontsize=18)
        page.insert_text(
            (72, 140),
            "The model learns features from data and generalizes.",
            fontsize=11,
        )
        page.insert_text((72, 170), "E = mc^2", fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


class TestOrchestrator:
    def test_run_produces_valid_input_data(self, sample_pdf):
        orchestrator = PipelineOrchestrator(enable_ocr=False)
        result = orchestrator.run(sample_pdf)
        assert result["document"]["page_count"] == 2
        assert len(result["pages"]) == 2
        for page_result in result["pages"]:
            assert page_result["status"] == "ok", page_result.get("error")
            input_data = page_result["input_data"]
            assert input_data["schema_version"] == PAGEPRINT_SCHEMA_VERSION
            assert input_data["units"], "extraction native attendue"
            report = input_data["debug"]["validation"]
            assert report["valid"], report["errors"]

    def test_page_selection_string(self, sample_pdf):
        orchestrator = PipelineOrchestrator(enable_ocr=False)
        result = orchestrator.run(sample_pdf, pages="2")
        assert [p["page"] for p in result["pages"]] == [2]

    def test_bboxes_in_points_within_page(self, sample_pdf):
        orchestrator = PipelineOrchestrator(enable_ocr=False)
        result = orchestrator.run(sample_pdf, pages="1")
        input_data = result["pages"][0]["input_data"]
        geometry = input_data["page"]["geometry"]
        assert geometry["unit"] == "pt"
        assert abs(geometry["width"] - 595) < 1.0
        for unit in input_data["units"]:
            bbox = unit["geometry"]["bbox"]
            if bbox:
                assert unit["geometry"]["bbox_unit"] == "pt"
                assert 0 <= bbox[0] <= geometry["width"] + 1
                assert 0 <= bbox[3] <= geometry["height"] + 1

    def test_pipeline_trace_present(self, sample_pdf):
        orchestrator = PipelineOrchestrator(enable_ocr=False)
        result = orchestrator.run(sample_pdf, pages="1")
        trace = result["pages"][0]["pipeline_trace"]
        assert "page_renderer" in trace
        assert "raw_extractors" in trace
        assert "pageprint" in trace
        assert trace["pageprint"]["valid"] is True

    def test_understanding_disabled_still_valid(self, sample_pdf):
        orchestrator = PipelineOrchestrator(enable_ocr=False, enable_understanding=False)
        result = orchestrator.run(sample_pdf, pages="1")
        page_result = result["pages"][0]
        assert page_result["status"] == "ok"
        assert page_result["input_data"]["debug"]["validation"]["valid"]

    def test_status_reports_modules(self):
        status = PipelineOrchestrator().status()
        assert status["pageprint_schema"] == "pageprint.input.v1"
        assert status["modules"]["pageprint"] is True
        assert status["modules"]["fitz"] is True

    def test_compatibility_keeps_legacy_structure(self, sample_pdf):
        orchestrator = PipelineOrchestrator(enable_ocr=False)
        result = orchestrator.run(sample_pdf, pages="1")
        legacy = result["pages"][0]["input_data"]["compatibility"]["legacy_page_structure"]
        assert legacy.get("blocks")


class TestParsePages:
    def test_none_returns_all(self):
        assert _parse_pages(None, 3) == [0, 1, 2]

    def test_string_ranges(self):
        assert _parse_pages("1,3-5", 10) == [0, 2, 3, 4]

    def test_out_of_range_filtered(self):
        assert _parse_pages("1,99", 2) == [0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
