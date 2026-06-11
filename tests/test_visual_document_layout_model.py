from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_document_layout_model import VisualDocumentLayoutModel, _extract_json


def test_extract_json_from_model_response():
    raw = 'analysis...\n{"faults":[{"type":"overlap"}],"placements":[],"warnings":[]}\n'
    parsed = _extract_json(raw)
    assert parsed["faults"][0]["type"] == "overlap"


def test_make_panel_contains_original_and_rendered(tmp_path):
    original_path = tmp_path / "original.png"
    Image.new("RGB", (20, 30), (240, 240, 240)).save(original_path)
    rendered = Image.new("RGB", (30, 20), (255, 255, 255))

    model = VisualDocumentLayoutModel(model_id="missing-local-model")
    panel = model._make_panel(str(original_path), rendered)

    assert panel.width > rendered.width
    assert panel.height >= 30


def test_analyze_returns_unavailable_without_model_load():
    model = VisualDocumentLayoutModel(model_id="missing-local-model")
    model._loaded = True
    model._available = False

    result = model.analyze(
        original_image_path=None,
        rendered_image=Image.new("RGB", (16, 16), "white"),
        layout_payload={"page": {"width": 10, "height": 10}},
    )

    assert result["available"] is False
    assert result["faults"] == []
    assert "visual_layout_model_unavailable" in result["warnings"]
