from __future__ import annotations

import fitz

from logical_block_rebuilder import rebuild_logical_blocks_from_pdf


class _FakePage:
    rect = fitz.Rect(0, 0, 300, 400)

    def get_text(self, kind):
        assert kind == "words"
        return [
            (10, 10, 30, 20, "Here,", 0, 0, 0),
            (34, 10, 42, 20, "Z", 0, 0, 1),
            (46, 10, 54, 20, "is", 0, 0, 2),
            (58, 10, 88, 20, "output", 0, 0, 3),
            (92, 10, 110, 20, "of", 0, 0, 4),
            (114, 10, 160, 20, "softmax", 0, 0, 5),
            (40, 40, 75, 60, "∂E", 0, 1, 0),
            (78, 40, 100, 60, "∂Zk", 0, 1, 1),
            (10, 80, 45, 90, "These", 0, 2, 0),
            (48, 80, 82, 90, "delta", 0, 2, 1),
            (85, 80, 126, 90, "values", 0, 2, 2),
            (130, 80, 168, 90, "enable", 0, 2, 3),
            (10, 94, 62, 104, "calculation", 0, 3, 0),
            (66, 94, 80, 104, "of", 0, 3, 1),
            (84, 94, 126, 104, "gradient", 0, 3, 2),
        ]


def test_rebuild_logical_blocks_from_pdf_subtracts_formula_regions():
    page_data = {
        "dimensions": {"width": 300, "height": 400, "unit": "px"},
        "blocks": [{"id": "old", "role": "body", "bbox": [0, 0, 200, 110], "lines": []}],
        "formula_regions": [
            {"id": "f0", "formula_subregions": [{"bbox": [35, 35, 110, 65]}]},
            {"id": "f1", "formula_subregions": [{"bbox": [200, 200, 220, 220]}]},
        ],
    }

    out, info = rebuild_logical_blocks_from_pdf(page_data, _FakePage(), sx=1.0, sy=1.0)

    assert info["changed"] is True
    text = " ".join(block.get("text") or "" for block in out["blocks"])
    assert "∂E" not in text
    assert "Here" in text
    assert "These delta values enable calculation of gradient" in text
