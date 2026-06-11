from __future__ import annotations

import fitz

from layout_compiler import compile_page_layout


def test_compile_page_layout_moves_text_below_fixed_obstacle():
    result = compile_page_layout(
        page_bbox=[0, 0, 300, 400],
        fixed_items=[{"block_id": "formula", "bbox": [40, 70, 260, 120]}],
        text_items=[
            {"block_id": "a", "bbox": [40, 40, 260, 75], "role": "body"},
            {"block_id": "b", "bbox": [40, 95, 260, 130], "role": "body"},
            {"block_id": "c", "bbox": [40, 135, 260, 165], "role": "body"},
        ],
    )

    assert result["ok"] is True
    placements = {item["block_id"]: item for item in result["placements"]}
    rects = []
    for block_id, original in {
        "a": [40, 40, 260, 75],
        "b": [40, 95, 260, 130],
        "c": [40, 135, 260, 165],
    }.items():
        p = placements[block_id]
        rects.append(fitz.Rect(original[0] + p["dx"], original[1] + p["dy"], original[2] + p["dx"], original[3] + p["dy"]))

    fixed = fitz.Rect(40, 70, 260, 120)
    assert all((rect & fixed).get_area() <= 0.5 for rect in rects)
    assert all((rects[idx] & rects[idx + 1]).get_area() <= 0.5 for idx in range(len(rects) - 1))


def test_compile_page_layout_refuses_when_no_space():
    result = compile_page_layout(
        page_bbox=[0, 0, 200, 120],
        fixed_items=[{"block_id": "formula", "bbox": [20, 30, 180, 90]}],
        text_items=[
            {"block_id": "a", "bbox": [20, 20, 180, 70], "role": "body"},
            {"block_id": "b", "bbox": [20, 65, 180, 115], "role": "body"},
        ],
    )

    assert result["ok"] is False
    assert result["placements"] == []
