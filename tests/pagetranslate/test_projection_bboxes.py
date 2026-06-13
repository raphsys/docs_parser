from pagetranslate.projection import _semantic_reconstruction_unit


def test_semantic_unit_keeps_separate_bboxes():
    item = {
        "level": "semantic_phrase", "role": "body_paragraph",
        "source_text": "x", "translated_text": "y", "bbox": [50, 50, 400, 260],
        "source_unit_ids": ["blk"],
        "render_target": {"layout_bbox": [50, 50, 400, 260], "patch_bbox": [50, 50, 400, 260],
                          "coverage_bbox": [50, 50, 400, 260], "anchor_bbox": [50, 50, 400, 60],
                          "reconstruction_unit_id": "ru_0001"},
    }
    ru = _semantic_reconstruction_unit(item, {})
    assert ru["layout_bbox"] == [50, 50, 400, 260]
    assert ru["anchor_bbox"] == [50, 50, 400, 60]
    assert ru["patch_bbox"] == [50, 50, 400, 260]
    assert ru["bbox"] == ru["layout_bbox"]


def test_small_render_target_bbox_does_not_shrink_layout():
    # render_target only carries a first-line bbox; logical item.bbox must win.
    item = {"level": "semantic_phrase", "role": "body_paragraph", "source_text": "x",
            "translated_text": "y", "bbox": [50, 50, 400, 260], "source_unit_ids": [],
            "render_target": {"bbox": [50, 50, 400, 60]}}
    ru = _semantic_reconstruction_unit(item, {})
    assert ru["layout_bbox"] == [50, 50, 400, 260]
