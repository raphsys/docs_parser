from pipelines.ocr_router import route_ocr


def test_native_text_not_enough_for_large_image_region():
    result = route_ocr(
        {
            "layout_type": "single_column",
            "native_text_density": 0.02,
            "dimensions": {"width": 600, "height": 800},
            "regions": [{"id": "fig1", "type": "figure_region", "bbox": [50, 100, 550, 500]}],
        },
        native_available=True,
        image_available=True,
    )
    assert result["mode"] == "targeted_regions"
