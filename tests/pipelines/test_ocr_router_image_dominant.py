from pipelines.ocr_router import route_ocr


def test_ocr_router_image_dominant():
    result = route_ocr(
        {
            "page_role": "cover",
            "layout_type": "image_dominant",
            "dimensions": {"width": 600, "height": 800},
            "regions": [{"id": "img1", "type": "image_region", "bbox": [0, 0, 600, 700], "text_probable": True}],
        },
        native_available=True,
        image_available=True,
    )
    assert result["mode"] == "targeted_regions"
    assert result["ocr_claims"][0]["source"] == "ocr_targeted_region"
