from pipelines.ocr_router import route_ocr


def test_ocr_claims_not_direct_policy():
    result = route_ocr(
        {"layout_type": "image_dominant", "regions": [{"id": "r1", "type": "diagram_region", "bbox": [0, 0, 100, 100]}]},
        native_available=True,
        image_available=True,
    )
    assert result["decision_policy"] == "ocr_outputs_are_claims_for_evidence_resolver"
