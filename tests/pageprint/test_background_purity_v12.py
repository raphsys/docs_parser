from pageprint.detection.builder import normalize_detected_region
from pageprint.region_index import build_regions
from pipelines.background_cover import collect_background_purity_boxes
from pipelines.background_cover import _local_fill_color
from source_ownership import build_source_ownership
from PIL import Image


def _unit(uid, level, bbox, text="", parent=None):
    return {
        "unit_id": uid,
        "level": level,
        "parent_id": parent,
        "content": {"text": text},
        "geometry": {"bbox": bbox},
        "understanding": {"role": "body_paragraph" if text else level},
        "policy": {"translatable": bool(text)},
    }


def test_detector_hit_stays_candidate_without_explicit_confirmation():
    region = normalize_detected_region({"special_class": "formula", "bbox": [10, 10, 20, 20]})
    assert region["region_type"] == "formula_candidate_region"
    assert region["observation_only"] is True
    assert region["skip_translation"] is False


def test_region_index_does_not_promote_candidate_automatically():
    regions = build_regions({"special_regions": [{
        "region_type": "formula_candidate_region",
        "bbox": [10, 10, 20, 20],
        "confidence": 0.99,
    }]})
    assert [r["region_type"] for r in regions] == ["formula_candidate_region"]


def test_special_region_does_not_claim_page_container():
    data = {
        "units": [
            _unit("page", "page", [0, 0, 500, 700]),
            _unit("line", "line", [10, 10, 100, 20], "ordinary prose", "page"),
        ],
        "regions": [{
            "region_id": "formula",
            "region_type": "formula_region",
            "bbox": [300, 300, 340, 320],
            "members": {},
        }],
    }
    ownership = build_source_ownership(data)
    assert ownership["page"]["state"] == "background_visual"
    assert ownership["line"]["state"] == "translation_candidate"


def test_background_cover_rejects_page_sized_container_box():
    data = {
        "page": {"geometry": {"width": 500, "height": 700}},
        "units": [
            _unit("page", "page", [0, 0, 500, 700]),
            _unit("line", "line", [20, 20, 200, 35], "translate me", "page"),
        ],
        "regions": [],
    }
    boxes = collect_background_purity_boxes(data)
    assert [0.0, 0.0, 500.0, 700.0] not in boxes
    assert any(b[0] <= 20 and b[2] >= 200 for b in boxes)


def test_local_fill_keeps_dark_coloured_substrate():
    image = Image.new("RGB", (100, 40), (72, 112, 139))
    for x in range(35, 65):
        for y in range(15, 22):
            if (x + y) % 3 == 0:
                image.putpixel((x, y), (245, 245, 245))
    fill = _local_fill_color(image, [30, 10, 70, 28], (255, 255, 255))
    assert fill[2] > fill[0]
    assert sum(fill) < 400
