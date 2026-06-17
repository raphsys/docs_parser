"""M6 — zones spéciales + preservation_audit."""
from pagereconstruct.composition.special_zone_preserver import classify_zones
from pubready.stages import preservation_audit


def test_classify_zones_levels():
    plan = {"page": {"height_pt": 800}, "protected_regions": [
        {"id": "z1", "bbox": [10, 10, 50, 20], "reason": "formula"},
        {"id": "z2", "bbox": [400, 20, 470, 36], "reason": "page_number"},
        {"id": "z3", "bbox": [50, 300, 470, 600], "reason": "image_region"}]}
    zones = classify_zones(plan)
    lv = {z.zone_id: z.level for z in zones}
    assert lv["z1"] == "block" and lv["z2"] == "page" and lv["z3"] == "background"
    assert [z for z in zones if z.zone_id == "z1"][0].critical


def test_preservation_audit_flags_text_over_critical_zone():
    plan = {"page": {}, "protected_regions": [{"id": "f", "bbox": [10, 10, 200, 40], "reason": "formula"}],
            "layers": {"translated_text": [{"id": "t", "layout_bbox": [10, 10, 200, 40]}], "patches": []}}
    r = preservation_audit.audit_page(plan, {})
    assert r.status == "ko" and "special_zone_overlap" in r.hard_blockers


def test_preservation_audit_ok_when_no_overlap():
    plan = {"page": {}, "protected_regions": [{"id": "f", "bbox": [10, 10, 100, 40], "reason": "formula"}],
            "layers": {"translated_text": [{"id": "t", "layout_bbox": [10, 200, 100, 230]}], "patches": []}}
    r = preservation_audit.audit_page(plan, {})
    assert r.status == "ok"
