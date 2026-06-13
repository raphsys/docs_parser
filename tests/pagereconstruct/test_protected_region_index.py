from pagereconstruct import build_protected_region_index
from pagereconstruct.protected_region_index import ProtectedRegionIndex
from pagereconstruct.schema import ProtectedRegion


def test_intersections_and_ratio():
    idx = ProtectedRegionIndex([ProtectedRegion(id="p1", source="x", reason="formula", bbox=[0, 0, 100, 100])])
    assert idx.overlaps([50, 50, 150, 150], min_ratio=0.1)
    assert not idx.overlaps([200, 200, 250, 250])
    assert idx.overlap_ratio([0, 0, 50, 50]) == 1.0  # fully inside


def test_build_from_plans_and_regions():
    idx = build_protected_region_index(
        units={},
        preservation_plan=[{"reason": "formula", "bbox": [10, 10, 50, 50], "preservation_mode": "preserve_as_visual_overlay"}],
        exclusion_plan=[{"reason": "publisher_mark", "bbox": [0, 600, 100, 620]}],
        regions=[{"object_type": "image", "bbox": [200, 200, 300, 300]}],
    )
    assert len(idx) == 3
    assert idx.overlaps([15, 15, 40, 40])
