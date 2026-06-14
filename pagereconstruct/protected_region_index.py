"""Spatial index of protected regions (no R-tree needed at V1, a list suffices).

A protected region is an area the renderer must not write translated text over,
and the patch planner must not erase. Built from preservation_plan, exclusion_plan,
unit policies, detected regions and visual layers (directive §15).
"""

from __future__ import annotations

from .schema import ProtectedRegion

# object_type / reason that mark a hard visual protection
_PROTECTED_OBJECTS = {"formula", "formula_expression", "equation", "image", "figure",
                      "code", "code_block", "table_grid", "logo", "watermark", "publisher_mark", "diagram"}
# Confirmed visual objects (regions). Formula/code "candidate" regions are
# observations only and must NOT hard-protect (they over-fire on citations and
# single glyphs); real formulas are protected via unit roles below.
_CONFIRMED_REGION_OBJECTS = {"image", "figure", "picture", "logo", "watermark",
                             "table_grid", "diagram", "chart"}


def _valid_bbox(b) -> bool:
    return isinstance(b, (list, tuple)) and len(b) == 4


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection_area(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


class ProtectedRegionIndex:
    def __init__(self, regions: list[ProtectedRegion]):
        self.regions = [r for r in regions if _valid_bbox(r.bbox)]

    def intersections(self, bbox, min_ratio: float = 0.0) -> list[ProtectedRegion]:
        if not _valid_bbox(bbox):
            return []
        denom = max(1e-6, _area([float(x) for x in bbox]))
        out = []
        for r in self.regions:
            ratio = _intersection_area([float(x) for x in bbox], [float(x) for x in r.bbox]) / denom
            if ratio > min_ratio:
                out.append(r)
        return out

    def overlaps(self, bbox, min_ratio: float = 0.01) -> bool:
        return bool(self.intersections(bbox, min_ratio=min_ratio))

    def overlap_ratio(self, bbox) -> float:
        if not _valid_bbox(bbox):
            return 0.0
        denom = max(1e-6, _area([float(x) for x in bbox]))
        best = 0.0
        for r in self.regions:
            best = max(best, _intersection_area([float(x) for x in bbox], [float(x) for x in r.bbox]) / denom)
        return round(best, 4)

    def __len__(self) -> int:
        return len(self.regions)


def build_protected_region_index(*, units: dict, preservation_plan: list, exclusion_plan: list,
                                 regions: list | None = None, visual_layers: dict | None = None) -> ProtectedRegionIndex:
    out: list[ProtectedRegion] = []
    n = 0

    def add(source, reason, bbox, hard=True, z="preserve_original"):
        nonlocal n
        if not _valid_bbox(bbox):
            return
        n += 1
        out.append(ProtectedRegion(id=f"prot_{n:04d}", source=source, reason=str(reason or "protected"),
                                   bbox=[float(x) for x in bbox], hard=hard, z_policy=z))

    for p in preservation_plan or []:
        # text-preserved exact (page refs, captions labels) is over_text; visual is original.
        mode = p.get("preservation_mode")
        z = "over_text" if mode == "preserve_text_exactly" else "preserve_original"
        add("preservation_plan", p.get("reason"), p.get("bbox"), z=z)
    for e in exclusion_plan or []:
        add("exclusion_plan", e.get("reason") or "excluded", e.get("bbox"))
    for u in (units or {}).values() if isinstance(units, dict) else (units or []):
        # Sub-line fragments do not hard-protect: their block/line parent does,
        # and protecting every span/word/char floods false collisions.
        if u.get("level") in {"span", "word", "char"}:
            continue
        policy = u.get("policy") or {}
        bbox = (u.get("geometry") or {}).get("bbox")
        role = (u.get("understanding") or {}).get("role")
        ot = (u.get("understanding") or {}).get("object_type")
        if policy.get("render_policy") in {"background_only", "fixed_preserve", "preserve_overlay"} \
                or policy.get("preservation_mode") in {"preserve_as_visual_overlay"} \
                or role in {"formula_expression", "publisher_mark", "watermark"} \
                or str(ot or "").lower() in _PROTECTED_OBJECTS:
            add("unit_policy", role or ot or policy.get("render_policy"), bbox)
    for r in regions or []:
        rtype = str(r.get("region_type") or "").lower()
        # Candidate / observation-only regions are not hard protections.
        if "candidate" in rtype or r.get("observation_only") or r.get("policy_pending"):
            continue
        ot = str(r.get("object_type") or rtype).lower()
        if any(k in ot for k in _CONFIRMED_REGION_OBJECTS):
            add("region", r.get("object_type") or r.get("region_type"), r.get("bbox"))
    return ProtectedRegionIndex(out)
