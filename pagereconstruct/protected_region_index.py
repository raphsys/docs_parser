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


def _zone_of(bbox, zones, min_ratio: float = 0.6):
    if not zones or not _valid_bbox(bbox):
        return None
    b = [float(x) for x in bbox]
    denom = max(1e-6, _area(b))
    for z in zones:
        if _intersection_area(b, z["bbox"]) / denom >= min_ratio:
            return z
    return None


def build_protected_region_index(*, units: dict, preservation_plan: list, exclusion_plan: list,
                                 regions: list | None = None, visual_layers: dict | None = None,
                                 translated_source_ids: set | None = None,
                                 special_zones: list | None = None) -> ProtectedRegionIndex:
    # Coherence invariant: a source unit that is being TRANSLATED must never also
    # be protected — else the same content lands in both the translated layer and
    # the protected layer and they self-collide (the upstream formula/protected
    # over-detection symptom). translated_source_ids carries every source id the
    # translated_text layer consumes; we drop any protection that targets one.
    translated_source_ids = translated_source_ids or set()
    special_zones = special_zones or []
    out: list[ProtectedRegion] = []
    n = 0

    def add(source, reason, bbox, hard=True, z="preserve_original"):
        nonlocal n
        if not _valid_bbox(bbox):
            return
        n += 1
        out.append(ProtectedRegion(id=f"prot_{n:04d}", source=source, reason=str(reason or "protected"),
                                   bbox=[float(x) for x in bbox], hard=hard, z_policy=z))

    # Merged formula/code zones are THE tight protection for their block
    # (one rectangle replacing many oversized per-unit formula bboxes).
    for z in special_zones:
        add("special_zone", z.get("kind") or "formula", z.get("bbox"))

    for p in preservation_plan or []:
        # Skip a preservation whose source is also translated (over-detected
        # formula/protected on a line the pipeline translates).
        if translated_source_ids & set(p.get("source_unit_ids") or []):
            continue
        # A preservation inside a merged zone is already covered by the zone.
        if _zone_of(p.get("bbox"), special_zones):
            continue
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
        # A translated unit must not also be protected (coherence invariant).
        if u.get("unit_id") in translated_source_ids:
            continue
        # Candidate-region units are special-region-detector OBSERVATIONS (inline
        # "1 × 1" fragments sitting inside translatable prose); they over-fire and
        # must not hard-protect. Real formulas protect via block/line roles above.
        if "candidate" in str(u.get("unit_id") or ""):
            continue
        policy = u.get("policy") or {}
        bbox = (u.get("geometry") or {}).get("bbox")
        role = (u.get("understanding") or {}).get("role")
        ot = (u.get("understanding") or {}).get("object_type")
        rp = policy.get("render_policy")
        ot_l = str(ot or "").lower()
        # A unit hard-protects ONLY if it carries a confirmed visual identity:
        #  - a real protected object_type (formula/image/code/table_grid…), OR
        #  - a confirmed visual role (formula_expression/publisher_mark/watermark), OR
        #  - render_policy == background_only (true page background).
        # render_policy fixed_preserve / preserve_overlay alone (page numbers,
        # small artifacts, unknown-role text) must NOT hard-protect: those flood
        # false collisions over the translated text and never carry real visuals.
        is_object = ot_l in _PROTECTED_OBJECTS
        is_visual_role = role in {"formula_expression", "publisher_mark", "watermark"}
        is_background = rp == "background_only"
        if not (is_object or is_visual_role or is_background):
            continue
        # A formula/code unit inside a merged zone is already protected by the
        # tight zone rectangle; its own (often oversized) bbox would re-introduce
        # the collisions with neighbouring prose. Let the zone own it.
        is_formula_code = role in {"formula_expression", "code_line", "code_block"} \
            or ot_l in {"formula", "formula_expression", "equation", "code", "code_block"}
        if is_formula_code and _zone_of(bbox, special_zones):
            continue
        add("unit_policy", role or ot or rp, bbox)
    for r in regions or []:
        rtype = str(r.get("region_type") or "").lower()
        # Candidate / observation-only regions are not hard protections.
        if "candidate" in rtype or r.get("observation_only") or r.get("policy_pending"):
            continue
        ot = str(r.get("object_type") or rtype).lower()
        if any(k in ot for k in _CONFIRMED_REGION_OBJECTS):
            add("region", r.get("object_type") or r.get("region_type"), r.get("bbox"))
    return ProtectedRegionIndex(out)
