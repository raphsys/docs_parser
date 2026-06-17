"""
MultiBlock Layout Solver — external, non-invasive module.

Purpose:
    Resolve dense-page collisions after local candidate placement by optimizing
    groups of blocks inside flow regions (columns, index columns, figure text
    bands). This complements the existing local PlacementSolver.

Core idea:
    - Build FlowRegions from contract blocks and protected obstacles.
    - Group blocks by reading order / column / role.
    - Generate candidate boxes for each block.
    - Pack candidates vertically with hard no-overlap constraints.
    - Return layout patches. Do not mutate unless explicitly requested.

No third-party optimizer dependency. Deterministic greedy+backtracking; OR-Tools
can later replace solve_region() without changing the API.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class LayoutPatch:
    block_id: str
    old_bbox: BBox
    new_bbox: BBox
    strategy: str
    score_delta: float
    findings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FlowRegion:
    region_id: str
    bbox: BBox
    block_ids: List[str]
    obstacles: List[BBox] = field(default_factory=list)
    mode: str = "vertical_flow"  # vertical_flow | index_columns | locked


@dataclass
class MultiBlockSolveResult:
    status: str  # ok | review | ko | skipped
    patches_by_block_id: Dict[str, LayoutPatch]
    regions: List[FlowRegion]
    findings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "patches_by_block_id": {k: asdict(v) for k, v in self.patches_by_block_id.items()},
            "regions": [asdict(r) for r in self.regions],
            "findings": self.findings,
        }


def _bbox(b: Any) -> Optional[BBox]:
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return tuple(float(x) for x in b)
    return None


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersect(a: BBox, b: BBox) -> Optional[BBox]:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _overlap_ratio(a: BBox, b: BBox) -> float:
    inter = _intersect(a, b)
    return 0.0 if not inter else _area(inter) / max(1e-6, min(_area(a), _area(b)))


def _height(b: BBox) -> float:
    return max(0.0, b[3] - b[1])


def _shift_y(b: BBox, y0: float) -> BBox:
    h = _height(b)
    return (b[0], y0, b[2], y0 + h)


def _block_bbox(block: Any) -> Optional[BBox]:
    layout = getattr(block, "layout", None)
    return _bbox(getattr(layout, "layout_bbox", None) or getattr(layout, "coverage_bbox", None) or getattr(layout, "source_bbox", None))


def _block_id(block: Any) -> str:
    return str(getattr(block, "block_id", ""))


def _role(block: Any) -> str:
    return str(getattr(block, "role", ""))


def _is_locked(block: Any) -> bool:
    role = _role(block)
    if role in {"table_body_cell", "table_header_cell", "table_numeric_cell", "formula",
                "formula_expression", "equation", "code", "code_block", "code_line",
                "diagram_label", "axis_label", "legend_label", "page_number", "page_reference"}:
        return True
    layout = getattr(block, "layout", None)
    return bool(getattr(layout, "bbox_locked", False))


def collect_protected_obstacles(contract: Any) -> List[BBox]:
    obstacles: List[BBox] = []
    for obj in getattr(contract, "objects", []) or []:
        b = _bbox(getattr(obj, "bbox", None))
        role = str(getattr(obj, "role", "") or getattr(obj, "object_type", ""))
        policy = str(getattr(obj, "preservation_policy", "") or "")
        if b and (policy or role in {"image", "figure", "formula", "logo", "watermark", "table_grid"}):
            obstacles.append(b)
    # Also accept contract.preservation entries if present.
    pres = getattr(contract, "preservation", None)
    for item in getattr(pres, "items", []) or []:
        b = _bbox(getattr(item, "bbox", None))
        if b:
            obstacles.append(b)
    return obstacles


def build_flow_regions(contract: Any) -> List[FlowRegion]:
    blocks = [b for b in (getattr(contract, "blocks", []) or []) if _block_bbox(b)]
    if not blocks:
        return []
    page = getattr(contract, "page", None) or getattr(contract, "page_info", None)
    width = float(getattr(page, "width_pt", 0.0) or getattr(page, "width", 0.0) or 0.0)
    height = float(getattr(page, "height_pt", 0.0) or getattr(page, "height", 0.0) or 0.0)
    if not width or not height:
        xs = [v for b in blocks for v in (_block_bbox(b)[0], _block_bbox(b)[2])]
        ys = [v for b in blocks for v in (_block_bbox(b)[1], _block_bbox(b)[3])]
        width, height = max(xs), max(ys)

    obstacles = collect_protected_obstacles(contract)
    unlocked = [b for b in blocks if not _is_locked(b)]
    if not unlocked:
        return []

    # crude column clustering by x-center. Stable and dependency-free.
    centers = [(b, (_block_bbox(b)[0] + _block_bbox(b)[2]) / 2.0) for b in unlocked]
    centers.sort(key=lambda x: x[1])
    mid = width / 2.0
    left = [b for b, cx in centers if cx < mid]
    right = [b for b, cx in centers if cx >= mid]
    regions: List[FlowRegion] = []
    if left and right and min(len(left), len(right)) >= 2:
        regions.append(FlowRegion("col_left", (0.0, 0.0, mid, height), [_block_id(b) for b in left], obstacles, "vertical_flow"))
        regions.append(FlowRegion("col_right", (mid, 0.0, width, height), [_block_id(b) for b in right], obstacles, "vertical_flow"))
    else:
        regions.append(FlowRegion("page_flow", (0.0, 0.0, width, height), [_block_id(b) for b in unlocked], obstacles, "vertical_flow"))
    return regions


def _candidate_boxes(base: BBox, region: FlowRegion, max_shift: float = 36.0) -> List[Tuple[str, BBox]]:
    cands = [("original", base)]
    # local moves, then larger flow moves.
    for dy in (-24, -16, -8, 8, 16, 24, 36, -36):
        nb = (base[0], base[1] + dy, base[2], base[3] + dy)
        if nb[1] >= region.bbox[1] and nb[3] <= region.bbox[3] and abs(dy) <= max_shift:
            cands.append((f"shift_{dy}", nb))
    return cands


def _valid_box(box: BBox, placed: List[BBox], obstacles: List[BBox], *, text_overlap: float, protected_overlap: float) -> bool:
    for p in placed:
        if _overlap_ratio(box, p) > text_overlap:
            return False
    for ob in obstacles:
        if _overlap_ratio(box, ob) > protected_overlap:
            return False
    return True


def _max_overlap(box: BBox, others: List[BBox], obstacles: List[BBox]) -> float:
    best = 0.0
    for o in others:
        best = max(best, _overlap_ratio(box, o))
    for ob in obstacles:
        best = max(best, _overlap_ratio(box, ob))
    return best


def solve_region(block_by_id: Dict[str, Any], region: FlowRegion, *, text_overlap=0.10, protected_overlap=0.01) -> Tuple[Dict[str, LayoutPatch], List[Dict[str, Any]]]:
    """Net-improvement only: ne déplace QUE les blocs qui collisionnent vraiment,
    et seulement si le candidat réduit leur chevauchement, sans en créer ailleurs.
    Une page propre ne reçoit AUCUN patch (ne casse jamais un bon layout)."""
    findings: List[Dict[str, Any]] = []
    ordered = [block_by_id[i] for i in region.block_ids if i in block_by_id]
    ordered.sort(key=lambda b: (_block_bbox(b)[1], _block_bbox(b)[0]))
    # positions courantes (toutes d'origine au départ)
    cur: Dict[str, BBox] = {}
    for b in ordered:
        bb = _block_bbox(b)
        if bb:
            cur[_block_id(b)] = bb
    patches: Dict[str, LayoutPatch] = {}
    obstacles = region.obstacles

    for block in ordered:
        bid = _block_id(block)
        base = cur.get(bid)
        if not base:
            continue
        others = [v for k, v in cur.items() if k != bid]
        base_bad = _max_overlap(base, others, obstacles)
        thr = max(text_overlap, protected_overlap)
        if base_bad <= thr:
            continue  # ce bloc ne collisionne pas -> on n'y touche pas
        best, best_bad = None, base_bad
        for strategy, cand in _candidate_boxes(base, region):
            if strategy == "original":
                continue
            bad = _max_overlap(cand, others, obstacles)
            if bad < best_bad - 1e-3:
                best, best_bad = (strategy, cand), bad
        if best and best_bad < base_bad - 1e-3:
            strategy, newb = best
            cur[bid] = newb
            patches[bid] = LayoutPatch(bid, base, newb, strategy, round(base_bad - best_bad, 3),
                                       [{"type": "multiblock_layout_adjusted", "strategy": strategy,
                                         "overlap_before": round(base_bad, 3), "overlap_after": round(best_bad, 3)}])
        else:
            findings.append({"type": "multiblock_no_improvement", "block_id": bid, "severity": "info", "region_id": region.region_id})
    return patches, findings


def solve_multiblock_layout(contract: Any, *, enabled: bool = True) -> MultiBlockSolveResult:
    if not enabled:
        return MultiBlockSolveResult("skipped", {}, [], [{"type": "multiblock_solver_disabled"}])
    blocks = getattr(contract, "blocks", []) or []
    block_by_id = {_block_id(b): b for b in blocks if _block_id(b)}
    regions = build_flow_regions(contract)
    all_patches: Dict[str, LayoutPatch] = {}
    findings: List[Dict[str, Any]] = []
    for region in regions:
        patches, f = solve_region(block_by_id, region)
        all_patches.update(patches)
        findings.extend(f)
    status = "ok"
    if any(x.get("severity") == "ko" for x in findings):
        status = "ko"
    elif findings or all_patches:
        status = "review"
    return MultiBlockSolveResult(status, all_patches, regions, findings)


def apply_layout_patches_in_place(contract: Any, result: MultiBlockSolveResult) -> Any:
    for block in getattr(contract, "blocks", []) or []:
        bid = _block_id(block)
        patch = result.patches_by_block_id.get(bid)
        if not patch:
            continue
        layout = getattr(block, "layout", None)
        if layout is None:
            continue
        nb = [float(x) for x in patch.new_bbox]
        if hasattr(layout, "layout_bbox"):
            setattr(layout, "layout_bbox", nb)
        if hasattr(layout, "safe_bbox"):
            setattr(layout, "safe_bbox", nb)
    if hasattr(contract, "findings"):
        contract.findings.extend(result.findings)
    return contract
