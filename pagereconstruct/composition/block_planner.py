"""BlockPlanner — étape ancienne 2 : placer les GRANDS blocs avant l'intra-bloc.

Réutilise le solveur multi-blocs existant (net-improvement gardé) et l'expose
comme une étape de composition explicite produisant un BlockPlacementPlan
(régions de flux, blocs verrouillés/mobiles, obstacles). Ne réécrit pas le
solveur : on porte/organise, on ne réinvente pas (cf. legacy registry
_build_block_reconstruction_plan → ADAPT).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..multiblock_layout_solver import (
    build_flow_regions, collect_protected_obstacles, _is_locked, _block_id, _block_bbox,
    solve_multiblock_layout, apply_layout_patches_in_place,
)


@dataclass
class BlockPlacementPlan:
    flow_regions: list = field(default_factory=list)      # [FlowRegion]
    placed_blocks: list = field(default_factory=list)     # [block_id]
    locked_blocks: list = field(default_factory=list)
    movable_blocks: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)         # [bbox]
    findings: list = field(default_factory=list)

    def to_dict(self):
        return {"flow_regions": [getattr(r, "region_id", "") for r in self.flow_regions],
                "placed": len(self.placed_blocks), "locked": self.locked_blocks,
                "movable": self.movable_blocks, "obstacles": len(self.obstacles),
                "findings": self.findings}


def plan_blocks(contract) -> BlockPlacementPlan:
    """Construit le plan de placement des grands blocs depuis le contrat."""
    blocks = getattr(contract, "blocks", []) or []
    regions = build_flow_regions(contract)
    obstacles = collect_protected_obstacles(contract)
    locked, movable, placed = [], [], []
    for b in blocks:
        bid = _block_id(b)
        if not bid or not _block_bbox(b):
            continue
        placed.append(bid)
        (locked if _is_locked(b) else movable).append(bid)
    return BlockPlacementPlan(flow_regions=regions, placed_blocks=placed,
                              locked_blocks=locked, movable_blocks=movable, obstacles=obstacles)


def solve_and_apply(contract):
    """Résout le placement multi-blocs (net-improvement) et applique au contrat.
    Retourne (BlockPlacementPlan, MultiBlockSolveResult)."""
    plan = plan_blocks(contract)
    mb = solve_multiblock_layout(contract, enabled=True)
    if mb.status != "ko" and mb.patches_by_block_id:
        apply_layout_patches_in_place(contract, mb)
    return plan, mb
