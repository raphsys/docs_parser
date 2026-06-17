"""Compatibility wrapper for previous spacing-reflow patch.

If plan_compiler imports solve_spacing_reflow/apply_spacing_reflow_patches_in_place,
route them to the new v2 flow geometry optimizer.
"""

from __future__ import annotations

from .flow_geometry_optimizer import solve_flow_geometry, apply_flow_geometry_patches_in_place


def solve_spacing_reflow(contract, *, enabled: bool = True):
    return solve_flow_geometry(contract, enabled=enabled)


def apply_spacing_reflow_patches_in_place(contract, result):
    return apply_flow_geometry_patches_in_place(contract, result)
