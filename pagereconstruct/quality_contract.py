"""QualityContract — seuils publication-ready (gates durs) repris/du­rcis.

Reprend `BlockRenderVerdict` (must_render / no clip / no overlap) au niveau page.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class QualityContract:
    text_presence_min: float = 1.0
    non_text_presence_min: float = 0.99
    overlap_min: float = 0.99
    position_min: float = 0.95
    typography_min: float = 0.95
    source_text_leak_min: float = 0.98
    publication_ready_min: float = 0.95
    # blockers durs
    require_clean_background: bool = True
    forbid_source_text_leak_high: bool = True
    forbid_patch_protected_overlap: bool = True
    forbid_missing_text: bool = True
    forbid_renderer_failure: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
