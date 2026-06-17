"""PlacementSolver — choisit le meilleur candidat de rendu SANS collision, AVANT
le rendu (reprend `PlacementResult` legacy). Génère les candidats (normal →
shrink → interligne compact → shift local selon les libertés du LayoutContract),
les fait scorer par le CandidateEngine, retient le meilleur VALIDE. Aucun valide
→ review (le bloc n'est pas peint au hasard).
"""

from __future__ import annotations

from .candidate_engine import RenderCandidate, score_candidate
from .text_measure import measure_block


def _candidates(block, unit, renderer):
    layout = block.layout
    style = unit.get("style") or {}
    base_size = float(style.get("font_size_pt") or 10.0)
    align = (style.get("alignment") if style.get("alignment") in {"left", "center", "right"} else "left")
    bbox = unit.get("layout_bbox") or unit.get("coverage_bbox") or unit.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return

    def mk(strategy, b, st):
        lay = measure_block(unit["translated_text"], b, st, align=align)
        if lay.get("lines"):
            yield RenderCandidate(strategy, lay, lay.get("text_bbox"), base_size)

    yield from mk("normal", bbox, style)
    if layout.allow_shrink:
        for frac in (0.07, layout.max_shrink):
            st = dict(style); st["font_size_pt"] = base_size * (1.0 - frac)
            yield from mk(f"shrink_{int(frac*100)}", bbox, st)
    if layout.allow_local_shift:
        for dy in (8.0, -8.0, 16.0):
            yield from mk(f"shift_{int(dy)}", [bbox[0], bbox[1] + dy, bbox[2], bbox[3] + dy], style)


def solve_block(block, unit, renderer, *, protected_boxes, placed_boxes,
                forbidden_protected=0.01, forbidden_text=0.10):
    """Retourne (lay, strategy, status, findings)."""
    best, best_val = None, -1.0
    seen = False
    for cand in _candidates(block, unit, renderer):
        seen = True
        s = score_candidate(cand, protected_boxes=protected_boxes, placed_boxes=placed_boxes,
                            forbidden_protected=forbidden_protected, forbidden_text=forbidden_text)
        # un candidat valide bat toujours un invalide
        rank = (1.0 if s.valid else 0.0) + s.value
        if rank > best_val:
            best, best_val = cand, rank
    if not seen or best is None:
        return None, "none", "review", [{"type": "no_render_candidate", "unit_id": block.block_id, "severity": "review"}]
    findings = []
    if not best.score.valid:
        findings.append({"type": "placement_unresolved", "unit_id": block.block_id,
                         "causes": best.score.hard_failures, "strategy": best.strategy, "severity": "review"})
        return best.lay, best.strategy, "review", findings
    if best.strategy != "normal":
        findings.append({"type": "placement_adjusted", "unit_id": block.block_id,
                         "strategy": best.strategy, "severity": "info"})
    return best.lay, best.strategy, "ok", findings
