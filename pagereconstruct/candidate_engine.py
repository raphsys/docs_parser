"""CandidateEngine — génère et score des candidats de rendu d'un bloc.

Reprend `RenderCandidate` / `CandidateScore` legacy. Six scores nommés (directive
Phase 7) : text_fit / style_similarity / position / collision / readability /
preservation. Un candidat est INVALIDE s'il chevauche une zone protégée dure ou
si sa police devient illisible. Le PlacementSolver choisit le meilleur valide.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _inter(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _area(b) -> float:
    return max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def _ratio(box, region) -> float:
    return _inter(box, region) / _area(box)


@dataclass
class CandidateScore:
    text_fit: float = 1.0
    style_similarity: float = 1.0
    position: float = 1.0
    collision: float = 1.0
    readability: float = 1.0
    preservation: float = 1.0
    hard_failures: list = field(default_factory=list)

    @property
    def value(self) -> float:
        return (self.text_fit + self.style_similarity + self.position
                + self.collision + self.readability + self.preservation) / 6.0

    @property
    def valid(self) -> bool:
        return not self.hard_failures


@dataclass
class RenderCandidate:
    strategy: str
    lay: dict
    text_bbox: list | None
    base_size: float
    score: CandidateScore = field(default_factory=CandidateScore)


def score_candidate(cand: RenderCandidate, *, protected_boxes, placed_boxes,
                    forbidden_protected=0.01, forbidden_text=0.10,
                    min_readable_pt=6.0) -> CandidateScore:
    s = CandidateScore()
    tb = cand.text_bbox
    if not tb:
        s.hard_failures.append("no_geometry")
        return s
    prot = max((_ratio(tb, r) for r in protected_boxes), default=0.0)
    txt = max((_ratio(tb, p) for p in placed_boxes), default=0.0)
    s.collision = max(0.0, 1.0 - prot - txt)
    s.preservation = max(0.0, 1.0 - prot)
    if prot > forbidden_protected:
        s.hard_failures.append(f"protected_overlap={prot:.2f}")
    if txt > forbidden_text:
        s.hard_failures.append(f"text_overlap={txt:.2f}")
    size = float(cand.lay.get("size") or 0)
    if size < min_readable_pt:
        s.hard_failures.append(f"font_too_small={size:.1f}")
        s.readability = 0.0
    else:
        s.readability = min(1.0, size / max(1.0, cand.base_size))
    if cand.lay.get("overflow"):
        s.text_fit = 0.5
    # style_similarity: shrink pénalise (taille s'éloigne de la source)
    s.style_similarity = max(0.0, min(1.0, size / max(1.0, cand.base_size)))
    s.position = 0.9 if cand.strategy.startswith("shift") else 1.0
    cand.score = s
    return s
