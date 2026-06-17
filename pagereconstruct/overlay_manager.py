"""OverlayManager — gère les objets préservés (immutable overlays legacy).

Classe underlays (sous le texte : figures, fonds, formules) vs overlays (au-dessus
: numéros de page, labels exacts), attribue le z_index, produit les PreservationOps,
et garantit qu'aucun patch ne détruit un objet préservé (les bbox préservées sont
des trous de patch). Reprend `_insert_immutable_overlays` / FormulaItem legacy.
"""

from __future__ import annotations

from .render_ops import PreservationOp



def _dedupe_ops(ops: list[PreservationOp]) -> list[PreservationOp]:
    seen = set()
    out = []
    for op in ops:
        bbox = op.bbox
        rb = tuple(round(float(x), 1) for x in bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None
        key = (str(op.text or ""), rb, tuple(sorted(str(s) for s in (op.source_unit_ids or []))))
        if key in seen:
            continue
        seen.add(key)
        out.append(op)
    return out

def preservation_boxes(contract) -> list:
    """Toutes les bbox préservées — à exclure des patches (patch non destructeur)."""
    return [o.bbox for o in contract.preservation.objects
            if isinstance(o.bbox, (list, tuple)) and len(o.bbox) == 4]


def build_preservation_ops(contract, *, source_path: str | None):
    """(underlay_ops, overlay_ops) prêts à exécuter, z_index ordonné."""
    under, over = [], []
    for o in contract.preservation.underlays:
        under.append(PreservationOp(bbox=o.bbox, method=o.method, source_path=source_path,
                                    text=o.text, source_unit_ids=list(getattr(o, "source_unit_ids", []) or []),
                                    z=20))
    for o in contract.preservation.overlays:
        over.append(PreservationOp(bbox=o.bbox, method=o.method, source_path=source_path,
                                   text=o.text, source_unit_ids=list(getattr(o, "source_unit_ids", []) or []),
                                   z=40))
    return _dedupe_ops(under), _dedupe_ops(over)
