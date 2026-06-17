"""Audit typographie GRANULAIRE: bloc → phrase, chaque dimension (classe de
police, taille, couleur, gras, italique, alignement) comparée à l'origine
pageprint, individuellement puis combinée."""

from __future__ import annotations

from ..schema import StageAuditResult, ElementAudit, DimensionScore, Finding, OK, REVIEW, KO
from ..evidence import index_units, source_style, source_text, font_class_of


def _hex(c, default=(0, 0, 0)):
    s = str(c or "").lstrip("#")
    if len(s) == 6:
        try:
            return tuple(int(s[k:k + 2], 16) for k in (0, 2, 4))
        except ValueError:
            pass
    return default


def _recon_style(unit: dict) -> dict:
    s = unit.get("style") or {}
    flags = s.get("flags") or {}
    return {
        "font_class": s.get("font_class") or ("mono" if flags.get("monospace") else "sans" if flags.get("serif") is False else "serif"),
        "font_size_pt": s.get("font_size_pt"),
        "color": s.get("color") or "#000000",
        "bold": bool(flags.get("bold")), "italic": bool(flags.get("italic")),
        "alignment": s.get("alignment") or "left",
        "size_source": s.get("size_source"),
    }


def _dim_font_class(src_cls, rec_cls) -> DimensionScore:
    if not src_cls:
        return DimensionScore("font_class", 0.85, src_cls, rec_cls, REVIEW, weight=1.0, finding="source_font_class_unknown")
    if src_cls == rec_cls:
        return DimensionScore("font_class", 1.0, src_cls, rec_cls, OK, 1.0)
    # serif<->mono = grave ; serif<->sans = mineur
    grave = {src_cls, rec_cls} & {"mono"}
    return DimensionScore("font_class", 0.0 if grave else 0.6, src_cls, rec_cls,
                          KO if grave else REVIEW, 1.0, "font_class_mismatch")


def _dim_size(src_pt, rec_pt, size_source) -> DimensionScore:
    if not src_pt or not rec_pt:
        # source manquante: si résolue depuis l'image (em) on fait confiance, sinon neutre.
        ok_em = size_source == "ocr_em_estimator"
        return DimensionScore("font_size", 0.9 if ok_em else 0.8, src_pt, rec_pt,
                              OK if ok_em else REVIEW, 1.2, None if ok_em else "source_size_unreliable")
    ratio = rec_pt / src_pt if src_pt else 1.0
    d = abs(1.0 - ratio)
    score = 1.0 if d <= 0.12 else max(0.0, 1.0 - (d - 0.12) * 2)
    return DimensionScore("font_size", round(score, 3), round(src_pt, 2), round(rec_pt, 2),
                          OK if score >= 0.95 else REVIEW if score >= 0.7 else KO, 1.2,
                          None if score >= 0.95 else "font_size_drift")


def _dim_color(src_c, rec_c) -> DimensionScore:
    a, b = _hex(src_c), _hex(rec_c)
    delta = sum(abs(a[i] - b[i]) for i in range(3))
    score = 1.0 if delta <= 24 else max(0.0, 1.0 - (delta - 24) / 400)
    return DimensionScore("color", round(score, 3), src_c, rec_c,
                          OK if score >= 0.95 else REVIEW if score >= 0.7 else KO, 0.8,
                          None if score >= 0.95 else "color_drift")


def _dim_flag(name, src, rec, weight=0.7) -> DimensionScore:
    return DimensionScore(name, 1.0 if src == rec else 0.5, src, rec,
                          OK if src == rec else REVIEW, weight, None if src == rec else f"{name}_mismatch")


def _dim_align(src, rec) -> DimensionScore:
    if not src:
        return DimensionScore("alignment", 0.9, src, rec, OK, 0.6)
    return DimensionScore("alignment", 1.0 if src == rec else 0.8, src, rec,
                          OK if src == rec else REVIEW, 0.6, None if src == rec else "alignment_mismatch")


def _audit_against(src: dict, rec: dict, *, level, eid, role, stext, ttext) -> ElementAudit:
    dims = [
        _dim_font_class(src.get("font_class") or font_class_of(src.get("font_family"), src.get("mono")), rec["font_class"]),
        _dim_size(src.get("font_size_pt"), rec.get("font_size_pt"), rec.get("size_source")),
        _dim_color(src.get("color"), rec.get("color")),
        _dim_flag("bold", src.get("bold", False), rec.get("bold", False)),
        _dim_flag("italic", src.get("italic", False), rec.get("italic", False)),
        _dim_align(src.get("alignment"), rec.get("alignment")),
    ]
    el = ElementAudit(element_id=eid, level=level, role=role, source_text=stext, translated_text=ttext, dimensions=dims)
    el.combine()
    return el


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    units = index_units(normalized)
    res = StageAuditResult(stage_name="typography")
    blocks = (plan.get("layers") or {}).get("translated_text") or []
    if not blocks:
        res.score, res.status = 1.0, OK
        return res
    block_audits = []
    for b in blocks:
        rec = _recon_style(b)
        sids = b.get("source_unit_ids") or []
        srcs = [units[s] for s in sids if s in units]
        # dominant source style (1er non vide) pour le bloc
        dom = source_style(srcs[0]) if srcs else {}
        dom["font_class"] = font_class_of(dom.get("font_family"), dom.get("mono"))
        ba = _audit_against(dom, rec, level="block", eid=b.get("id"), role=b.get("role"),
                            stext=b.get("source_text") or "", ttext=b.get("translated_text") or "")
        # phrases (chaque unité source comparée au style appliqué du bloc)
        for s in srcs:
            ss = source_style(s); ss["font_class"] = font_class_of(ss.get("font_family"), ss.get("mono"))
            ph = _audit_against(ss, rec, level="phrase", eid=s.get("unit_id"),
                                role=(s.get("understanding") or {}).get("role") or "",
                                stext=source_text(s), ttext="")
            ba.children.append(ph)
        ba.combine()
        block_audits.append(ba)
    res.elements = block_audits
    res.score = round(sum(b.score for b in block_audits) / len(block_audits), 3)

    # Hiérarchie de page : un heading doit être rendu >= corps (taille).
    def _rsize(b):
        return (_recon_style(b).get("font_size_pt") or 0)
    body = [_rsize(b) for b in blocks if (b.get("role") or "") in {"body_paragraph", "paragraph", "list_item"}]
    body_med = sorted(body)[len(body)//2] if body else 0
    if body_med:
        for b in blocks:
            if (b.get("role") or "") in {"title", "section_heading", "subsection_heading", "chapter_heading"}:
                if _rsize(b) and _rsize(b) < body_med:
                    res.findings.append(Finding(type="heading_rendered_as_body", severity=REVIEW,
                                                element_id=b.get("id"),
                                                detail={"heading": _rsize(b), "body": body_med}))
                    res.score = min(res.score, 0.92)

    if any(b.status == KO for b in block_audits):
        res.status = KO
    elif res.score < 0.95 or any(b.status == REVIEW for b in block_audits) or res.findings:
        res.status = REVIEW
    # findings synthétiques
    for b in block_audits:
        for d in b.dimensions:
            if d.status != OK and d.finding:
                res.findings.append(Finding(type=f"typo_{d.finding}", severity=d.status, element_id=b.element_id,
                                            detail={"expected": d.expected, "observed": d.observed}))
    if res.findings:
        res.findings = res.findings[:20]
    return res
