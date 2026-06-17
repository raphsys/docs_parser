"""
OCR Typography Engine — external, non-invasive module.

Purpose:
    Upgrade OCR/scanned-page typography from crude line-height metrics to a
    stable font-em estimate and a page/document style ladder.

Inputs:
    - pageprint units/regions/style_system/visual_layers/assets
    - pagetranslate reconstruction_units/contracts
    - pagereconstruct FinalReconstructionContract blocks

Output:
    TypographyEnhancementResult with per-block style overrides.

Integration:
    contract = enhance_contract_typography(contract, pageprint_data, page_image_path)

No mandatory OpenCV dependency. If cv2 is present, connected component analysis
can be added in _measure_crop_components(). The default implementation is safe
and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None


def measure_glyph_metrics_px(gray_crop):
    """(cap_height_px, x_height_px) depuis les composantes connexes d'un crop
    binarisé (encre = sombre). (None, None) si pas assez de glyphes."""
    if cv2 is None or np is None or gray_crop is None or getattr(gray_crop, "size", 0) == 0:
        return None, None
    _, binimg = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    h_img = gray_crop.shape[0]
    heights = []
    for i in range(1, n):
        w, h, area = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]), int(stats[i, cv2.CC_STAT_AREA])
        if area < 3 or h < 2 or h > h_img * 1.3 or w > h_img * 8:
            continue
        heights.append(float(h))
    if len(heights) >= 2:
        heights.sort()
        x_px = median(heights)
        cap_px = heights[int(0.85 * (len(heights) - 1))]
        return cap_px, x_px
    # Fallback profil d'encre (lignes denses où les glyphes fusionnent) : la
    # bande d'encre = hauteur cap-to-baseline ; les rangées denses ≈ x-height.
    rows = (binimg > 0).sum(axis=1)
    if rows.max() <= 0:
        return None, None
    ink = np.where(rows > 0)[0]
    if ink.size < 3:
        return None, None
    cap_px = float(ink[-1] - ink[0] + 1)
    dense = np.where(rows > 0.5 * rows.max())[0]
    x_px = float(dense[-1] - dense[0] + 1) if dense.size >= 2 else cap_px * 0.62
    return cap_px, x_px

BBox = Tuple[float, float, float, float]


@dataclass
class TypographyEvidence:
    block_id: str
    role: str
    source_bbox: BBox
    line_height_pt: Optional[float] = None
    glyph_height_pt: Optional[float] = None
    cap_height_pt: Optional[float] = None
    x_height_pt: Optional[float] = None
    baseline_y_pt: Optional[float] = None
    font_family_hint: Optional[str] = None
    font_class_hint: str = "unknown"
    raw_font_size_pt: Optional[float] = None
    source: str = "contract+geometry"
    confidence: float = 0.0
    findings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResolvedOcrTypography:
    block_id: str
    role: str
    font_size_pt_em: float
    line_height_pt: float
    font_class: str
    font_family_normalized: Optional[str]
    confidence: float
    method: str
    findings: List[Dict[str, Any]] = field(default_factory=list)

    def to_style_patch(self) -> Dict[str, Any]:
        return {
            "font_size_pt": round(self.font_size_pt_em, 3),
            "line_height": round(self.line_height_pt, 3),
            "font_class": self.font_class,
            "font_family_normalized": self.font_family_normalized,
            "size_source": "ocr_em_estimator",
            "confidence": round(self.confidence, 3),
            "typography_method": self.method,
            "findings": list(self.findings),
        }


@dataclass
class TypographyEnhancementResult:
    page_score: float
    patches_by_block_id: Dict[str, Dict[str, Any]]
    resolved: List[ResolvedOcrTypography]
    findings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_score": self.page_score,
            "patches_by_block_id": self.patches_by_block_id,
            "resolved": [asdict(r) for r in self.resolved],
            "findings": self.findings,
        }


# Conservative ratios. They are deliberately broad; page-level clustering
# stabilizes them.
_FONT_CLASS_RATIOS = {
    "serif": {"cap_to_em": 0.68, "x_to_em": 0.48, "line_to_em": 1.18},
    "sans": {"cap_to_em": 0.70, "x_to_em": 0.52, "line_to_em": 1.18},
    "mono": {"cap_to_em": 0.69, "x_to_em": 0.50, "line_to_em": 1.15},
    "unknown": {"cap_to_em": 0.69, "x_to_em": 0.50, "line_to_em": 1.18},
}

_ROLE_BOUNDS = {
    "body_paragraph": (7.0, 14.0), "paragraph": (7.0, 14.0), "list_item": (7.0, 14.0),
    "author_bio": (7.0, 14.0), "bibliography_entry": (6.5, 12.5),
    "index_entry": (6.0, 11.5), "index_subentry": (5.8, 11.0),
    "table_body_cell": (5.5, 12.0), "table_header_cell": (5.5, 12.0),
    "diagram_label": (4.5, 12.0), "axis_label": (4.5, 12.0),
    "title": (9.0, 32.0), "section_heading": (8.0, 28.0), "heading": (8.0, 28.0),
}


def bbox_height(b: Any) -> Optional[float]:
    if isinstance(b, (list, tuple)) and len(b) == 4:
        h = float(b[3]) - float(b[1])
        return h if h > 0 else None
    return None


def normalize_font_class(font_family: Optional[str], fallback: str = "unknown") -> str:
    f = (font_family or "").lower()
    if any(k in f for k in ("mono", "courier", "consol", "menlo", "code", "ubuntumono")):
        return "mono"
    if any(k in f for k in ("sans", "arial", "helvet", "calibri", "verdana", "franklin", "gothic")):
        return "sans"
    if any(k in f for k in ("times", "janson", "baskerville", "garamond", "minion", "georgia", "serif")):
        return "serif"
    return fallback if fallback in {"serif", "sans", "mono"} else "unknown"


def _clamp(v: float, role: str) -> float:
    # Défaut PRUDENT corps de texte : un rôle non reconnu ne doit JAMAIS pouvoir
    # recevoir une taille de titre (32pt) depuis une estimation cap-height
    # aberrante. Seuls les rôles titre/heading explicites autorisent les grandes
    # tailles (cf. _ROLE_BOUNDS).
    lo, hi = _ROLE_BOUNDS.get(role, (5.0, 16.0))
    return max(lo, min(hi, v))


def _robust_style_ladder(values: List[ResolvedOcrTypography]) -> Dict[str, float]:
    """Build a page-level size ladder by role to remove per-block OCR jitter."""
    by_role: Dict[str, List[float]] = {}
    for v in values:
        by_role.setdefault(v.role, []).append(v.font_size_pt_em)
    ladder: Dict[str, float] = {}
    for role, sizes in by_role.items():
        if not sizes:
            continue
        ladder[role] = median(sizes)
    # enforce common body/list consistency
    body_roles = [r for r in ("body_paragraph", "paragraph", "list_item") if r in ladder]
    if body_roles:
        body = median([ladder[r] for r in body_roles])
        for r in body_roles:
            ladder[r] = body
    return ladder


def collect_evidence_from_contract(contract: Any, *, gray_image=None, scale: float = 1.0) -> List[TypographyEvidence]:
    evidence: List[TypographyEvidence] = []
    H, W = (gray_image.shape[:2] if gray_image is not None else (0, 0))
    for block in getattr(contract, "blocks", []) or []:
        style = getattr(block, "style", None)
        layout = getattr(block, "layout", None)
        bbox = getattr(layout, "source_bbox", None) or getattr(layout, "layout_bbox", None)
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        line_h = bbox_height(bbox)
        family = getattr(style, "font_family", None) if style else None
        raw_size = getattr(style, "font_size_pt", None) if style else None
        font_class = getattr(style, "font_class", None) if style else None
        cap_pt = x_pt = None
        # Mesure réelle cap/x-height par composantes connexes sur le crop image.
        if gray_image is not None and scale:
            x0, y0, x1, y1 = (int(bbox[0] * scale), int(bbox[1] * scale), int(bbox[2] * scale), int(bbox[3] * scale))
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 > 3 and y1 - y0 > 3:
                cap_px, x_px = measure_glyph_metrics_px(gray_image[y0:y1, x0:x1])
                if cap_px and x_px:
                    cap_pt, x_pt = cap_px / scale, x_px / scale
        evidence.append(TypographyEvidence(
            block_id=str(getattr(block, "block_id", "")),
            role=str(getattr(block, "role", "")),
            source_bbox=tuple(float(x) for x in bbox),
            line_height_pt=line_h,
            glyph_height_pt=None if (cap_pt or x_pt) else (line_h * 0.78 if line_h else None),
            cap_height_pt=cap_pt, x_height_pt=x_pt,
            font_family_hint=family,
            font_class_hint=normalize_font_class(family, font_class or "unknown"),
            raw_font_size_pt=float(raw_size) if raw_size else None,
            confidence=0.45 if line_h else 0.20,
        ))
    return evidence


def resolve_ocr_typography(e: TypographyEvidence) -> ResolvedOcrTypography:
    font_class = normalize_font_class(e.font_family_hint, e.font_class_hint)
    ratios = _FONT_CLASS_RATIOS.get(font_class, _FONT_CLASS_RATIOS["unknown"])
    findings: List[Dict[str, Any]] = []
    estimates: List[Tuple[str, float, float]] = []  # method, value, weight

    if e.cap_height_pt:
        estimates.append(("cap_height", e.cap_height_pt / ratios["cap_to_em"], 0.45))
    if e.x_height_pt:
        estimates.append(("x_height", e.x_height_pt / ratios["x_to_em"], 0.35))
    if e.glyph_height_pt:
        # glyph height is usually between x-height and cap-height; conservative em conversion
        estimates.append(("glyph_height", e.glyph_height_pt / 0.72, 0.30))
    if e.line_height_pt:
        estimates.append(("line_height", e.line_height_pt / ratios["line_to_em"], 0.25))
    if e.raw_font_size_pt and e.raw_font_size_pt >= 6.0:
        estimates.append(("raw_extracted", e.raw_font_size_pt, 0.30))
    elif e.raw_font_size_pt:
        findings.append({"type": "raw_font_size_probably_metric_not_em", "raw": round(e.raw_font_size_pt, 3)})

    if not estimates:
        size = _clamp(9.5, e.role)
        confidence = 0.15
        method = "fallback_role_default"
        findings.append({"type": "no_typography_evidence"})
    else:
        total_w = sum(w for _, _, w in estimates)
        size = sum(v * w for _, v, w in estimates) / total_w
        size = _clamp(size, e.role)
        confidence = min(0.95, 0.25 + sum(w for _, _, w in estimates))
        method = "+".join(m for m, _, _ in estimates)

    line_height = e.line_height_pt or size * 1.18
    if line_height < size * 1.05:
        line_height = size * 1.15
        findings.append({"type": "line_height_adjusted"})

    return ResolvedOcrTypography(
        block_id=e.block_id,
        role=e.role,
        font_size_pt_em=round(size, 3),
        line_height_pt=round(line_height, 3),
        font_class=font_class if font_class != "unknown" else "serif",
        font_family_normalized=e.font_family_hint,
        confidence=round(confidence, 3),
        method=method,
        findings=findings,
    )


def enhance_contract_typography(contract: Any, *, pageprint_data: Optional[dict] = None,
                                page_image_path: Optional[str] = None,
                                min_confidence_for_ok: float = 0.70) -> TypographyEnhancementResult:
    """Return style patches. Does not mutate by default.

    Integration code may apply patches to block.style.font_size_pt / line_height.
    Keeping this module external makes it safe to disable.
    """
    gray, scale = None, 1.0
    if page_image_path and cv2 is not None:
        try:
            img = cv2.imread(page_image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                gray = img
                pi = getattr(contract, "page_info", None)
                dpi = float(getattr(pi, "dpi_reference", 0) or 0) if pi else 0.0
                ps = getattr(pi, "page_size", None) if pi else None
                if ps and ps[0]:
                    scale = img.shape[1] / float(ps[0])      # px / pt
                elif dpi:
                    scale = dpi / 72.0
        except Exception:
            gray = None
    evidence = collect_evidence_from_contract(contract, gray_image=gray, scale=scale)
    resolved = [resolve_ocr_typography(e) for e in evidence]
    ladder = _robust_style_ladder(resolved)
    findings: List[Dict[str, Any]] = []

    patched: List[ResolvedOcrTypography] = []
    for r in resolved:
        if r.role in ladder:
            stable = _clamp(ladder[r.role], r.role)
            if abs(stable - r.font_size_pt_em) > 0.6:
                r.findings.append({"type": "page_style_ladder_stabilized", "from": r.font_size_pt_em, "to": round(stable, 3)})
                r.font_size_pt_em = round(stable, 3)
        patched.append(r)

    patches_by_id = {r.block_id: r.to_style_patch() for r in patched if r.block_id}
    if not patched:
        page_score = 0.0
        findings.append({"type": "no_blocks_for_typography_enhancement", "severity": "review"})
    else:
        page_score = sum(r.confidence for r in patched) / len(patched)
        if page_score < min_confidence_for_ok:
            findings.append({"type": "ocr_typography_confidence_below_ok_threshold", "score": round(page_score, 3), "severity": "review"})
    return TypographyEnhancementResult(round(page_score, 3), patches_by_id, patched, findings)


def apply_typography_patches_in_place(contract: Any, result: TypographyEnhancementResult) -> Any:
    """Optional mutator for integration tests / adapter.

    Prefer immutable copy in production; this is kept simple for compatibility.
    """
    for block in getattr(contract, "blocks", []) or []:
        patch = result.patches_by_block_id.get(str(getattr(block, "block_id", "")))
        if not patch:
            continue
        style = getattr(block, "style", None)
        if style is None:
            continue
        # NE PAS écraser font_size_pt (taille de RENDU = ajustée à la boîte) : grossir
        # le texte recréerait des collisions. On enregistre l'em comme MÉTADONNÉE de
        # fidélité (taille source connue de façon fiable) + la classe de police.
        setattr(style, "font_size_pt_em", float(patch.get("font_size_pt") or 0.0))
        if patch.get("font_class"):
            setattr(style, "font_class", str(patch["font_class"]))
        if hasattr(style, "source"):
            setattr(style, "source", "ocr_em_estimator")
        if hasattr(style, "confidence"):
            setattr(style, "confidence", float(patch.get("confidence") or 0.0))
        setattr(style, "typo_method", str(patch.get("typography_method") or ""))
    contract.findings.extend(result.findings) if hasattr(contract, "findings") else None
    return contract
