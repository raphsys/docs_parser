"""Collect competing PAGEPRINT evidence claims from normalized units."""

from __future__ import annotations

import re
from collections import defaultdict

from .claim_model import make_claim


URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/[\w.-]+/|\.{1,2}/)[^\s]+")
TOC_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?\S.{3,}?(?:\.{2,}|\s{2,})\s*[ivxlcdm\d]+\s*$", re.IGNORECASE)
INDEX_RE = re.compile(r"^\s*[A-Za-z][^,]{1,80},\s*(?:\d+[,-–\s]*)+\s*$")
CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table|tab\.)\s+\d+(?:[.-]\d+)?\s*[:.-]?\s+\S+", re.IGNORECASE)
SECTION_NUMBER_RE = re.compile(r"^\s\d*(?:\d+\.)+\d*\.?\s*$")
PAGE_REF_RE = re.compile(r"^\s*(?:[ivxlcdm]+|\d+)(?:[-–]\d+)?\s*$", re.IGNORECASE)
ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9&./+-]{1,12}$")
COMMAND_RE = re.compile(r"^(?:copy|dir|del|findstr|mkdir|rmdir|cd|ls|cat|grep|sudo|docker|npm|pip|python|git)\b", re.IGNORECASE)


def collect_claims(
    units: list[dict],
    regions: list[dict],
    page_intelligence: dict | None = None,
) -> dict:
    """Collect claims and attach them to units in-place.

    Returns a page-level evidence bundle:
    ``{"claims": [...], "claims_by_unit": {...}, "region_claims": [...]}``.
    """
    claims: list[dict] = []
    region_claims = _region_claims(regions)
    claims.extend(region_claims)
    claims_by_unit: dict[str, list[dict]] = defaultdict(list)
    region_claims_by_id = {
        claim.get("evidence", {}).get("region_id"): claim
        for claim in region_claims
        if claim.get("evidence", {}).get("region_id")
    }

    for unit in units:
        if not isinstance(unit, dict) or not unit.get("unit_id"):
            continue
        unit_claims = _unit_text_claims(unit, page_intelligence or {})
        unit_claims.extend(_unit_region_membership_claims(unit, region_claims_by_id))
        for claim in unit_claims:
            claims.append(claim)
            claims_by_unit[unit["unit_id"]].append(claim)
        unit.setdefault("evidence", {})["claims"] = unit_claims

    return {
        "schema_version": "pageprint.evidence_claims.v1",
        "claims": claims,
        "claims_by_unit": dict(claims_by_unit),
        "region_claims": region_claims,
    }


def _region_claims(regions: list[dict]) -> list[dict]:
    claims = []
    for region in regions or []:
        region_type = str(region.get("region_type") or "")
        object_type = str(region.get("object_type") or region.get("object_class") or region_type)
        claim_type = _claim_type_for_region(region_type, object_type)
        if not claim_type:
            continue
        claims.append(make_claim(
            source=region.get("source") or region.get("detection_source") or "region_index",
            target_unit_id=None,
            claim_type=claim_type,
            value=object_type,
            confidence=region.get("confidence") or 0.5,
            reason="region_observation",
            evidence={
                "region_id": region.get("region_id"),
                "region_type": region_type,
                "object_type": object_type,
            },
            bbox=region.get("bbox"),
        ))
    return claims


def _claim_type_for_region(region_type: str, object_type: str) -> str | None:
    combined = f"{region_type} {object_type}".lower()
    if "formula" in combined or "equation" in combined or "math" in combined:
        return "formula_candidate"
    if "code" in combined or "algorithm" in combined:
        return "code_candidate"
    if "table" in combined:
        return "table_candidate"
    if "toc" in combined:
        return "toc_candidate"
    if "index" in combined:
        return "index_candidate"
    if "caption" in combined:
        return "caption_candidate"
    if "watermark" in combined:
        return "watermark"
    if "publisher" in combined or "logo" in combined:
        return "publisher_mark_candidate"
    return None


def _unit_text_claims(unit: dict, page_intelligence: dict) -> list[dict]:
    text = str((unit.get("content") or {}).get("text") or "").strip()
    if not text:
        return []
    unit_id = unit["unit_id"]
    understanding = unit.get("understanding") or {}
    extraction = unit.get("extraction") or {}
    role = str(understanding.get("role") or "").lower()
    level = unit.get("level")
    source = extraction.get("source") or extraction.get("source_kind") or "heuristic"
    base_conf = extraction.get("confidence") or 0.7
    claims = [
        make_claim(
            source=source,
            target_unit_id=unit_id,
            claim_type="natural_text" if _looks_natural_text(text) else "text_observation",
            value=text,
            confidence=base_conf,
            reason="text_extraction",
            evidence={"level": level, "role": role},
            bbox=(unit.get("geometry") or {}).get("bbox"),
        )
    ]

    page_role = str(page_intelligence.get("page_role") or "").lower()
    if page_role == "toc" or TOC_RE.match(text) or role.startswith("toc"):
        claims.append(_simple_claim(unit, "toc_candidate", text, 0.82, "toc_pattern_or_context"))
    if page_role == "index" or INDEX_RE.match(text) or role.startswith("index"):
        claims.append(_simple_claim(unit, "index_candidate", text, 0.82, "index_pattern_or_context"))
    if CAPTION_RE.match(text) or "caption" in role:
        claims.append(_simple_claim(unit, "caption_candidate", text, 0.86, "caption_pattern_or_role"))
    if PATH_RE.search(text):
        claims.append(_simple_claim(unit, "file_path", text, 0.94, "path_pattern"))
    if URL_RE.fullmatch(text):
        claims.append(_simple_claim(unit, "url", text, 0.98, "url_pattern"))
    if EMAIL_RE.fullmatch(text):
        claims.append(_simple_claim(unit, "email", text, 0.98, "email_pattern"))
    if ACRONYM_RE.fullmatch(text):
        claims.append(_simple_claim(unit, "acronym", text, 0.85, "acronym_pattern"))
    # A numeric/roman token is not a page reference by syntax alone.  In
    # particular, PDF spans split headings such as "CHAPTER 7" into "C" and
    # "7"; treating those spans as page references creates exact-preservation
    # ops on top of the translated heading.  Page references are autonomous
    # line/phrase units and require page/role/position context.
    if (
        PAGE_REF_RE.fullmatch(text)
        and level in {"phrase", "line"}
        and _has_page_reference_context(unit, page_intelligence, page_role, role)
    ):
        claims.append(_simple_claim(unit, "page_reference", text, 0.78, "page_reference_pattern"))
    if SECTION_NUMBER_RE.fullmatch(f" {text} "):
        claims.append(_simple_claim(unit, "section_number", text, 0.82, "section_number_pattern"))
    if COMMAND_RE.match(text):
        claims.append(_simple_claim(unit, "command_name", text.split()[0], 0.86, "command_pattern"))
    if _formula_score(text) >= 4:
        claims.append(_simple_claim(unit, "formula_candidate", text, min(0.95, 0.45 + _formula_score(text) / 10), "formula_score"))
    if _code_score(text, unit) >= 4:
        claims.append(_simple_claim(unit, "code_candidate", text, min(0.95, 0.45 + _code_score(text, unit) / 10), "code_score"))
    return claims


def _unit_region_membership_claims(unit: dict, region_claims_by_id: dict[str, dict]) -> list[dict]:
    claims = []
    for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
        base = region_claims_by_id.get(membership.get("region_id"))
        if not base:
            continue
        overlap = float(membership.get("overlap_ratio") or 0.0)
        evidence = dict(base.get("evidence") or {})
        evidence.update({
            "overlap_ratio": overlap,
            "coverage_mode": membership.get("coverage_mode"),
            "membership_role": membership.get("membership_role"),
        })
        claims.append(make_claim(
            source=base.get("source") or "region_index",
            target_unit_id=unit.get("unit_id"),
            claim_type=base.get("claim_type"),
            value=base.get("value"),
            confidence=(base.get("confidence") or 0.5) * max(0.2, overlap),
            reason="region_membership_observation",
            evidence=evidence,
            bbox=(unit.get("geometry") or {}).get("bbox"),
        ))
    return claims


def _simple_claim(unit: dict, claim_type: str, value: str, confidence: float, reason: str) -> dict:
    return make_claim(
        source="pageprint.evidence.collector",
        target_unit_id=unit.get("unit_id"),
        claim_type=claim_type,
        value=value,
        confidence=confidence,
        reason=reason,
        evidence={"level": unit.get("level")},
        bbox=(unit.get("geometry") or {}).get("bbox"),
    )


def _has_page_reference_context(
    unit: dict,
    page_intelligence: dict,
    page_role: str,
    role: str,
) -> bool:
    if page_role in {"toc", "index"} or "page_reference" in role or role == "page_number":
        return True
    bbox = (unit.get("geometry") or {}).get("bbox")
    page_geometry = page_intelligence.get("page_geometry") or {}
    height = float(page_geometry.get("height") or 0.0)
    dpi = float(page_geometry.get("render_dpi") or 72.0)
    if height and dpi:
        height = height * 72.0 / dpi
    if not height or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    y0, y1 = float(bbox[1]), float(bbox[3])
    return y1 < height * 0.12 or y0 > height * 0.88


def _looks_natural_text(text: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
    if len(words) >= 4:
        return True
    return bool(len(words) >= 2 and re.search(r"[.!?;:,]", text))


def _formula_score(text: str) -> int:
    score = 0
    if re.search(r"[∑∫√∞≈≠≤≥±×÷∂∆λµπσΔαβγ]", text):
        score += 4
    if re.search(r"\w\s*(?:=|≈|<=|>=|≤|≥)\s*\w", text):
        score += 3
    if len(re.findall(r"[=+\*/^<>≤≥≈]", text)) >= 2:
        score += 2
    if len(re.findall(r"[A-Za-z]{3,}", text)) > 5:
        score -= 3
    if re.fullmatch(r"\([^)]{1,40}\)", text):
        score -= 3
    return score


def _code_score(text: str, unit: dict) -> int:
    style = (unit.get("visual") or {}).get("style") or {}
    font = str(style.get("font_family") or style.get("font") or "").lower()
    score = 0
    if "mono" in font or "courier" in font or "consolas" in font:
        score += 3
    if re.search(r"(==|!=|:=|=>|->|[{};])", text):
        score += 2
    if re.search(r"\b(def|class|import|return|function|SELECT|FROM|WHERE)\b", text):
        score += 2
    if PATH_RE.search(text) or COMMAND_RE.match(text):
        score += 3
    if len(re.findall(r"[A-Za-z]{3,}", text)) >= 8 and re.search(r"\s", text):
        score -= 3
    return score
