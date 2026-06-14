"""Compile translated_input_data into a PageRenderPlan (no PDF rendering).

Merges the four views (reconstruction_units + reconstruction/preservation/
exclusion plans) plus units geometry and visual layers, computing consumed /
excluded source ids and the protected-region index, while forbidding
parent/child double rendering (directive §5, §6, §17.1).
"""

from __future__ import annotations

from .background_resolver import resolve_background
from .input_adapter import PageReconstructInputAdapter
from .layout_box_resolver import resolve_layout
from .patch_planner import plan_patches
from .protected_region_index import build_protected_region_index
from .schema import PageRenderPlan, PreservedUnit, TranslatedTextUnit
from .style_resolver import resolve_style

RENDERER_BY_ROLE = {
    "body_paragraph": "paragraph", "list_item": "paragraph", "author_bio": "paragraph",
    "index_subentry": "paragraph", "formula_explanation": "paragraph",
    "title": "heading", "subtitle": "heading", "section_heading": "heading",
    "subsection_heading": "heading", "chapter_heading": "heading",
    "figure_caption": "caption", "figure_caption_text": "caption",
    "table_caption": "caption", "table_caption_text": "caption",
    "table_header_cell": "table", "table_body_cell": "table", "table_numeric_cell": "table",
    "toc_entry_title": "anchored_label", "toc_entry": "anchored_label",
    "index_entry": "anchored_label", "index_head_term": "anchored_label",
    "diagram_label": "anchored_label", "diagram_text_label": "anchored_label",
    "axis_label": "anchored_label", "legend_label": "anchored_label",
    "code_line": "code", "code_block": "code",
    "formula_expression": "formula",
}

DEFAULT_RENDER_POLICY = {
    "fail_on_missing_reconstruction_units": True,
    "fail_on_unresolved_style": False,
    "allow_legacy_blocks_fallback": False,
}
DEFAULT_QUALITY_EXPECTATIONS = {
    "require_text_coverage": True,
    "require_no_protected_overlap": True,
    "require_no_source_text_leak": True,
}


def choose_renderer(role, object_type) -> str:
    role = str(role or "")
    if role in RENDERER_BY_ROLE:
        return RENDERER_BY_ROLE[role]
    ot = str(object_type or "").lower()
    if ot in {"code_block", "inline_code", "code"}:
        return "code"
    if ot in {"formula_block", "equation", "formula_expression"}:
        return "formula"
    if ot in {"table_cell"}:
        return "table"
    # Unknown role must NOT silently become paragraph (directive §8).
    return "anchored_label_review"


def _union(boxes) -> list | None:
    boxes = [b for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _coverage_bbox(source_unit_ids, unit_index, fallback) -> list | None:
    boxes = []
    for sid in source_unit_ids or []:
        u = unit_index.get(sid)
        if u:
            b = (u.get("geometry") or {}).get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append(b)
    if fallback:
        boxes.append(fallback)
    return _union(boxes) or fallback


def _descendants(uid: str, children_map: dict) -> set:
    out, stack = set(), list(children_map.get(uid) or [])
    while stack:
        c = stack.pop()
        if c in out:
            continue
        out.add(c)
        stack.extend(children_map.get(c) or [])
    return out


def compile_page_render_plan(translated_input_data: dict, *, reconstruction_mode: str = "debug") -> PageRenderPlan:
    normalized = PageReconstructInputAdapter().normalize(translated_input_data)

    units = normalized["units"]
    unit_index = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    children_map = {u.get("unit_id"): (u.get("children_ids") or []) for u in units if isinstance(u, dict)}

    translated_units = normalized["translated_units"]
    preservation_plan = normalized["preservation_plan"]
    exclusion_plan = normalized["exclusion_plan"]
    reconstruction_plan = normalized["reconstruction_plan"]
    style_system = normalized.get("style_system") or {}

    # Index the reconstruction_plan so each translated unit can pull its style /
    # render_contract / role (directive §9, lot 1).
    plan_by_tuid, plan_by_uid, plan_by_sid = {}, {}, {}
    for entry in reconstruction_plan:
        if entry.get("translation_unit_id"):
            plan_by_tuid[entry["translation_unit_id"]] = entry
        if entry.get("unit_id"):
            plan_by_uid[entry["unit_id"]] = entry
        for sid in entry.get("source_unit_ids") or []:
            plan_by_sid.setdefault(sid, entry)

    def _plan_for(item):
        return (plan_by_tuid.get(item.get("translation_unit_id"))
                or plan_by_uid.get(item.get("unit_id"))
                or next((plan_by_sid.get(s) for s in (item.get("source_unit_ids") or []) if s in plan_by_sid), None))

    findings: list[dict] = []

    # consumed: source ids covered by a chosen reconstruction unit (+ descendants).
    consumed: set = set()
    for item in translated_units:
        for sid in item.get("source_unit_ids") or []:
            consumed.add(sid)
            if item.get("consume_source_units") or item.get("preferred_over_children"):
                consumed |= _descendants(sid, children_map)

    excluded: set = set()
    for e in exclusion_plan:
        for sid in e.get("source_unit_ids") or []:
            excluded.add(sid)

    protected_index = build_protected_region_index(
        units=unit_index, preservation_plan=preservation_plan, exclusion_plan=exclusion_plan,
        regions=normalized["regions"], visual_layers=normalized["visual_layers"],
    )

    # translated text layer (no parent/child double render).
    translated_text: list[TranslatedTextUnit] = []
    rendered_ids: set = set()
    for i, item in enumerate(translated_units, start=1):
        sids = list(item.get("source_unit_ids") or [])
        if sids and all(s in rendered_ids for s in sids):
            findings.append({"type": "duplicate_render_skipped", "source_unit_ids": sids,
                             "translation_unit_id": item.get("translation_unit_id")})
            continue
        rt = item.get("render_target") or {}
        bbox = item.get("layout_bbox") or rt.get("layout_bbox") or rt.get("bbox") or item.get("bbox")
        anchor_line = item.get("anchor_bbox") or rt.get("anchor_bbox")
        budget = item.get("layout_budget") or {}
        plan_item = _plan_for(item)
        # Role: most specific available (reconstruction_plan > translation unit).
        role = (plan_item or {}).get("role") or item.get("role")
        coverage = item.get("coverage_bbox") or rt.get("coverage_bbox") or _coverage_bbox(sids, unit_index, bbox)
        # Font size uses the single-line anchor height (not the full block).
        style = resolve_style(item, plan_item, unit_index, style_system, role=role, line_bbox=anchor_line or bbox)
        if style.get("confidence", 0) <= 0.2:
            findings.append({"type": "unresolved_style", "translation_unit_id": item.get("translation_unit_id"),
                             "severity": "review"})
        # ABSOLUTE PRIORITY fix: flow text must lay out in the full block, not one line.
        layout_bbox, patch_bbox, _anchor, lb_findings = resolve_layout(role, bbox, coverage)
        patch_bbox = item.get("patch_bbox") or rt.get("patch_bbox") or patch_bbox
        for f in lb_findings:
            findings.append({**f, "translation_unit_id": item.get("translation_unit_id")})
        translated_text.append(TranslatedTextUnit(
            id=f"ru_{i:04d}", kind="translated_text",
            renderer=choose_renderer(role, item.get("object_type")),
            source_unit_ids=sids, source_text=item.get("text"), translated_text=item.get("translated_text"),
            role=role, object_type=item.get("object_type"), semantic_kind=item.get("semantic_kind"),
            page_role=item.get("page_role"), bbox=bbox,
            coverage_bbox=coverage, layout_bbox=layout_bbox, patch_bbox=patch_bbox,
            bbox_reliable=bool(budget.get("bbox_reliable", bbox is not None)),
            style=style, render_target=rt,
            render_contract=(plan_item or {}).get("render_contract") or item.get("render_contract") or {},
        ))
        for s in sids:
            rendered_ids.add(s)
            rendered_ids |= _descendants(s, children_map)

    # preserved layers from preservation_plan (over_text vs original).
    underlays, overlays = [], []
    for i, p in enumerate(preservation_plan, start=1):
        mode = p.get("preservation_mode")
        z = "over_text" if mode == "preserve_text_exactly" else "preserve_original"
        pu = PreservedUnit(id=f"pres_{i:04d}", source="preservation_plan", reason=str(p.get("reason") or "preserve"),
                           bbox=p.get("bbox"), text=p.get("text"), preservation_mode=mode, z_policy=z)
        (overlays if z == "over_text" else underlays).append(pu)

    if normalized["page"].get("page_role") and not translated_text:
        findings.append({"type": "no_translated_text", "severity": "review"})

    geom = (normalized["page"].get("geometry") or {})
    page = {
        "page_index": (normalized["page"].get("page_index")),
        "width_pt": geom.get("width"), "height_pt": geom.get("height"),
        "rotation": normalized["page"].get("rotation") or 0,
        "coordinate_unit": geom.get("unit") or "pt",
        "coordinate_origin": geom.get("origin") or "top_left",
        "render_width_px": geom.get("render_width_px"), "render_height_px": geom.get("render_height_px"),
        "page_role": normalized["page"].get("page_role"),
    }

    patches, patch_findings = plan_patches(translated_text, protected_index)
    findings.extend(patch_findings)

    background = resolve_background(normalized)
    findings.extend(background.get("findings") or [])

    # Publication gate: incomplete/ko translations cannot be reconstructed as
    # publication-ready (directive PR-Lot 1).
    tr = normalized.get("translation_result") or {}
    ling = tr.get("linguistic_quality_status")
    coverage = (((tr.get("linguistic_quality_validation") or {}).get("translation_coverage_ratio"))
                or (tr.get("quality") or {}).get("translation_coverage_ratio"))
    publication_blocked = False
    if reconstruction_mode == "publication":
        reasons = []
        if ling and ling != "ok":
            reasons.append(f"linguistic_quality_status={ling}")
        if coverage is not None and coverage < 0.98:
            reasons.append(f"translation_coverage_ratio={coverage}")
        if reasons:
            publication_blocked = True
            findings.append({"type": "publication_blocked", "severity": "ko", "reasons": reasons})
    render_policy = dict(DEFAULT_RENDER_POLICY)
    render_policy.update({"reconstruction_mode": reconstruction_mode,
                          "publication_blocked": publication_blocked,
                          "translation_coverage_ratio": coverage})

    return PageRenderPlan(
        page=page, translated_text=translated_text, background=[background],
        preserved_underlays=underlays, preserved_overlays=overlays, patches=patches,
        protected_regions=protected_index.regions,
        consumed_source_unit_ids=sorted(consumed), excluded_source_unit_ids=sorted(excluded),
        render_policy=render_policy, quality_expectations=dict(DEFAULT_QUALITY_EXPECTATIONS),
        findings=findings,
    )


class PageRenderPlanCompiler:
    def compile(self, normalized_or_data: dict) -> PageRenderPlan:
        # Accept either raw translated_input_data or a dict already carrying views.
        return compile_page_render_plan(normalized_or_data)
