"""SourceTextLifecycleLedger.

One row per PagePrint text unit: translation decision, reconstruction input,
render operation, and visual-audit status. This is a contract audit; real image
visibility is still handled by VisualImageAudit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


TEXT_LEVELS = {"block", "line", "phrase", "span", "word"}
VALID_EXCLUSION_REASONS = {
    "artifact",
    "publisher_mark",
    "watermark",
    "page_number",
    "formula",
    "code",
    "protected_visual_region",
    "background_only",
    "exclude_as_artifact",
}


@dataclass
class SourceTextLifecycleEntry:
    source_unit_id: str
    level: str
    source_text: str
    source_bbox: list | None = None
    pagetranslate_state: str = "missing"
    translation_unit_ids: list[str] = field(default_factory=list)
    reconstruction_unit_ids: list[str] = field(default_factory=list)
    textop_ids: list[str] = field(default_factory=list)
    preservationop_ids: list[str] = field(default_factory=list)
    visual_verified: bool = False
    status: str = "ko"
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_source_text_lifecycle_ledger(plan: dict, normalized: dict) -> list[SourceTextLifecycleEntry]:
    units = [
        u for u in normalized.get("units") or []
        if isinstance(u, dict)
        and u.get("unit_id")
        and u.get("level") in TEXT_LEVELS
        and _text(u)
    ]
    unit_ids = {u["unit_id"] for u in units}
    parent_by_id, children_by_parent = _hierarchy(normalized.get("units") or [])

    translated_units = normalized.get("translated_units") or []
    preservation_plan = normalized.get("preservation_plan") or []
    exclusion_plan = normalized.get("exclusion_plan") or []
    render_ops = plan.get("render_ops") or []

    translation_ids_by_source = _ids_by_source(translated_units, "translation_unit_id", unit_ids, parent_by_id, children_by_parent)
    reconstruction_ids_by_source = _ids_by_source(translated_units, "reconstruction_unit_id", unit_ids, parent_by_id, children_by_parent)
    textop_ids_by_source = _ids_by_source(render_ops, "run_id", unit_ids, parent_by_id, children_by_parent, op_type="text")
    # Preservation is local/immutable.  Do not propagate a preserved page number
    # or caption fragment to its ancestor block, otherwise the ancestor appears
    # both translated and preserved even when the child preservation is valid.
    preservationop_ids_by_source = _ids_by_source(
        render_ops, "op_id", unit_ids, parent_by_id, children_by_parent,
        op_type="preservation", include_ancestors=False,
    )
    preservation_plan_by_source = _ids_by_source(
        preservation_plan, "id", unit_ids, parent_by_id, children_by_parent,
        include_ancestors=False,
    )

    excluded: dict[str, str] = {}
    for item in exclusion_plan:
        reason = str(item.get("reason") or "")
        for sid in _covered_source_ids(item.get("source_unit_ids") or [], unit_ids, parent_by_id, children_by_parent):
            excluded[sid] = reason

    visual = plan.get("visual_image_audit") or {}
    visual_verified = bool(visual.get("image_qa_executed")) and not bool(visual.get("blockers"))

    ledger: list[SourceTextLifecycleEntry] = []
    for unit in units:
        uid = unit["unit_id"]
        entry = SourceTextLifecycleEntry(
            source_unit_id=uid,
            level=str(unit.get("level") or ""),
            source_text=_text(unit),
            source_bbox=(unit.get("geometry") or {}).get("bbox") or unit.get("bbox"),
            translation_unit_ids=sorted(translation_ids_by_source.get(uid) or []),
            reconstruction_unit_ids=sorted(reconstruction_ids_by_source.get(uid) or []),
            textop_ids=sorted(textop_ids_by_source.get(uid) or []),
            preservationop_ids=sorted(preservationop_ids_by_source.get(uid) or []),
            visual_verified=visual_verified,
        )
        preserved_by_plan = bool(preservation_plan_by_source.get(uid))
        excluded_reason = excluded.get(uid)

        if entry.reconstruction_unit_ids:
            entry.pagetranslate_state = "translated_or_projected"
            if not entry.textop_ids:
                entry.findings.append("source_text_missing_render_op")
        elif preserved_by_plan:
            entry.pagetranslate_state = "preserved"
            if not entry.preservationop_ids:
                entry.findings.append("source_text_missing_preservation_op")
        elif excluded_reason in VALID_EXCLUSION_REASONS:
            entry.pagetranslate_state = f"excluded:{excluded_reason}"
        else:
            entry.findings.append("source_text_missing_pagetranslate_decision")

        entry.status = "ok" if not entry.findings else "ko"
        ledger.append(entry)
    return ledger


def audit_source_text_lifecycle(plan: dict, normalized: dict) -> dict:
    ledger = build_source_text_lifecycle_ledger(plan, normalized)
    blockers = sorted({finding for entry in ledger for finding in entry.findings})
    missing = [entry.to_dict() for entry in ledger if entry.status != "ok"]
    return {
        "status": "ko" if blockers else "ok",
        "hard_blockers": blockers,
        "source_text_unit_count": len(ledger),
        "missing_count": len(missing),
        "missing": missing,
        "ledger": [entry.to_dict() for entry in ledger],
    }


def _text(unit: dict) -> str:
    return str((unit.get("content") or {}).get("text") or unit.get("text") or "").strip()


def _hierarchy(units: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    parent_by_id: dict[str, str] = {}
    children_by_parent: dict[str, list[str]] = {}
    for unit in units:
        uid = unit.get("unit_id")
        if not uid:
            continue
        parent = unit.get("parent_id") or unit.get("parent_unit_id")
        if parent:
            parent_by_id[uid] = parent
            children_by_parent.setdefault(parent, []).append(uid)
        for child_id in unit.get("children_ids") or []:
            parent_by_id.setdefault(child_id, uid)
            children_by_parent.setdefault(uid, []).append(child_id)
    return parent_by_id, {k: list(dict.fromkeys(v)) for k, v in children_by_parent.items()}


def _descendants(uid: str, children_by_parent: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    stack = list(children_by_parent.get(uid) or [])
    while stack:
        current = stack.pop(0)
        if current in out:
            continue
        out.add(current)
        stack.extend(children_by_parent.get(current) or [])
    return out


def _ancestors(uid: str, parent_by_id: dict[str, str]) -> set[str]:
    out: set[str] = set()
    parent = parent_by_id.get(uid)
    while parent:
        out.add(parent)
        parent = parent_by_id.get(parent)
    return out


def _covered_source_ids(source_ids: list[str], unit_ids: set[str], parent_by_id: dict[str, str],
                        children_by_parent: dict[str, list[str]], *, include_ancestors: bool = True) -> set[str]:
    covered: set[str] = set()
    for sid in source_ids:
        if sid in unit_ids:
            covered.add(sid)
        covered |= (_descendants(sid, children_by_parent) & unit_ids)
        if include_ancestors:
            covered |= (_ancestors(sid, parent_by_id) & unit_ids)
    return covered


def _ids_by_source(items: list[dict], id_key: str, unit_ids: set[str], parent_by_id: dict[str, str],
                   children_by_parent: dict[str, list[str]], op_type: str | None = None,
                   *, include_ancestors: bool = True) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for index, item in enumerate(items):
        if op_type and item.get("op_type") != op_type:
            continue
        source_ids = item.get("source_unit_ids") or []
        if not source_ids:
            continue
        item_id = str(item.get(id_key) or item.get("unit_id") or f"{op_type or 'item'}_{index}")
        for sid in _covered_source_ids(source_ids, unit_ids, parent_by_id, children_by_parent, include_ancestors=include_ancestors):
            out.setdefault(sid, set()).add(item_id)
    return out
