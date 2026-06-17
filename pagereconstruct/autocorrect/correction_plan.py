"""Traduction des findings d'audit en actions de correction (CorrectionPlan)."""

from __future__ import annotations

from dataclasses import dataclass, field

# finding/blocker -> action de correction
_RULES = {
    "overflow": "shrink_block",
    "block_text_missing": "reflow_block",
    "block_text_overlap_critical": "reflow_block",
    "block_protected_overlap": "move_block",
    "text_protected_overlap_critical": "move_block",
    "special_zone_overlap": "move_block",
    "patch_protected_overlap": "regenerate_background_zone",
    "font_size_drift": "adjust_line_height",
    "typo_font_class_mismatch": "force_code_preserve",
    "translation_truncated": "mark_review",
    "missing_translatable_text": "mark_review",
    "source_text_leak": "regenerate_background_zone",
}


@dataclass
class CorrectionAction:
    action: str
    target: str | None = None
    params: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


def plan_corrections(page_report) -> list[CorrectionAction]:
    """Dérive des actions depuis un PagePublicationReadyReport (sans doublon)."""
    actions, seen = [], set()
    items = list(page_report.hard_blockers) + [getattr(f, "type", f.get("type") if isinstance(f, dict) else "")
                                               for f in page_report.findings]
    for it in items:
        act = _RULES.get(it)
        if act and (act, it) not in seen:
            seen.add((act, it))
            actions.append(CorrectionAction(action=act, reason=it))
    return actions
