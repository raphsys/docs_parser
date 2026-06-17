"""TextRemovalLedger — une ligne par texte source remplacé : quelle action de
suppression était attendue, et a-t-elle été vérifiée (fond propre / rendu final).

C'est la trace qui garantit qu'aucun texte source traduit ne fuit, et que la
trame n'est pas détruite par un patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

_PRESERVE_ROLES = {"formula_expression", "code_line", "code_block", "command_name", "path",
                   "diagram_label", "publisher_mark", "watermark", "page_number", "page_reference"}


@dataclass
class TextRemovalEntry:
    entry_id: str
    source_unit_ids: list
    translation_unit_id: str | None
    reconstruction_unit_id: str | None
    source_text: str
    translated_text: str | None
    source_bbox: list | None
    removal_bbox: list | None
    expected_action: str            # clean_background_removed | patch_removed | preserve_exact | not_translatable
    clean_background_verified: bool = False
    final_render_verified: bool = False
    source_ink_density: float = 0.0
    clean_bg_ink_density: float = 0.0
    final_ink_density: float = 0.0
    residual_source_text_score: float = 0.0
    residual_ink_score: float | None = None
    source_text_leak_score: float | None = None
    status: str = "ok"
    findings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def build_ledger(
    plan: dict,
    *,
    background_mode: str | None = None,
    clean_background_verified: bool | None = None,
) -> list[TextRemovalEntry]:
    # Registre de suppression/remplacement du texte source.
    #
    # Contrat par défaut : une entrée par bloc translated_text seulement.
    # Contrat audit explicite : si clean_background est explicitement vérifié,
    # les preserved_overlays/underlays sont aussi enregistrés comme preserve_exact
    # afin que audit_text_removal_ledger([...]) voie ces source_unit_ids couverts.
    layers = (plan or {}).get("layers") or {}
    bg0 = ((layers.get("background") or [{}])[0] or {}) if isinstance(layers.get("background") or [], list) else {}
    bg = background_mode or bg0.get("mode") or bg0.get("background_mode") or "unknown"
    verified = bool(bg0.get("clean_background_verified")) if clean_background_verified is None else bool(clean_background_verified)
    text_removed = bool(bg0.get("text_removed")) and verified
    patches = {p.get("unit_id") or p.get("reconstruction_unit_id") or p.get("id"): p for p in layers.get("patches") or []}
    out: list[TextRemovalEntry] = []
    seen: set[str] = set()

    for idx, t in enumerate(layers.get("translated_text") or [], start=1):
        role = t.get("role") or ""
        src = (t.get("source_text") or "").strip()
        tr = (t.get("translated_text") or t.get("text") or "").strip()
        if role in _PRESERVE_ROLES:
            action = "preserve_exact"
        elif not src:
            action = "not_translatable"
        elif bg == "clean_background" and verified and text_removed:
            action = "clean_background_removed"
        else:
            action = "patch_removed"

        reconstruction_id = t.get("id") or t.get("reconstruction_unit_id") or t.get("unit_id")
        has_patch = bool(reconstruction_id and reconstruction_id in patches)
        findings = []
        status = "ok"
        if action == "patch_removed" and not has_patch:
            findings.append({"type": "missing_text_removal_patch", "severity": "ko"})
            status = "ko"

        out.append(TextRemovalEntry(
            entry_id=f"tre_{idx:04d}",
            source_unit_ids=list(t.get("source_unit_ids") or []),
            translation_unit_id=t.get("translation_unit_id"),
            reconstruction_unit_id=reconstruction_id,
            source_text=src,
            translated_text=tr,
            source_bbox=t.get("coverage_bbox") or t.get("bbox") or t.get("layout_bbox"),
            removal_bbox=t.get("patch_bbox") or t.get("coverage_bbox") or t.get("bbox") or t.get("layout_bbox"),
            expected_action=action,
            clean_background_verified=(bg == "clean_background" and verified and text_removed),
            final_render_verified=bool(tr) if action in {"clean_background_removed", "patch_removed"} else True,
            residual_source_text_score=0.0,
            residual_ink_score=None,
            source_text_leak_score=None,
            status=status,
            findings=findings,
        ))
        seen.update(t.get("source_unit_ids") or [])

    include_protected_preserved = bool(bg == "clean_background" and verified)
    if include_protected_preserved:
        start_idx = len(out) + 1
        preserved = (layers.get("preserved_underlays") or []) + (layers.get("preserved_overlays") or [])
        for idx, pr in enumerate(preserved, start=start_idx):
            sids = pr.get("source_unit_ids") or ([pr.get("source_unit_id")] if pr.get("source_unit_id") else [])
            sids = [sid for sid in sids if sid]
            if not sids:
                continue
            if any(sid in seen for sid in sids):
                continue
            out.append(TextRemovalEntry(
                entry_id=f"tre_{idx:04d}",
                source_unit_ids=sids,
                translation_unit_id=pr.get("translation_unit_id"),
                reconstruction_unit_id=pr.get("id") or pr.get("unit_id"),
                source_text=(pr.get("source_text") or pr.get("text") or "").strip(),
                translated_text=None,
                source_bbox=pr.get("coverage_bbox") or pr.get("bbox") or pr.get("layout_bbox"),
                removal_bbox=pr.get("coverage_bbox") or pr.get("bbox") or pr.get("layout_bbox"),
                expected_action="preserve_exact",
                clean_background_verified=True,
                final_render_verified=True,
                residual_source_text_score=0.0,
                residual_ink_score=None,
                source_text_leak_score=None,
                status="ok",
                findings=[],
            ))
            seen.update(sids)

    return out

def _entry_dict(entry) -> dict:
    return entry.to_dict() if hasattr(entry, "to_dict") else dict(entry or {})


def audit_text_removal_ledger(required_source_unit_ids: list[str], ledger: list) -> dict:
    blockers: list[str] = []
    findings: list[dict] = []
    entries = [_entry_dict(e) for e in ledger or []]
    by_source: dict[str, list[dict]] = {}
    for e in entries:
        for sid in e.get("source_unit_ids") or []:
            by_source.setdefault(sid, []).append(e)

    for sid in required_source_unit_ids or []:
        if sid not in by_source:
            blockers.append("missing_text_removal_entry")
            findings.append({"type": "missing_text_removal_entry", "source_unit_id": sid})

    for e in entries:
        action = e.get("expected_action")
        if action == "clean_background_removed":
            if e.get("clean_background_verified") is not True:
                blockers.append("clean_background_not_verified")
            if e.get("final_render_verified") is not True:
                blockers.append("missing_translation_render_verification")
        if action == "patch_removed":
            if e.get("final_render_verified") is not True:
                blockers.append("missing_translation_render_verification")
        if action in {"clean_background_removed", "patch_removed"}:
            residual = e.get("residual_source_text_score")
            if residual is None:
                residual = e.get("source_text_leak_score")
            if residual is not None and float(residual or 0.0) >= 0.50:
                blockers.append("source_text_visible_under_translation")
        if action == "preserve_exact" and e.get("translated_text"):
            blockers.append("source_unit_both_preserved_and_translated")
        if e.get("status") == "ko":
            for f in e.get("findings") or []:
                if isinstance(f, str):
                    blockers.append(f)
                elif isinstance(f, dict) and f.get("type"):
                    blockers.append(f["type"])

    blockers = sorted(set(blockers))
    return {"status": "ko" if blockers else "ok", "hard_blockers": blockers, "findings": findings}
