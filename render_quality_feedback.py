def _block_defect(block, defect_type, severity="warning", fix_hint="", recommended_strategy="", description=""):
    return {
        "block_id": block.get("block_id"),
        "object_type": block.get("object_type"),
        "severity": severity,
        "type": defect_type,
        "fix_hint": fix_hint,
        "recommended_strategy": recommended_strategy,
        "description": description,
    }


def build_second_pass_report(summary):
    defects = []
    for block in (summary or {}).get("blocks") or []:
        verdict = dict(block.get("render_verdict") or {})
        if verdict.get("status") == "failed":
            for finding in verdict.get("findings") or []:
                defects.append(
                    _block_defect(
                        block,
                        finding.get("type") or "render_failed",
                        severity=finding.get("severity") or "failed",
                        fix_hint=finding.get("fix_hint") or "",
                        recommended_strategy=finding.get("recommended_strategy") or verdict.get("recommended_strategy") or "",
                        description="Defaut bloquant detecte par le verdict interne de rendu.",
                    )
                )
        if not block.get("region_presence_ok", True):
            defects.append(
                _block_defect(
                    block,
                    "text_missing",
                    severity="failed",
                    fix_hint="rerender_missing_text_in_origin_block",
                    recommended_strategy="expand_block_or_split_paragraph",
                    description="Tout le texte attendu n'est pas present dans le bloc d'origine.",
                )
            )
        if block.get("source_overlay_findings"):
            for finding in block.get("source_overlay_findings") or []:
                defects.append(
                    _block_defect(
                        block,
                        "source_overlay",
                        severity="failed",
                        fix_hint=finding.get("fix_hint") or "repatch_background_then_rerender",
                        recommended_strategy=finding.get("recommended_strategy") or "source_overlay_cleanup_then_rerender",
                        description=finding.get("description") or "Le texte source reste visible sous la traduction.",
                    )
                )
        if block.get("background_findings"):
            for finding in block.get("background_findings") or []:
                defects.append(
                    _block_defect(
                        block,
                        finding.get("type") or "background_mismatch",
                        severity=finding.get("severity") or "warning",
                        fix_hint="sample_local_background_then_rerender",
                        recommended_strategy="local_background_patch",
                        description=finding.get("description") or "Fond local non conforme.",
                    )
                )
        if block.get("glyph_findings"):
            for finding in block.get("glyph_findings") or []:
                defects.append(
                    _block_defect(
                        block,
                        "glyph_loss",
                        severity=finding.get("severity") or "failed",
                        fix_hint=finding.get("fix_hint") or "rerender_with_target_language_font",
                        recommended_strategy=finding.get("recommended_strategy") or "unicode_font_fallback_same_style",
                        description=finding.get("description") or "Glyphes de la langue cible absents ou corrompus.",
                    )
                )
        cell_validation = block.get("cell_validation")
        if isinstance(cell_validation, dict):
            for finding in cell_validation.get("findings") or []:
                defects.append(
                    _block_defect(
                        block,
                        finding.get("type") or "table_cell_validation_failed",
                        severity=finding.get("severity") or "failed",
                        fix_hint=finding.get("fix_hint") or "rerender_cell_locked_or_preserve_source",
                        recommended_strategy=finding.get("recommended_strategy") or "cell_locked_fit_then_source_preserve",
                        description="Validation cellule par cellule echouee.",
                    )
                )
        if not block.get("style_ok", True):
            report = dict(block.get("style_preservation_report") or {})
            if report.get("font_size_ratio") is not None and block.get("style_min_ratio") is not None:
                try:
                    if float(report.get("font_size_ratio")) < float(block.get("style_min_ratio")):
                        defects.append(
                            _block_defect(
                                block,
                                "font_too_small",
                                severity="failed",
                                fix_hint="expand_block_before_shrink",
                                recommended_strategy="expanded_bbox_or_page_rebalance",
                                description="La taille de police rendue est sous le seuil du contrat.",
                            )
                        )
                except Exception:
                    pass
            if report.get("line_height_ratio") is not None:
                try:
                    if float(report.get("line_height_ratio")) < 0.92:
                        defects.append(
                            _block_defect(
                                block,
                                "line_height_too_dense",
                                severity="warning",
                                fix_hint="rerender_with_larger_line_height",
                                recommended_strategy="expanded_bbox_or_reflow",
                                description="La hauteur de ligne est trop compressee.",
                            )
                        )
                except Exception:
                    pass
            flag_matches = dict(report.get("flag_matches") or {})
            lost_flags = [name for name, ok in flag_matches.items() if ok is False]
            if lost_flags:
                defects.append(
                    _block_defect(
                        block,
                        "style_lost",
                        severity="failed",
                        fix_hint="rerender_with_style_preserving_font",
                        recommended_strategy="compatible_font_same_style",
                        description=f"Styles perdus: {', '.join(lost_flags)}.",
                    )
                )
        if not block.get("protected_tokens_ok", True):
            defects.append(
                _block_defect(
                    block,
                    "protected_missing",
                    severity="failed",
                    fix_hint="rerender_protected_token",
                    recommended_strategy="exact_preserve_anchored",
                    description="Un token protege attendu est absent du bloc.",
                )
            )
    blocking = [item for item in defects if item.get("severity") == "failed"]
    return {
        "case_id": (summary or {}).get("case_id"),
        "defect_count": len(defects),
        "blocking_defect_count": len(blocking),
        "converged": len(blocking) == 0,
        "auto_corrections_applied": [],
        "remaining_defects": defects,
    }
