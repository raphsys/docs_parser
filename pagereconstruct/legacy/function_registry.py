"""Registre de migration des fonctions de l'ANCIEN moteur (source unique).

Chaque entrée trace une fonction/classe legacy utile et sa décision de migration.
Décisions: KEEP_AS_IS | ADAPT | WRAP | MERGE | DROP | REPLACE_TESTED.
Statut de portage: done | partial | todo | dropped.

Règle (anti-régression) : on ne crée pas un module moderne sans vérifier ici
qu'il n'existe pas déjà un équivalent ancien à porter.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class MigrationEntry:
    source_file: str
    symbol: str
    role: str
    decision: str
    destination: str
    status: str
    test: str = ""

    def to_dict(self):
        return asdict(self)


REGISTRY: list[MigrationEntry] = [
    # --- mesure / typographie ---
    MigrationEntry("reconstructor.py", "_measure_text / _measure_text_width", "Mesure texte/largeur",
                   "ADAPT", "pagereconstruct/text_measure.py", "done", "test_render_ops"),
    MigrationEntry("reconstructor.py", "_resolve_compatible_font / FontResolver", "Police compatible + fallback glyphes",
                   "WRAP", "pagereconstruct/font_resolver_bridge.py", "partial", ""),
    MigrationEntry("reconstructor.py", "BlockSemanticProfile (dominant font)", "Profil typo dominant",
                   "ADAPT", "pagereconstruct/ocr_typography_engine.py", "done", "test_pubready_core"),
    # --- contrat / blocs ---
    MigrationEntry("reconstructor.py", "BlockReconstructionPlan", "Contrat de bloc",
                   "ADAPT", "pagereconstruct/block_contract.py", "done", "test_final_contract"),
    MigrationEntry("reconstructor.py", "PlacableUnit / LineTemplate", "Unité à placer / gabarit ligne",
                   "ADAPT", "pagereconstruct/layout_contract.py", "partial", ""),
    MigrationEntry("reconstructor.py", "_build_block_reconstruction_plan", "Construction plan bloc",
                   "ADAPT", "pagereconstruct/composition/block_planner.py", "todo", "test_block_planner"),
    # --- placement / candidats ---
    MigrationEntry("reconstructor.py", "RenderCandidate / CandidateScore", "Candidats + score",
                   "ADAPT", "pagereconstruct/candidate_engine.py", "done", "test_render_ops"),
    MigrationEntry("reconstructor.py", "PlacementResult / PlacementCursor", "Résultat/curseur placement",
                   "ADAPT", "pagereconstruct/placement_solver.py", "done", "test_render_ops"),
    MigrationEntry("reconstructor.py", "_score_render_candidate / _render_plan_with_validation", "Scoring + validation candidats",
                   "ADAPT", "pagereconstruct/candidate_engine.py + autocorrect", "partial", ""),
    # --- composition intra-bloc (à porter) ---
    MigrationEntry("reconstructor.py", "_compose_paragraphs_in_box / _render_prose_reflow", "Composition paragraphe / reflow",
                   "ADAPT", "pagereconstruct/composition/intrablock_composer.py", "todo", "test_intrablock_composer"),
    MigrationEntry("reconstructor.py", "_render_with_scale / _composed_paragraphs_have_overflow", "Shrink-to-fit / overflow",
                   "ADAPT", "pagereconstruct/composition/text_fitter.py", "todo", "test_legacy_text_fit_equivalence"),
    MigrationEntry("reconstructor.py", "_render_label_stack / _render_relative_slots", "Stacks labels / slots relatifs",
                   "ADAPT", "pagereconstruct/composition/line_layout_engine.py", "todo", ""),
    # --- fond / inpaint ---
    MigrationEntry("ocr_server.py", "_collect_text_regions_for_inpainting", "Zones texte à inpaint (protège non-texte 25%)",
                   "WRAP", "pipelines/background_cleaner.py", "done", "test_clean_background"),
    MigrationEntry("text_removal_strategy.py", "TextRemovalStrategy.remove", "Inpaint Telea bbox",
                   "WRAP", "pipelines/background_cleaner.py", "done", "test_clean_background"),
    MigrationEntry("ocr_server.py", "_erase_uncovered_pdf_words", "Effacer mots PDF non couverts (en-têtes/pieds)",
                   "ADAPT", "pipelines/background_cleaner.py", "todo", "test_erase_uncovered_words"),
    MigrationEntry("background_inpainter.py", "BackgroundInpainter (LaMa/opencv)", "Inpaint local par crop",
                   "WRAP", "pipelines/background_cleaner.py (option)", "available", ""),
    MigrationEntry("reconstructor.py", "_clean_page_background_path / _insert_page_background", "Fond propre",
                   "ADAPT", "pagereconstruct/background_resolver.py + backends", "done", "test_render_ops"),
    # --- overlays / préservation ---
    MigrationEntry("ocr_server.py / reconstructor.py", "immutable_overlays / _insert_immutable_overlays", "Overlays immuables",
                   "ADAPT", "pagereconstruct/overlay_manager.py + preservation_contract.py", "done", "test_legacy_contracts"),
    MigrationEntry("final_page_compiler.py", "FormulaItem (source_rect + clips)", "Formule = copie région source",
                   "ADAPT", "pagereconstruct/preservation_contract.py (copy_source_region)", "partial", ""),
    MigrationEntry("final_page_compiler.py", "DrawOp (text/source_rect/erase_rects)", "Op de dessin unifiée",
                   "ADAPT", "pagereconstruct/render_ops.py", "done", "test_render_ops"),
    MigrationEntry("document_object_contract.py", "build_document_object_contract", "Contrat objet + inline protection",
                   "ADAPT", "pagereconstruct/object_contract.py", "partial", "test_legacy_contracts"),
    # --- backend PDF ---
    MigrationEntry("reconstructor.py", "_emit_text_run / _emit_rotated_textbox_run", "Insertion texte PyMuPDF (rotation/fontfile)",
                   "ADAPT", "pagereconstruct/backends/pdf_vector.py (execute_ops)", "partial", "test_render_ops"),
    # --- garde-fous ---
    MigrationEntry("reconstructor.py", "_render_page_text_rescue / _enforce_page_block_text_coverage", "Jamais page image-only si texte attendu",
                   "ADAPT", "pubready/stages/* (gate) + validator", "todo", "test_text_presence_gate"),
    # --- abandon ---
    MigrationEntry("reconstructor.py", "_get_render_agent / _ai_refine_render_strategy", "Agents IA dans le rendu",
                   "DROP", "—", "dropped", ""),
]


def by_status(status: str) -> list[MigrationEntry]:
    return [e for e in REGISTRY if e.status == status]


def pending() -> list[MigrationEntry]:
    return [e for e in REGISTRY if e.status in {"todo", "partial"}]


def to_markdown() -> str:
    L = ["# LEGACY_FUNCTION_REGISTRY",
         "",
         "Migration fonction par fonction de l'ancien moteur → moderne. Source unique: "
         "`pagereconstruct/legacy/function_registry.py`. Décisions: KEEP_AS_IS / ADAPT / WRAP / "
         "MERGE / DROP / REPLACE_TESTED.",
         "",
         "| Ancien fichier | Symbole | Rôle | Décision | Destination | Statut | Test |",
         "|---|---|---|---|---|---|---|"]
    for e in REGISTRY:
        L.append(f"| {e.source_file} | {e.symbol} | {e.role} | {e.decision} | {e.destination} | {e.status} | {e.test} |")
    done = sum(e.status == "done" for e in REGISTRY)
    L += ["", f"**Bilan**: {done}/{len(REGISTRY)} done · "
          f"{sum(e.status=='partial' for e in REGISTRY)} partial · "
          f"{sum(e.status=='todo' for e in REGISTRY)} todo · "
          f"{sum(e.status=='dropped' for e in REGISTRY)} dropped.",
          "",
          "## Reste à porter (todo/partial) — ordre de priorité",
          "1. background/inpaint: `_erase_uncovered_pdf_words` (en-têtes/pieds).",
          "2. composition intra-bloc: `_compose_paragraphs_in_box`, `_render_with_scale` (overflow/shrink).",
          "3. backend PDF: maturité `_emit_rotated_textbox_run` (rotation/clipping).",
          "4. préservation: `FormulaItem` copy_source_region exact.",
          "5. garde-fou: `_render_page_text_rescue` → gate présence texte.",
          ""]
    return "\n".join(L)
