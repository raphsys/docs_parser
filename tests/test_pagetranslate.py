import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageprint import build_pageprint_input_data
from pagetranslate import PAGETRANSLATE_SCHEMA_VERSION, build_page_translation
from pagetranslate.protection import protect_text, restore_text


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate_text(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return f"FR::{text}"


class DroppingTranslator:
    def translate_text(self, text, **kwargs):
        return "FR::texte traduit sans jetons proteges"


class SameTranslator:
    def translate_text(self, text, **kwargs):
        return text


class SelectiveErrorTranslator:
    def translate_text(self, text, **kwargs):
        if "First" in text:
            raise RuntimeError("boom")
        return f"FR::{text}"


def make_input_data():
    return build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "page_family": "body_text",
            "document_type": "technical_book",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [
                {
                    "id": "b1",
                    "bbox": [50, 50, 550, 170],
                    "role": "body",
                    "source_kind": "native",
                    "translation_contract": {
                        "translatable": True,
                        "translation_strategy": "semantic_reflow",
                        "render_policy": "paragraph_flow",
                        "coverage_required": "strict",
                    },
                    "lines": [
                        {
                            "id": "l1",
                            "bbox": [50, 50, 550, 90],
                            "line_text": "First sentence. Second starts",
                            "phrases": [
                                {
                                    "id": "p1",
                                    "bbox": [50, 50, 250, 90],
                                    "texte": "First sentence.",
                                    "spans": [
                                        {
                                            "id": "s1",
                                            "bbox": [50, 50, 250, 90],
                                            "text": "First sentence.",
                                            "style": {"font": "Times", "size": 12},
                                        }
                                    ],
                                },
                                {
                                    "id": "p2",
                                    "bbox": [260, 50, 550, 90],
                                    "texte": "Second starts",
                                    "spans": [
                                        {
                                            "id": "s2",
                                            "bbox": [260, 50, 550, 90],
                                            "text": "Second starts",
                                            "style": {"font": "Times", "size": 12},
                                        }
                                    ],
                                },
                            ],
                        },
                        {
                            "id": "l2",
                            "bbox": [50, 100, 550, 140],
                            "line_text": "and continues here.",
                            "phrases": [
                                {
                                    "id": "p3",
                                    "bbox": [50, 100, 550, 140],
                                    "texte": "and continues here.",
                                    "spans": [
                                        {
                                            "id": "s3",
                                            "bbox": [50, 100, 550, 140],
                                            "text": "and continues here.",
                                            "style": {"font": "Times", "size": 12},
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        source_context={
            "document_id": "doc-translation-test",
            "source_path": "/tmp/doc.pdf",
            "file_name": "doc.pdf",
            "file_type": "pdf",
            "page_count": 1,
            "language": {"source_lang": "en", "target_lang": "fr"},
        },
        extraction_result={"pipeline": "test", "native_pdf_available": True},
    )


def test_pagetranslate_selects_pageprint_phrases_and_excludes_fine_tokens():
    input_data = make_input_data()
    fake = FakeTranslator()

    result = build_page_translation(input_data, translator=fake, target_lang="fr")

    assert result["schema_version"] == PAGETRANSLATE_SCHEMA_VERSION
    assert [unit["level"] for unit in result["translation_units"]] == ["phrase", "semantic_phrase"]
    assert all("_word_" not in unit["unit_id"] for unit in result["translation_units"])
    assert len(fake.calls) == 2


def test_pagetranslate_adds_sentence_characteristics():
    result = build_page_translation(make_input_data(), translator=FakeTranslator(), target_lang="fr")
    units = result["translation_units"]

    assert units[0]["sentence"]["is_sentence_start"] is True
    assert units[0]["sentence"]["is_sentence_end"] is True
    assert units[0]["sentence"]["end_reason"] == "terminal_punctuation"

    assert units[1]["sentence"]["is_sentence_start"] is True
    assert units[1]["level"] == "semantic_phrase"
    assert units[1]["sentence"]["coalesced_from_visual_units"] is True
    assert units[1]["sentence"]["is_sentence_end"] is True


def test_pagetranslate_reinjects_translations_in_translated_input_data():
    input_data = make_input_data()
    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")
    translated_input = result["translated_input_data"]
    by_id = {unit["unit_id"]: unit for unit in translated_input["units"]}

    for item in result["translation_units"]:
        if item["unit_id"] in by_id:
            translated = by_id[item["unit_id"]]["content"]["translated_text"]
            assert translated == f"FR::{item['source_text']}"

    block = next(unit for unit in translated_input["units"] if unit["level"] == "block")
    assert "FR::First sentence." in block["content"]["translated_text"]
    assert translated_input["translation_result"]["translated_count"] == 2


def test_pagetranslate_dry_run_does_not_require_translator():
    result = build_page_translation(make_input_data(), dry_run=True, target_lang="fr")

    assert result["quality"]["dry_run_count"] == 2
    assert all(unit["translated_text"] == unit["source_text"] for unit in result["translation_units"])
    assert all(unit["quality"]["unchanged_problem"] is False for unit in result["translation_units"])


def test_pagetranslate_prefers_semantic_phrase_over_visual_fragments():
    input_data = make_input_data()
    phrase_ids = [
        unit["unit_id"]
        for unit in input_data["units"]
        if unit["level"] == "phrase"
    ]
    input_data.setdefault("semantic_system", {})["semantic_phrases"] = [
        {
            "unit_id": "sem_phrase_001",
            "text": "Second starts and continues here.",
            "source_unit_ids": phrase_ids[1:],
            "bbox": [260, 50, 550, 140],
            "semantic_kind": "prose",
            "structural_context": {"block_unit_id": "p001_block_001"},
        }
    ]

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert [unit["level"] for unit in result["translation_units"]] == ["semantic_phrase"]
    assert result["translation_units"][0]["source_text"] == "Second starts and continues here."
    assert result["translation_units"][0]["source_unit_ids"] == phrase_ids[1:]


def test_pagetranslate_protects_and_restores_numbers_urls_and_doi():
    input_data = make_input_data()
    first_phrase = next(unit for unit in input_data["units"] if unit["level"] == "phrase")
    first_phrase["content"]["text"] = "Download 12 MB from https://example.com and cite doi:10.1000/xyz."

    fake = FakeTranslator()
    result = build_page_translation(input_data, translator=fake, target_lang="fr")
    item = result["translation_units"][0]

    assert "[[[PT0001]]]" in fake.calls[0]["text"]
    assert "12 MB" in item["translated_text"]
    assert "https://example.com" in item["translated_text"]
    assert "doi:10.1000/xyz" in item["translated_text"]
    assert item["quality"]["protected_token_mismatch"] is False


def test_pagetranslate_quality_flags_missing_protected_tokens():
    input_data = make_input_data()
    first_phrase = next(unit for unit in input_data["units"] if unit["level"] == "phrase")
    first_phrase["content"]["text"] = "Keep 42 kg and test@example.com."

    result = build_page_translation(input_data, translator=DroppingTranslator(), target_lang="fr")
    item = result["translation_units"][0]

    assert item["quality"]["protected_token_mismatch"] is True
    assert item["quality"]["needs_review"] is True
    assert result["quality"]["needs_review_count"] >= 1


def test_no_duplicate_semantic_phrase_and_visual_units():
    input_data = make_input_data()
    phrase_ids = [unit["unit_id"] for unit in input_data["units"] if unit["level"] == "phrase"]
    input_data.setdefault("semantic_system", {})["semantic_phrases"] = [
        {
            "unit_id": "sp_all",
            "text": "First sentence. Second starts and continues here.",
            "source_unit_ids": phrase_ids,
            "bbox": [50, 50, 550, 140],
            "structural_context": {"block_unit_id": "p001_block_001"},
        }
    ]

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert [unit["unit_id"] for unit in result["translation_units"]] == ["sp_all"]
    assert result["quality"]["unit_count"] == 1


def test_no_duplicate_semantic_phrase_and_semantic_group():
    input_data = make_input_data()
    phrase_ids = [unit["unit_id"] for unit in input_data["units"] if unit["level"] == "phrase"]
    semantic_system = input_data.setdefault("semantic_system", {})
    semantic_system["semantic_phrases"] = [
        {
            "unit_id": "sp_all",
            "text": "First sentence. Second starts and continues here.",
            "source_unit_ids": phrase_ids,
            "bbox": [50, 50, 550, 140],
            "structural_context": {"block_unit_id": "p001_block_001"},
        }
    ]
    semantic_system["semantic_groups"] = [
        {
            "unit_id": "sg_all",
            "text": "First sentence. Second starts and continues here.",
            "source_unit_ids": phrase_ids,
            "bbox": [50, 50, 550, 140],
            "structural_context": {"block_unit_id": "p001_block_001"},
        }
    ]

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert [unit["level"] for unit in result["translation_units"]] == ["semantic_phrase"]
    assert [unit["unit_id"] for unit in result["translation_units"]] == ["sp_all"]


def test_skip_non_translatable_semantic_entry_and_background_only():
    input_data = make_input_data()
    input_data.setdefault("semantic_system", {})["semantic_phrases"] = [
        {
            "unit_id": "sp_skip",
            "text": "Do not translate",
            "translatable": False,
            "translation_strategy": "background_only",
            "bbox": [50, 50, 550, 140],
            "structural_context": {"block_unit_id": "p001_block_001"},
        }
    ]

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert "sp_skip" not in [unit["unit_id"] for unit in result["translation_units"]]
    assert result["translation_units"] == []


def test_hyphenated_words_are_not_protected_as_formula():
    protected, protections = protect_text("A state-of-the-art pre-trained model.")

    assert protected == "A state-of-the-art pre-trained model."
    assert protections == []


def test_formula_power_protection_e_mc2_is_single_token():
    protected, protections = protect_text("E = mc^2")

    assert protected == "[[[PT0001]]]"
    assert [item["text"] for item in protections] == ["E = mc^2"]


def test_profile_and_item_protected_tokens_are_preserved():
    input_data = make_input_data()
    input_data.setdefault("translation_context", {})["protected_tokens"] = ["Alpha-Beta"]
    first_phrase = next(unit for unit in input_data["units"] if unit["level"] == "phrase")
    first_phrase["content"]["text"] = "Keep Alpha-Beta stable."

    fake = FakeTranslator()
    result = build_page_translation(input_data, translator=fake, target_lang="fr")

    assert "[[[PT0001]]]" in fake.calls[0]["text"]
    assert "Alpha-Beta" in result["translation_units"][0]["translated_text"]


def test_unchanged_translation_needs_review():
    result = build_page_translation(make_input_data(), translator=SameTranslator(), source_lang="en", target_lang="fr")

    assert result["translation_units"][0]["status"] == "unchanged_suspect"
    assert result["translation_units"][0]["preserve_reason"] == "identical_output_suspected"
    assert result["translation_units"][0]["quality"]["unchanged_problem"] is True
    assert result["translation_units"][0]["quality"]["needs_review"] is True


def test_semantic_phrase_projection_consumes_source_units():
    input_data = make_input_data()
    phrase_ids = [unit["unit_id"] for unit in input_data["units"] if unit["level"] == "phrase"]
    input_data.setdefault("semantic_system", {})["semantic_phrases"] = [
        {
            "unit_id": "sp_all",
            "text": "First sentence. Second starts and continues here.",
            "source_unit_ids": phrase_ids,
            "bbox": [50, 50, 550, 140],
            "structural_context": {"block_unit_id": "p001_block_001"},
        }
    ]

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")
    translated_units = {unit["unit_id"]: unit for unit in result["translated_input_data"]["units"]}
    reconstruction_units = result["translated_input_data"]["views"]["reconstruction_units"]

    assert all(translated_units[unit_id]["translation"]["skip_individual_render"] is True for unit_id in phrase_ids)
    semantic_reconstruction = next(unit for unit in reconstruction_units if unit["unit_id"] == "sp_all")
    assert semantic_reconstruction["render_level"] == "semantic_phrase"
    assert semantic_reconstruction["source_unit_ids"] == phrase_ids
    assert semantic_reconstruction["preferred_over_children"] is True


def test_pageprint_direct_phrase_reconstruction_no_parent_duplicates():
    input_data = make_input_data()
    input_data["semantic_system"]["semantic_phrases"] = []
    # Make every phrase terminal so the coalescer does not synthesize a semantic unit.
    for unit in input_data["units"]:
        if unit["level"] == "phrase" and not unit["content"]["text"].endswith("."):
            unit["content"]["text"] += "."

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")
    reconstruction_units = result["translated_input_data"]["views"]["reconstruction_units"]
    translated_unit_ids = {unit["unit_id"] for unit in result["translation_units"]}

    assert {unit["unit_id"] for unit in reconstruction_units} == translated_unit_ids
    assert all(unit["render_level"] == "phrase" for unit in reconstruction_units)


def test_single_span_backfill_does_not_duplicate_render():
    result = build_page_translation(make_input_data(), translator=FakeTranslator(), target_lang="fr")
    translated_input = result["translated_input_data"]
    reconstruction_ids = {unit["unit_id"] for unit in translated_input["views"]["reconstruction_units"]}
    spans = [unit for unit in translated_input["units"] if unit["level"] == "span"]

    assert all(span["unit_id"] not in reconstruction_ids for span in spans)


def test_pagetranslate_coalesces_open_sentence_units():
    result = build_page_translation(make_input_data(), translator=FakeTranslator(), target_lang="fr")
    coalesced = [unit for unit in result["translation_units"] if unit.get("coalesced")]

    assert len(coalesced) == 1
    assert coalesced[0]["source_text"] == "Second starts and continues here."
    assert coalesced[0]["level"] == "semantic_phrase"


def test_non_translatable_semantic_sources_block_fallback():
    input_data = make_input_data()
    phrase_id = next(unit["unit_id"] for unit in input_data["units"] if unit["level"] == "phrase")
    input_data.setdefault("semantic_system", {})["semantic_phrases"] = [
        {
            "unit_id": "sp_hidden",
            "text": "First sentence.",
            "source_unit_ids": [phrase_id],
            "translatable": False,
            "translation_strategy": "background_only",
            "bbox": [50, 50, 250, 90],
            "structural_context": {"block_unit_id": "p001_block_001"},
        }
    ]

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert phrase_id not in {
        source_id
        for unit in result["translation_units"]
        for source_id in unit.get("source_unit_ids", [])
    }


def test_reconstruction_units_use_visual_style():
    result = build_page_translation(make_input_data(), translator=FakeTranslator(), target_lang="fr")
    direct = next(unit for unit in result["translated_input_data"]["views"]["reconstruction_units"] if unit["level"] == "phrase")

    assert direct["style"]["font_family"] == "Times"


def test_error_units_not_in_reconstruction_units():
    result = build_page_translation(input_data=make_input_data(), translator=SelectiveErrorTranslator(), target_lang="fr")
    error_ids = {unit["unit_id"] for unit in result["translation_units"] if unit["status"] == "error"}
    reconstruction_ids = {unit["unit_id"] for unit in result["translated_input_data"]["views"]["reconstruction_units"]}

    assert error_ids.isdisjoint(reconstruction_ids)


def test_translator_exception_is_local_to_unit():
    result = build_page_translation(input_data=make_input_data(), translator=SelectiveErrorTranslator(), target_lang="fr")

    statuses = [unit["status"] for unit in result["translation_units"]]
    assert statuses[0] == "error"
    assert statuses[1:] == ["translated"]
    assert result["translation_units"][0]["quality"]["needs_review"] is True


def test_translator_bridge_passes_rich_context_when_supported():
    input_data = make_input_data()
    fake = FakeTranslator()

    build_page_translation(input_data, translator=fake, source_lang="en", target_lang="fr")

    call = fake.calls[0]
    assert call["source_lang"] == "en"
    assert call["target_lang"] == "fr"
    assert "context_after" in call
    assert "wysiwyg_constraints" in call


def test_pageprint_detected_formula_is_not_translated_or_reconstructed_as_text():
    input_data = build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "page_family": "body_text",
            "document_type": "scientific_paper",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [
                {
                    "id": "body",
                    "bbox": [50, 50, 550, 110],
                    "role": "body",
                    "lines": [
                        {
                            "id": "body_l1",
                            "bbox": [50, 50, 550, 90],
                            "line_text": "This sentence should translate.",
                            "phrases": [
                                {
                                    "id": "body_p1",
                                    "bbox": [50, 50, 550, 90],
                                    "texte": "This sentence should translate.",
                                    "spans": [{"id": "body_s1", "bbox": [50, 50, 550, 90], "text": "This sentence should translate."}],
                                }
                            ],
                        }
                    ],
                },
                {
                    "id": "formula",
                    "bbox": [200, 200, 360, 250],
                    "role": "equation_block",
                    "lines": [{"id": "formula_l1", "bbox": [200, 200, 360, 250], "line_text": "E = mc^2", "phrases": []}],
                },
            ],
        },
        source_context={
            "document_id": "formula-doc",
            "source_path": "/tmp/doc.pdf",
            "file_name": "doc.pdf",
            "file_type": "pdf",
            "page_count": 1,
            "language": {"source_lang": "en", "target_lang": "fr"},
        },
        extraction_result={"pipeline": "test"},
    )

    assert input_data["views"]["protected_visual_units"]
    fake = FakeTranslator()
    result = build_page_translation(input_data, translator=fake, target_lang="fr")

    assert all("E = mc" not in call["text"] for call in fake.calls)
    assert all("E = mc" not in unit["source_text"] for unit in result["translation_units"])
    assert all("E = mc" not in (unit.get("text") or "") for unit in result["translated_input_data"]["views"]["reconstruction_units"])


def test_line_without_phrases_is_not_lost_when_same_block_has_phrases():
    input_data = build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "page_family": "body_text",
            "document_type": "technical_book",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [
                {
                    "id": "b1",
                    "bbox": [50, 50, 550, 160],
                    "role": "body",
                    "lines": [
                        {
                            "id": "l1",
                            "bbox": [50, 50, 550, 90],
                            "line_text": "First line text.",
                            "phrases": [{"id": "p1", "bbox": [50, 50, 550, 90], "texte": "First line text."}],
                        },
                        {
                            "id": "l2",
                            "bbox": [50, 100, 550, 140],
                            "line_text": "Second line text.",
                            "phrases": [],
                        },
                    ],
                }
            ],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
        extraction_result={"pipeline": "test"},
    )

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert [unit["level"] for unit in result["translation_units"]] == ["phrase", "line"]
    assert [unit["source_text"] for unit in result["translation_units"]] == ["First line text.", "Second line text."]


def test_uppercase_section_heading_is_not_rejected_as_acronym():
    input_data = make_input_data()
    first_phrase = next(unit for unit in input_data["units"] if unit["level"] == "phrase")
    first_phrase["content"]["text"] = "BACKGROUND"
    first_phrase["understanding"]["role"] = "section_heading"

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert any(unit["source_text"] == "BACKGROUND" for unit in result["translation_units"])


def test_visual_list_items_are_not_coalesced_into_one_sentence():
    input_data = build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "page_family": "body_text",
            "document_type": "technical_book",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [
                {
                    "id": "b1",
                    "bbox": [50, 50, 550, 210],
                    "role": "body",
                    "lines": [
                        {"id": "l1", "bbox": [50, 50, 550, 80], "line_text": "Item one", "phrases": [{"id": "p1", "bbox": [50, 50, 550, 80], "texte": "Item one"}]},
                        {"id": "l2", "bbox": [50, 90, 550, 120], "line_text": "Item two", "phrases": [{"id": "p2", "bbox": [50, 90, 550, 120], "texte": "Item two"}]},
                        {"id": "l3", "bbox": [50, 130, 550, 160], "line_text": "Item three", "phrases": [{"id": "p3", "bbox": [50, 130, 550, 160], "texte": "Item three"}]},
                    ],
                }
            ],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
        extraction_result={"pipeline": "test"},
    )

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert [unit["source_text"] for unit in result["translation_units"]] == ["Item one", "Item two", "Item three"]
    assert not any(unit.get("coalesced") for unit in result["translation_units"])


def test_unsafe_semantic_phrase_without_anchor_does_not_double_select():
    input_data = make_input_data()
    input_data.setdefault("semantic_system", {})["semantic_phrases"] = [
        {"unit_id": "unsafe_sp", "text": "Same sentence."}
    ]

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert "unsafe_sp" not in [unit["unit_id"] for unit in result["translation_units"]]


def test_tolerant_placeholder_restore_handles_altered_markers():
    protected, protections = protect_text("Keep 42 kg.")

    assert restore_text("FR [[[ PT0001 ]]].", protections) == "FR 42 kg."
    assert restore_text("FR PT0001.", protections) == "FR 42 kg."


def test_preferred_terminology_is_applied_after_translation():
    input_data = make_input_data()
    input_data.setdefault("translation_context", {})["terminology"] = {
        "preferred_terms": {"medical audit": "controle medical"}
    }
    first_phrase = next(unit for unit in input_data["units"] if unit["level"] == "phrase")
    first_phrase["content"]["text"] = "medical audit"

    result = build_page_translation(input_data, translator=FakeTranslator(), target_lang="fr")

    assert "controle medical" in result["translation_units"][0]["translated_text"]


def test_wysiwyg_constraints_include_pageprint_budget():
    result = build_page_translation(make_input_data(), translator=FakeTranslator(), target_lang="fr")
    constraints = result["translation_units"][0]["context"]["wysiwyg_constraints"]

    assert constraints["transformation_budget"]
    assert constraints["layout_freedom"]
    assert constraints["render_contract"]
