from __future__ import annotations

from pathlib import Path

from perfect_extraction_to_reconstruction import PerfectExtractionReconstructionAdapter


def _perfect_model_with_blocks(blocks):
    return {
        "schema_version": "perfect_extraction.v1",
        "source": {"path": ""},
        "document_profile": {"probable_type": "technical_book"},
        "pages": [
            {
                "page_index": 0,
                "page_number_physical": 1,
                "geometry": {"width": 72.0, "height": 144.0},
                "text": {"blocks": blocks},
                "images": [],
                "tables": [],
                "formulas_and_specials": [],
                "columns": {"count": 1},
            }
        ],
    }


def _span(text, bbox, font="Times-Roman", size=10.0):
    return {
        "id": f"span_{text[:4]}",
        "text": text,
        "bbox": bbox,
        "style": {
            "font_family": font,
            "font_postscript_name": font,
            "size_pt": size,
            "color_rgb": "#111111",
            "flags": {"is_serif": True},
        },
        "chars": [{"text": ch, "bbox": [bbox[0], bbox[1], bbox[0] + 1, bbox[3]]} for ch in text[:2]],
    }


def test_adapter_enriches_legacy_block_without_changing_contract_shape(tmp_path: Path):
    block = {
        "id": "b1",
        "source": "native_pdf",
        "role": "paragraph",
        "object_type": "text_block",
        "text": "This is a paragraph.",
        "bbox": [10.0, 20.0, 70.0, 40.0],
        "style_summary": {
            "dominant_font_family": "Times-Roman",
            "dominant_size_pt": 10.0,
            "dominant_color_rgb": "#111111",
            "has_serif": True,
        },
        "lines": [
            {
                "id": "l1",
                "text": "This is a paragraph.",
                "bbox": [10.0, 20.0, 70.0, 30.0],
                "spans": [_span("This is a paragraph.", [10.0, 20.0, 70.0, 30.0])],
            }
        ],
        "layout": {"alignment": "left", "column_index": 0},
    }

    adapted = PerfectExtractionReconstructionAdapter(tmp_path).adapt(_perfect_model_with_blocks([block]))
    page = adapted["pages"][0]
    out = page["blocks"][0]

    assert page["dimensions"]["source_unit"] == "pt"
    assert page["dimensions"]["reconstructor_unit"] == "px_150dpi"
    assert out["id"] == "b1"
    assert out["bbox"] == [20.833, 41.667, 145.833, 83.333]
    assert out["source_kind"] == "perfect_native_pdf_block"
    assert out["structure_hints"]["structural_role_hint"] == "editorial_body"
    assert out["source_layout_mode"]["render_contract"] == "paragraph_reflow"
    assert out["document_object_contract"]["schema_version"] == "document_object_contract.v1"
    assert out["render_policy"] == "translated_editorial"

    line = out["lines"][0]
    phrase = line["phrases"][0]
    span = phrase["spans"][0]
    assert line["source_kind"] == "perfect_line"
    assert phrase["source_kind"] == "perfect_span_phrase"
    assert span["source_kind"] == "perfect_span"
    assert span["style"]["font"] == "Times-Roman"
    assert span["style_attributes"]["font_family_primary"] == "Times-Roman"
    assert out["semantic_phrase_count"] == 1
    assert out["semantic_phrases"][0]["spans"][0]["texte"] == "This is a paragraph."


def test_adapter_classifies_known_legacy_object_types(tmp_path: Path):
    heading = {
        "id": "h1",
        "source": "native_pdf",
        "role": "heading",
        "text": "Introduction",
        "bbox": [10.0, 10.0, 50.0, 20.0],
        "style_summary": {"dominant_font_family": "Times-Bold", "dominant_size_pt": 14.0, "has_bold": True},
        "lines": [{"text": "Introduction", "bbox": [10.0, 10.0, 50.0, 20.0], "spans": [_span("Introduction", [10.0, 10.0, 50.0, 20.0], "Times-Bold", 14.0)]}],
    }
    code = {
        "id": "c1",
        "source": "native_pdf",
        "role": "code",
        "text": "print(value)",
        "bbox": [10.0, 30.0, 60.0, 42.0],
        "style_summary": {"dominant_font_family": "Courier", "dominant_size_pt": 9.0, "has_monospace": True},
        "lines": [{"text": "print(value)", "bbox": [10.0, 30.0, 60.0, 42.0], "spans": [_span("print(value)", [10.0, 30.0, 60.0, 42.0], "Courier", 9.0)]}],
    }
    toc = {
        "id": "t1",
        "source": "native_pdf",
        "role": "paragraph",
        "text": "Chapter One ........ 12",
        "bbox": [10.0, 50.0, 65.0, 62.0],
        "style_summary": {"dominant_font_family": "Times-Roman", "dominant_size_pt": 10.0},
        "lines": [{"text": "Chapter One ........ 12", "bbox": [10.0, 50.0, 65.0, 62.0], "spans": [_span("Chapter One ........ 12", [10.0, 50.0, 65.0, 62.0])]}],
    }

    page = PerfectExtractionReconstructionAdapter(tmp_path).adapt(_perfect_model_with_blocks([heading, code, toc]))["pages"][0]
    by_id = {block["id"]: block for block in page["blocks"]}

    assert by_id["h1"]["object_type"] == "section_heading"
    assert by_id["h1"]["role"] == "section_heading"
    assert by_id["h1"]["structure_hints"]["band_role_hint"] == "title_band"
    assert by_id["c1"]["object_type"] == "code_block"
    assert by_id["c1"]["object_class"] == "technical"
    assert by_id["c1"]["structure_hints"]["layout_behavior_hint"] == "fixed_lines"
    assert by_id["t1"]["object_type"] == "toc_entry"
    assert by_id["t1"]["role"] == "toc_entry"
    assert by_id["t1"]["document_object_contract"]["reconstruction"]["contract_key"] == "toc_entry"


def test_adapter_emits_complete_characteristics_for_all_levels(tmp_path: Path):
    block = {
        "id": "b_complete",
        "source": "native_pdf",
        "role": "paragraph",
        "text": "OpenAI docs are useful.",
        "bbox": [10.0, 20.0, 90.0, 40.0],
        "style_summary": {
            "dominant_font_family": "Times-Roman",
            "dominant_size_pt": 10.0,
            "dominant_color_rgb": "#111111",
            "has_serif": True,
        },
        "words": [{"id": "w_block", "text": "OpenAI", "bbox": [10.0, 20.0, 25.0, 30.0], "confidence": 1.0}],
        "lines": [
            {
                "id": "l_complete",
                "text": "OpenAI docs are useful.",
                "bbox": [10.0, 20.0, 90.0, 30.0],
                "words": [{"id": "w_line", "text": "docs", "bbox": [27.0, 20.0, 40.0, 30.0], "confidence": 1.0}],
                "spans": [_span("OpenAI docs are useful.", [10.0, 20.0, 90.0, 30.0])],
            }
        ],
    }

    adapted = PerfectExtractionReconstructionAdapter(tmp_path).adapt(_perfect_model_with_blocks([block]))
    page = adapted["pages"][0]
    out = page["blocks"][0]
    line = out["lines"][0]
    phrase = line["phrases"][0]
    span = phrase["spans"][0]
    char = span["chars"][0]
    word = line["words"][0]
    expression = phrase["expressions"][0]
    semantic_phrase = out["semantic_phrases"][0]
    semantic_run = out["semantic_runs"][0]
    semantic_group = out["semantic_groups"][0]

    required = {
        "schema_version",
        "level",
        "id",
        "parent_id",
        "source",
        "source_kind",
        "role",
        "object_type",
        "object_class",
        "object_subtype",
        "inline_object_type",
        "inline_object_subtype",
        "text",
        "raw_text",
        "translated_text",
        "bbox",
        "bbox_source_unit",
        "style",
        "style_attributes",
        "layout_attributes",
        "source_layout_mode",
        "structure_hints",
        "object_comprehension",
        "detection",
        "quality",
        "relationships",
        "translation_policy",
        "render_policy",
        "translatable",
        "document_object_contract",
        "inline_structure",
        "has_special_inline_objects",
        "children_counts",
        "perfect_source",
    }
    for unit in [
        adapted["document_characteristics"],
        page,
        out,
        line,
        phrase,
        span,
        char,
        word,
        expression,
        semantic_phrase,
        semantic_run,
        semantic_group,
    ]:
        missing = required - set(unit)
        assert not missing, (unit.get("level"), unit.get("id"), missing)

    assert adapted["complete_field_policy"] == "present_null_when_unknown_zero_for_counts_empty_string_for_text"
    assert page["children_counts"]["blocks"] == 1
    assert out["children_counts"]["lines"] == 1
    assert out["semantic_run_count"] >= 1
    assert out["semantic_group_count"] >= 1
    assert line["children_counts"]["words"] == 1
    assert span["children_counts"]["chars"] >= 1
    assert word["quality"]["confidence"] == 1.0
