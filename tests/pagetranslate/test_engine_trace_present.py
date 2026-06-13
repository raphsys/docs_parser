from tests.pagetranslate.test_batch_translation_engine import BatchEngine, _input_data
from pagetranslate import build_page_translation


def test_engine_trace_present():
    result = build_page_translation(_input_data(), translator=BatchEngine(), target_lang="fr")
    trace = result["translation_units"][0]["engine_trace"]
    assert trace["engine"] == "batch"
    assert trace["protected_source_text"]
    assert trace["raw_engine_output"]
    assert trace["post_glossary_output"]

