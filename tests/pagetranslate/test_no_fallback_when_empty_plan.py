import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pagetranslate import build_page_translation


def test_empty_translation_plan_does_not_fallback_silently():
    result = build_page_translation(
        {
            "schema_version": "pageprint.input.v1",
            "document": {},
            "page": {},
            "views": {"translation_plan": []},
            "units": [],
        },
        dry_run=True,
    )
    assert result["debug"]["selection_mode"] == "translation_plan_empty"
    assert result["debug"]["fallback_selector_used"] is False
    assert result["functional_validation"]["functional_status"] == "ko"
    assert "translation_plan_empty_no_fallback" in result["functional_validation"]["errors"]
