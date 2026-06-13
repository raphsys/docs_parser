from tests.functional.test_batch_audit_no_fallback import _page
from tools.run_functional_audit import audit_pages


def test_batch_audit_runs_pagetranslate():
    result = audit_pages([_page()], run_pagetranslate=True, dry_run=True)
    assert result["page_reports"][0]["pagetranslate_selection_mode"] == "translation_plan"
