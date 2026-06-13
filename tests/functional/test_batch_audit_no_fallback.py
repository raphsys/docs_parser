import json

from pageprint import build_pageprint_input_data
from tools.run_batch_functional_audit import main as batch_main
from tools.run_functional_audit import audit_pages


def _page():
    return build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{"id": "b1", "bbox": [50, 50, 550, 100], "role": "body", "lines": [{"id": "l1", "bbox": [50, 50, 550, 80], "line_text": "This sentence should be translated.", "phrases": [{"id": "p1", "bbox": [50, 50, 550, 80], "texte": "This sentence should be translated."}]}]}],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )


def test_batch_audit_no_fallback():
    result = audit_pages([_page()], run_pagetranslate=True, dry_run=True)
    assert result["functional_status"] == "ok"
    assert result["metrics"]["pages_using_fallback"] == 0


def test_batch_audit_cli_runs(tmp_path, monkeypatch, capsys):
    (tmp_path / "page.json").write_text(json.dumps(_page()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_batch_functional_audit.py", str(tmp_path), "--run-pagetranslate", "--dry-run"])
    assert batch_main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["metrics"]["pages_total"] == 1
