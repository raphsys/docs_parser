import asyncio
import json
import sys
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ocr_server


def _load_cases():
    path = Path(__file__).resolve().parent / "pdf_regression_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _process_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    try:
        pages = []
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_data = ocr_server.process_page(img, idx, pdf_path.name, pdf_page=page)
            pages.append(page_data)
        return pages
    finally:
        doc.close()


async def _run_case(case):
    pdf_path = ROOT / case["pdf"]
    pages = _process_pdf(pdf_path)
    response = await ocr_server.reconstruct_document(
        {"pages": pages},
        target_lang=case.get("target_lang", "fr"),
        debug_compare=True,
    )
    payload = json.loads(response.body.decode("utf-8"))
    coverage = payload.get("coverage_report") or {}
    rendered = coverage.get("rendered_text_report") or {}
    publication_qa = payload.get("publication_qa") or {}
    visual = payload.get("visual_compare") or {}
    rendered_score = rendered.get("coverage_score")
    if rendered_score is None:
        rendered_score = publication_qa.get("rendered_text_coverage_score")
    visual_score = visual.get("overall_score")
    if visual_score is None:
        visual_score = publication_qa.get("visual_similarity_score")
    return {
        "pdf": case["pdf"],
        "content_coverage_score": coverage.get("coverage_score"),
        "rendered_text_coverage_score": rendered_score,
        "visual_similarity_score": visual_score,
        "publication_ready": publication_qa.get("publication_ready"),
        "raw_payload": payload,
    }


def _assert_thresholds(case, result):
    failures = []
    checks = [
        ("content_coverage_score", case.get("min_content_coverage")),
        ("rendered_text_coverage_score", case.get("min_rendered_coverage")),
        ("visual_similarity_score", case.get("min_visual_similarity")),
    ]
    for key, minimum in checks:
        if minimum is None:
            continue
        value = result.get(key)
        if value is None or float(value) < float(minimum):
            failures.append(f"{key}={value} < {minimum}")
    return failures


async def main():
    failures = []
    for case in _load_cases():
        result = await _run_case(case)
        print(json.dumps({k: v for k, v in result.items() if k != "raw_payload"}, ensure_ascii=False))
        failures.extend(f"{case['pdf']}: {msg}" for msg in _assert_thresholds(case, result))
    if failures:
        print("REGRESSION_FAILURES")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
