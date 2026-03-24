import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ocr_server
from structure_extractor import LayoutV2Builder


def _page_signature(summary):
    flags = []
    if summary["columns"] >= 2:
        flags.append("two_col")
    if summary["has_equation"]:
        flags.append("equation")
    if summary["has_caption"]:
        flags.append("caption")
    if summary["has_diagram_labels"]:
        flags.append("diagram_labels")
    if summary["has_header_footer"]:
        flags.append("header_footer")
    if summary["has_section_heading"]:
        flags.append("section_heading")
    if summary["non_text_regions"] >= 3:
        flags.append("non_text_dense")
    return "|".join(
        [
            summary["page_role"],
            summary["page_family"],
            f"cols={summary['columns']}",
            ",".join(flags) if flags else "plain",
        ]
    )


def _summarize_page(page_number, page_data):
    blocks = page_data.get("blocks") or []
    layout = page_data.get("layout") or {}
    columns = layout.get("columns") or []
    role_counter = Counter((b.get("role") or "body") for b in blocks)
    summary = {
        "page": page_number,
        "page_role": str(page_data.get("page_role") or "body"),
        "page_family": str(page_data.get("page_family") or "body_text"),
        "columns": len(columns) or 1,
        "block_count": len(blocks),
        "role_counts": dict(role_counter),
        "images": len(page_data.get("images") or []),
        "drawings": len(page_data.get("drawings") or []),
        "non_text_regions": len(page_data.get("non_text_zones") or []),
        "has_equation": role_counter.get("equation_inline", 0) > 0,
        "has_caption": role_counter.get("figure_caption", 0) > 0,
        "has_diagram_labels": (
            role_counter.get("diagram_label", 0) > 0
            or role_counter.get("diagram_text_label", 0) > 0
        ),
        "has_header_footer": role_counter.get("header", 0) > 0 or role_counter.get("footer", 0) > 0,
        "has_section_heading": role_counter.get("section_heading", 0) > 0,
    }
    summary["signature"] = _page_signature(summary)
    return summary


def _process_page(doc, page_index, builder):
    page = doc[page_index]
    native = ocr_server.native_pdf_extractor.extract_page(page, sx=1.0, sy=1.0)
    blocks = native.get("blocks", [])
    blocks = ocr_server._postprocess_blocks(blocks, int(page.rect.width), int(page.rect.height))
    ocr_server._annotate_translation_contracts(blocks)
    layout_meta = ocr_server._annotate_layout(blocks, int(page.rect.width), int(page.rect.height))
    page_data = {
        "blocks": blocks,
        "images": native.get("images", []),
        "drawings": native.get("drawings", []),
        "non_text_zones": native.get("non_text_zones", []),
        "layout": layout_meta,
        "dimensions": {"width": int(page.rect.width), "height": int(page.rect.height)},
    }
    page_data = builder.build(page_data)
    return _summarize_page(page_index + 1, page_data)


def _build_report(pdf_path):
    builder = LayoutV2Builder()
    doc = fitz.open(pdf_path)
    try:
        page_summaries = []
        signature_examples = defaultdict(list)
        family_examples = defaultdict(list)
        role_examples = defaultdict(list)

        page_role_counts = Counter()
        page_family_counts = Counter()
        signature_counts = Counter()
        block_role_counts = Counter()

        for idx in range(doc.page_count):
            summary = _process_page(doc, idx, builder)
            page_summaries.append(summary)
            page_role_counts.update([summary["page_role"]])
            page_family_counts.update([summary["page_family"]])
            signature_counts.update([summary["signature"]])
            block_role_counts.update(summary["role_counts"])

            if len(signature_examples[summary["signature"]]) < 6:
                signature_examples[summary["signature"]].append(summary["page"])
            if len(family_examples[summary["page_family"]]) < 12:
                family_examples[summary["page_family"]].append(summary["page"])
            for role, count in summary["role_counts"].items():
                if count > 0 and len(role_examples[role]) < 12:
                    role_examples[role].append(summary["page"])

        top_signatures = []
        for signature, count in signature_counts.most_common(25):
            top_signatures.append(
                {
                    "signature": signature,
                    "count": count,
                    "example_pages": signature_examples[signature],
                }
            )

        return {
            "pdf": str(pdf_path),
            "page_count": doc.page_count,
            "page_role_counts": dict(page_role_counts),
            "page_family_counts": dict(page_family_counts),
            "block_role_counts": dict(block_role_counts),
            "family_example_pages": dict(family_examples),
            "role_example_pages": dict(role_examples),
            "top_signatures": top_signatures,
            "page_summaries": page_summaries,
        }
    finally:
        doc.close()


def _write_markdown(report, output_path):
    lines = []
    lines.append(f"# Cartographie de corpus")
    lines.append("")
    lines.append(f"- PDF: `{report['pdf']}`")
    lines.append(f"- Pages: `{report['page_count']}`")
    lines.append("")
    lines.append("## Familles de pages")
    lines.append("")
    for family, count in sorted(report["page_family_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        examples = ", ".join(str(x) for x in report["family_example_pages"].get(family, []))
        lines.append(f"- `{family}`: {count} pages. Exemples: {examples}")
    lines.append("")
    lines.append("## Roles de page")
    lines.append("")
    for role, count in sorted(report["page_role_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{role}`: {count} pages")
    lines.append("")
    lines.append("## Signatures dominantes")
    lines.append("")
    for item in report["top_signatures"]:
        examples = ", ".join(str(x) for x in item["example_pages"])
        lines.append(f"- `{item['signature']}`: {item['count']} pages. Exemples: {examples}")
    lines.append("")
    lines.append("## Roles de blocs")
    lines.append("")
    for role, count in sorted(report["block_role_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        examples = ", ".join(str(x) for x in report["role_example_pages"].get(role, []))
        lines.append(f"- `{role}`: {count}. Exemples: {examples}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    pdf_path = ROOT / "tests/doc_pdf/test_docintelligence.pdf"
    out_json = ROOT / "scripts" / "test_docintelligence_case_map.json"
    out_md = ROOT / "scripts" / "test_docintelligence_case_map.md"
    report = _build_report(pdf_path)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, out_md)
    print(json.dumps(
        {
            "pdf": report["pdf"],
            "page_count": report["page_count"],
            "page_family_counts": report["page_family_counts"],
            "page_role_counts": report["page_role_counts"],
            "top_signatures": report["top_signatures"][:10],
            "json_path": str(out_json),
            "md_path": str(out_md),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
