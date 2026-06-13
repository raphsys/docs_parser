"""Document-level context extraction for repeated headers/footers and marks."""

from __future__ import annotations

from collections import Counter, defaultdict


def build_document_context(page_structures: list[dict]) -> dict:
    top_texts: Counter[str] = Counter()
    bottom_texts: Counter[str] = Counter()
    publisher_marks: Counter[str] = Counter()
    watermarks: Counter[str] = Counter()
    page_numbers: Counter[str] = Counter()
    page_roles: dict[str, list[int]] = defaultdict(list)
    for idx, page in enumerate(page_structures or []):
        page_roles[str(page.get("page_role") or "unknown")].append(idx)
        height = float(((page.get("dimensions") or {}).get("height")) or 0.0)
        for block in page.get("blocks") or []:
            text = _block_text(block)
            if not text:
                continue
            bbox = block.get("bbox") or [0, 0, 0, 0]
            y0 = float(bbox[1]) if len(bbox) == 4 else 0.0
            y1 = float(bbox[3]) if len(bbox) == 4 else 0.0
            if height and y1 < height * 0.12:
                top_texts[text] += 1
                if _page_number(text):
                    page_numbers[text] += 1
            if height and y0 > height * 0.88:
                bottom_texts[text] += 1
                if _page_number(text):
                    page_numbers[text] += 1
            if any(word in text.lower() for word in ("ebook", "publisher", "manning", "watermark")):
                publisher_marks[text] += 1
            if any(word in text.lower() for word in ("watermark", "draft", "sample", "preview", "confidential")):
                watermarks[text] += 1
    return {
        "schema_version": "document_context.v1",
        "repeated_headers": _repeated(top_texts),
        "repeated_footers": _repeated(bottom_texts),
        "publisher_marks": _repeated(publisher_marks),
        "watermarks": _repeated(watermarks),
        "page_numbers": _repeated(page_numbers),
        "running_titles": _repeated(top_texts + bottom_texts),
        "page_roles": dict(page_roles),
        "toc_detected": bool(page_roles.get("toc")),
        "index_pages": page_roles.get("index", []),
    }


def _block_text(block: dict) -> str:
    if block.get("text"):
        return str(block.get("text")).strip()
    texts = []
    for line in block.get("lines") or []:
        if line.get("line_text"):
            texts.append(str(line.get("line_text")).strip())
    return " ".join(t for t in texts if t).strip()


def _repeated(counter: Counter[str]) -> list[dict]:
    return [
        {"text": text, "count": count}
        for text, count in counter.most_common()
        if count >= 2
    ]


def _page_number(text: str) -> bool:
    text = str(text or "").strip()
    return bool(text and (text.isdigit() or text.lower() in {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}))
