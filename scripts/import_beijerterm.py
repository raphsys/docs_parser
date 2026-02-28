#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict


def clean_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def norm_lang(lang):
    s = clean_text(lang).lower()
    aliases = {
        "eng": "en",
        "fra": "fr",
        "fre": "fr",
        "spa": "es",
        "deu": "de",
        "ger": "de",
        "ita": "it",
        "por": "pt",
        "ara": "ar",
        "zho": "zh",
        "jpn": "ja",
        "kor": "ko",
        "rus": "ru",
        "hin": "hi",
    }
    return aliases.get(s, s)


def parse_frontmatter_and_table(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    meta = {}
    i = 0
    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            line = lines[i]
            m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
            if m:
                k = m.group(1).strip().lower()
                v = m.group(2).strip().strip('"').strip("'")
                meta[k] = v
            i += 1
        if i < len(lines) and lines[i].strip() == "---":
            i += 1

    # Find first markdown table block.
    table = []
    in_table = False
    for line in lines[i:]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table.append(line.rstrip())
            in_table = True
        elif in_table:
            break

    return meta, table


def parse_markdown_table(table_lines):
    if len(table_lines) < 2:
        return [], []
    header = [clean_text(c) for c in table_lines[0].strip().strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [clean_text(c) for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        rows.append(cells)
    return header, rows


def merge_write(out_path, domain, src_lang, tgt_lang, entries):
    existing_entries = {}
    existing_norm = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                oe = old.get("entries", {})
                on = old.get("normalize", {})
                if isinstance(oe, dict):
                    for k, v in oe.items():
                        ks = clean_text(k).lower()
                        vs = clean_text(v)
                        if ks and vs:
                            existing_entries[ks] = vs
                if isinstance(on, dict):
                    for k, v in on.items():
                        ks = clean_text(k).lower()
                        vs = clean_text(v)
                        if ks and vs:
                            existing_norm[ks] = vs
        except Exception:
            pass

    for k, v in entries.items():
        existing_entries[k] = v

    payload = {
        "domain": domain,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "entries": dict(sorted(existing_entries.items(), key=lambda kv: kv[0])),
        "normalize": dict(sorted(existing_norm.items(), key=lambda kv: kv[0])),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(payload["entries"])


def main():
    ap = argparse.ArgumentParser(description="Import Beijerterm markdown glossaries into project JSON glossaries.")
    ap.add_argument(
        "--repo-dir",
        default="datasets/terminology_sources/external/beijerterm",
        help="Local path to beijerterm repository",
    )
    ap.add_argument(
        "--output-dir",
        default="ai_models/translation/glossaries",
        help="Target glossary directory",
    )
    args = ap.parse_args()

    gloss_dir = os.path.join(args.repo_dir, "content", "glossaries")
    if not os.path.isdir(gloss_dir):
        raise SystemExit(f"Beijerterm glossaries folder not found: {gloss_dir}")

    by_pair = defaultdict(dict)
    files_scanned = 0
    files_used = 0

    for root, _, files in os.walk(gloss_dir):
        for name in files:
            if not name.lower().endswith(".md"):
                continue
            path = os.path.join(root, name)
            files_scanned += 1
            meta, table = parse_frontmatter_and_table(path)
            src_lang = norm_lang(meta.get("source_lang", ""))
            tgt_lang = norm_lang(meta.get("target_lang", ""))
            domain = clean_text(meta.get("domain", "general")).lower() or "general"
            if not src_lang or not tgt_lang:
                continue
            if not table:
                continue
            _, rows = parse_markdown_table(table)
            if not rows:
                continue
            files_used += 1
            key = (domain, src_lang, tgt_lang)
            bucket = by_pair[key]
            for cells in rows:
                src = clean_text(cells[0]).lower()
                tgt = clean_text(cells[1])
                if not src or not tgt:
                    continue
                bucket[src] = tgt

    os.makedirs(args.output_dir, exist_ok=True)
    written = []
    total_entries = 0
    for (domain, src_lang, tgt_lang), entries in sorted(by_pair.items()):
        out_path = os.path.join(args.output_dir, f"{domain}_{src_lang}_{tgt_lang}.json")
        n = merge_write(out_path, domain, src_lang, tgt_lang, entries)
        written.append((out_path, n))
        total_entries += n

    print(f"[ok] scanned_md_files: {files_scanned}")
    print(f"[ok] imported_md_files: {files_used}")
    print(f"[ok] glossaries_updated: {len(written)}")
    print(f"[ok] cumulative_entries: {total_entries}")
    for p, n in written[:30]:
        print(f" - {p} ({n})")


if __name__ == "__main__":
    main()
