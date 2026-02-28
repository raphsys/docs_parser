#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict


def norm_lang(lang):
    s = (lang or "").strip().lower()
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


def clean_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def ensure_domain(store, domain):
    if domain not in store:
        store[domain] = {"pairs": defaultdict(dict), "normalize": defaultdict(dict)}


def add_pair(store, domain, src_lang, tgt_lang, src, tgt):
    src = clean_text(src).lower()
    tgt = clean_text(tgt)
    if not src or not tgt:
        return
    ensure_domain(store, domain)
    key = f"{norm_lang(src_lang)}_{norm_lang(tgt_lang)}"
    store[domain]["pairs"][key][src] = tgt


def import_json(path, store, default_domain, default_src, default_tgt):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Already in our target format.
    if isinstance(data, dict) and "entries" in data and "domain" in data:
        domain = clean_text(data.get("domain", default_domain)).lower() or default_domain
        src_lang = norm_lang(data.get("source_lang", default_src))
        tgt_lang = norm_lang(data.get("target_lang", default_tgt))
        entries = data.get("entries", {})
        if isinstance(entries, dict):
            for s, t in entries.items():
                add_pair(store, domain, src_lang, tgt_lang, s, t)
        elif isinstance(entries, list):
            for row in entries:
                if isinstance(row, dict):
                    add_pair(
                        store,
                        domain,
                        row.get("source_lang", src_lang),
                        row.get("target_lang", tgt_lang),
                        row.get("source"),
                        row.get("target"),
                    )
        return
    # Generic list/dict cases.
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            domain = clean_text(row.get("domain", default_domain)).lower() or default_domain
            src_lang = norm_lang(row.get("source_lang", default_src))
            tgt_lang = norm_lang(row.get("target_lang", default_tgt))
            src = row.get("source") or row.get("src") or row.get("term") or row.get("text")
            tgt = row.get("target") or row.get("tgt") or row.get("translation")
            if src and tgt:
                add_pair(store, domain, src_lang, tgt_lang, src, tgt)
    elif isinstance(data, dict):
        # Heuristic for {"term":"translation", ...}
        domain = default_domain
        for s, t in data.items():
            if isinstance(t, str):
                add_pair(store, domain, default_src, default_tgt, s, t)


def import_delimited(path, store, default_domain, default_src, default_tgt, delimiter):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            lower = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            domain = clean_text(lower.get("domain", default_domain)).lower() or default_domain
            src_lang = norm_lang(lower.get("source_lang", lower.get("src_lang", default_src)))
            tgt_lang = norm_lang(lower.get("target_lang", lower.get("tgt_lang", default_tgt)))
            src = (
                lower.get("source")
                or lower.get("src")
                or lower.get("term")
                or lower.get("en")
                or lower.get(src_lang)
            )
            tgt = (
                lower.get("target")
                or lower.get("tgt")
                or lower.get("translation")
                or lower.get(tgt_lang)
            )
            if src and tgt:
                add_pair(store, domain, src_lang, tgt_lang, src, tgt)


def import_tmx(path, store, default_domain, default_src, default_tgt):
    tree = ET.parse(path)
    root = tree.getroot()
    # TMX body/tu/tuv/seg
    for tu in root.findall(".//tu"):
        by_lang = {}
        for tuv in tu.findall("./tuv"):
            lang = (
                tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                or tuv.attrib.get("lang")
                or ""
            )
            seg = tuv.find("./seg")
            text = clean_text(seg.text if seg is not None else "")
            if lang and text:
                by_lang[norm_lang(lang)] = text
        src = by_lang.get(norm_lang(default_src))
        tgt = by_lang.get(norm_lang(default_tgt))
        if src and tgt:
            add_pair(store, default_domain, default_src, default_tgt, src, tgt)


def import_tbx(path, store, default_domain, default_src, default_tgt):
    tree = ET.parse(path)
    root = tree.getroot()
    # TBX basic: termEntry/langSet/tig/term
    for term_entry in root.findall(".//termEntry"):
        by_lang = {}
        for lang_set in term_entry.findall("./langSet"):
            lang = norm_lang(
                lang_set.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                or lang_set.attrib.get("lang")
                or ""
            )
            term = None
            tig = lang_set.find("./tig/term")
            if tig is not None and tig.text:
                term = clean_text(tig.text)
            else:
                alt = lang_set.find("./ntig/termGrp/term")
                if alt is not None and alt.text:
                    term = clean_text(alt.text)
            if lang and term:
                by_lang[lang] = term
        src = by_lang.get(norm_lang(default_src))
        tgt = by_lang.get(norm_lang(default_tgt))
        if src and tgt:
            add_pair(store, default_domain, default_src, default_tgt, src, tgt)


def write_glossaries(store, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for domain, data in sorted(store.items()):
        for pair_key, entries in sorted(data["pairs"].items()):
            src_lang, tgt_lang = pair_key.split("_", 1)
            out_path = os.path.join(out_dir, f"{domain}_{src_lang}_{tgt_lang}.json")
            merged_entries = {}
            merged_normalize = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if isinstance(existing, dict):
                        ee = existing.get("entries", {})
                        nn = existing.get("normalize", {})
                        if isinstance(ee, dict):
                            for k, v in ee.items():
                                ks = clean_text(k).lower()
                                vs = clean_text(v)
                                if ks and vs:
                                    merged_entries[ks] = vs
                        if isinstance(nn, dict):
                            for k, v in nn.items():
                                ks = clean_text(k).lower()
                                vs = clean_text(v)
                                if ks and vs:
                                    merged_normalize[ks] = vs
                except Exception:
                    pass
            # New imported entries override older values for same source key.
            for k, v in entries.items():
                merged_entries[k] = v
            for k, v in data["normalize"].get(tgt_lang, {}).items():
                merged_normalize[k] = v
            payload = {
                "domain": domain,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "entries": dict(sorted(merged_entries.items(), key=lambda kv: kv[0])),
                "normalize": dict(sorted(merged_normalize.items(), key=lambda kv: kv[0])),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            written.append((out_path, len(payload["entries"])))
    return written


def main():
    ap = argparse.ArgumentParser(description="Import terminology files to project glossary format.")
    ap.add_argument(
        "--input-dir",
        default="datasets/terminology_sources",
        help="Directory containing source files (.csv/.tsv/.json/.tmx/.tbx)",
    )
    ap.add_argument(
        "--output-dir",
        default="ai_models/translation/glossaries",
        help="Output glossary directory",
    )
    ap.add_argument("--domain", default="general", help="Default domain when missing")
    ap.add_argument("--source-lang", default="en", help="Default source language")
    ap.add_argument("--target-lang", default="fr", help="Default target language")
    args = ap.parse_args()

    in_dir = args.input_dir
    if not os.path.isdir(in_dir):
        raise SystemExit(f"Input directory not found: {in_dir}")

    store = {}
    counts = defaultdict(int)
    for root, _, files in os.walk(in_dir):
        for name in files:
            path = os.path.join(root, name)
            lower = name.lower()
            try:
                if lower.endswith(".json"):
                    import_json(path, store, args.domain, args.source_lang, args.target_lang)
                    counts["json"] += 1
                elif lower.endswith(".csv"):
                    import_delimited(path, store, args.domain, args.source_lang, args.target_lang, ",")
                    counts["csv"] += 1
                elif lower.endswith(".tsv"):
                    import_delimited(path, store, args.domain, args.source_lang, args.target_lang, "\t")
                    counts["tsv"] += 1
                elif lower.endswith(".tmx"):
                    import_tmx(path, store, args.domain, args.source_lang, args.target_lang)
                    counts["tmx"] += 1
                elif lower.endswith(".tbx"):
                    import_tbx(path, store, args.domain, args.source_lang, args.target_lang)
                    counts["tbx"] += 1
            except Exception as e:
                print(f"[warn] failed to import {path}: {e}")

    written = write_glossaries(store, args.output_dir)
    total_entries = sum(n for _, n in written)
    print(f"[ok] files imported: {dict(counts)}")
    print(f"[ok] glossaries written: {len(written)}")
    print(f"[ok] total entries: {total_entries}")
    for p, n in written[:20]:
        print(f" - {p} ({n})")


if __name__ == "__main__":
    main()
