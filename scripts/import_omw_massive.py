#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict


LANG3_TO_2 = {
    "eng": "en",
    "fra": "fr",
    "fre": "fr",
    "spa": "es",
    "deu": "de",
    "ger": "de",
    "ita": "it",
    "por": "pt",
    "arb": "ar",
    "zho": "zh",
    "cmn": "zh",
    "jpn": "ja",
    "kor": "ko",
    "hin": "hi",
    "rus": "ru",
    "ukr": "uk",
    "nld": "nl",
    "tur": "tr",
    "pol": "pl",
    "ces": "cs",
    "slk": "sk",
    "ron": "ro",
    "ell": "el",
    "bul": "bg",
    "hrv": "hr",
    "srp": "sr",
    "fin": "fi",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "nob": "no",
    "nno": "no",
    "cat": "ca",
    "glg": "gl",
    "eus": "eu",
    "hun": "hu",
    "lav": "lv",
    "lit": "lt",
    "est": "et",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "ben": "bn",
    "guj": "gu",
    "pan": "pa",
    "urd": "ur",
    "nep": "ne",
    "sin": "si",
    "fas": "fa",
    "heb": "he",
    "ind": "id",
    "zsm": "ms",
    "msa": "ms",
    "vie": "vi",
    "tha": "th",
}


def clean_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_lang(code):
    c = clean_text(code).lower()
    c = LANG3_TO_2.get(c, c)
    # keep only safe file/code chars
    c = re.sub(r"[^a-z0-9\-]", "", c)
    return c


def parse_lemma_tab(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            synset = clean_text(parts[0])
            tag = clean_text(parts[1]).lower()
            lemma = clean_text(parts[2])
            if not synset or not lemma:
                continue
            if not tag.endswith(":lemma"):
                continue
            yield synset, lemma


def merge_glossary(path, domain, src_lang, tgt_lang, entries):
    old_entries = {}
    old_norm = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                oe = old.get("entries", {})
                on = old.get("normalize", {})
                if isinstance(oe, dict):
                    for k, v in oe.items():
                        ks = clean_text(k).lower()
                        vs = clean_text(v)
                        if ks and vs:
                            old_entries[ks] = vs
                if isinstance(on, dict):
                    for k, v in on.items():
                        ks = clean_text(k).lower()
                        vs = clean_text(v)
                        if ks and vs:
                            old_norm[ks] = vs
        except Exception:
            pass
    old_entries.update(entries)
    payload = {
        "domain": domain,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "entries": dict(sorted(old_entries.items(), key=lambda kv: kv[0])),
        "normalize": dict(sorted(old_norm.items(), key=lambda kv: kv[0])),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(payload["entries"])


def guess_lang_from_filename(path):
    name = os.path.basename(path)
    # wn-data-XXX.tab
    m = re.match(r"^wn-data-([A-Za-z0-9\-]+)\.tab$", name)
    if m:
        return normalize_lang(m.group(1))
    # wn-<src>-XXX.tab
    m = re.match(r"^wn-[A-Za-z0-9\-]+-([A-Za-z0-9\-\?]+)\.tab$", name)
    if m:
        return normalize_lang(m.group(1))
    return ""


def is_english_tab(path):
    name = os.path.basename(path).lower()
    return name in {"wn-data-eng.tab", "wn-wikt-eng.tab", "wn-cldr-eng.tab"}


def is_ignored_tab(path):
    name = os.path.basename(path).lower()
    if name.endswith("changes.tab"):
        return True
    if name in {"wn-wikt-pwn.tab"}:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Massive OMW import: all compatible tab files EN->X.")
    ap.add_argument("--repo-dir", default="datasets/terminology_sources/external/omw-data")
    ap.add_argument("--output-dir", default="ai_models/translation/glossaries")
    ap.add_argument("--domain", default="general")
    args = ap.parse_args()

    wns_dir = os.path.join(args.repo_dir, "wns")
    if not os.path.isdir(wns_dir):
        raise SystemExit(f"Missing directory: {wns_dir}")

    tab_files = []
    for root, _, files in os.walk(wns_dir):
        for name in files:
            if name.endswith(".tab"):
                tab_files.append(os.path.join(root, name))
    tab_files.sort()

    eng_tabs = [p for p in tab_files if is_english_tab(p)]
    if not eng_tabs:
        raise SystemExit("No English source tabs found in OMW.")

    # Build EN synset->lemmas map by combining all EN sources.
    en_map = defaultdict(list)
    for p in eng_tabs:
        for synset, lemma in parse_lemma_tab(p):
            lemma = clean_text(lemma)
            if lemma and lemma not in en_map[synset]:
                en_map[synset].append(lemma)

    os.makedirs(args.output_dir, exist_ok=True)
    by_tgt = defaultdict(dict)

    scanned = 0
    used = 0
    for p in tab_files:
        scanned += 1
        if is_ignored_tab(p) or is_english_tab(p):
            continue
        tgt_lang = guess_lang_from_filename(p)
        if not tgt_lang or tgt_lang == "en":
            continue
        used += 1
        bucket = by_tgt[tgt_lang]
        for synset, tgt_lemma in parse_lemma_tab(p):
            src_terms = en_map.get(synset)
            if not src_terms:
                continue
            tgt = clean_text(tgt_lemma)
            if not tgt:
                continue
            for src in src_terms:
                s = clean_text(src).lower()
                if not s:
                    continue
                if s not in bucket:
                    bucket[s] = tgt

    report = []
    cumulative = 0
    for tgt_lang, entries in sorted(by_tgt.items()):
        if not entries:
            continue
        out_path = os.path.join(args.output_dir, f"{args.domain}_en_{tgt_lang}.json")
        n = merge_glossary(out_path, args.domain, "en", tgt_lang, entries)
        cumulative += n
        report.append((out_path, n))

    print(f"[ok] tab_files_scanned: {scanned}")
    print(f"[ok] tab_files_used: {used}")
    print(f"[ok] output_glossaries: {len(report)}")
    print(f"[ok] cumulative_entries: {cumulative}")
    for p, n in report[:80]:
        print(f" - {p} ({n})")


if __name__ == "__main__":
    main()
