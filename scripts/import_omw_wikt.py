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
}


def clean_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def to_lang(code):
    c = clean_text(code).lower()
    return LANG3_TO_2.get(c, c)


def parse_tab_lemmas(path):
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


def load_src_map(src_file):
    out = defaultdict(list)
    for synset, lemma in parse_tab_lemmas(src_file):
        if lemma not in out[synset]:
            out[synset].append(lemma)
    return out


def merge_json(path, domain, src_lang, tgt_lang, entries):
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


def main():
    ap = argparse.ArgumentParser(description="Import massive EN->X glossaries from OMW Wiktionary tabs.")
    ap.add_argument("--repo-dir", default="datasets/terminology_sources/external/omw-data")
    ap.add_argument("--output-dir", default="ai_models/translation/glossaries")
    ap.add_argument("--domain", default="general")
    ap.add_argument("--max-target-files", type=int, default=0, help="0 means all files")
    args = ap.parse_args()

    wikt_dir = os.path.join(args.repo_dir, "wns", "wikt")
    src_file = os.path.join(wikt_dir, "wn-wikt-eng.tab")
    if not os.path.isfile(src_file):
        raise SystemExit(f"Source file not found: {src_file}")

    src_map = load_src_map(src_file)
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(wikt_dir) if f.startswith("wn-wikt-") and f.endswith(".tab")])
    target_files = [f for f in files if f not in {"wn-wikt-eng.tab", "wn-wikt-pwn.tab"}]
    if args.max_target_files > 0:
        target_files = target_files[: args.max_target_files]

    updated = []
    total_entries = 0
    for fname in target_files:
        code = fname[len("wn-wikt-") : -len(".tab")]
        tgt_lang = to_lang(code)
        if tgt_lang == "en":
            continue

        entries = {}
        path = os.path.join(wikt_dir, fname)
        for synset, tgt_lemma in parse_tab_lemmas(path):
            src_terms = src_map.get(synset)
            if not src_terms:
                continue
            tgt = clean_text(tgt_lemma)
            if not tgt:
                continue
            for src in src_terms:
                s = clean_text(src).lower()
                if not s:
                    continue
                if s not in entries:
                    entries[s] = tgt
        if not entries:
            continue

        out_path = os.path.join(args.output_dir, f"{args.domain}_en_{tgt_lang}.json")
        n = merge_json(out_path, args.domain, "en", tgt_lang, entries)
        total_entries += n
        updated.append((out_path, n))

    print(f"[ok] target_files: {len(updated)}")
    print(f"[ok] cumulative_entries: {total_entries}")
    for p, n in updated[:60]:
        print(f" - {p} ({n})")


if __name__ == "__main__":
    main()
