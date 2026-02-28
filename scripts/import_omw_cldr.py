#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict


LANG3_TO_2 = {
    "eng": "en",
    "fra": "fr",
    "spa": "es",
    "deu": "de",
    "ita": "it",
    "por": "pt",
    "arb": "ar",
    "hin": "hi",
    "jpn": "ja",
    "kor": "ko",
    "zho": "zh",
    "cmn": "zh",
    "rus": "ru",
    "ukr": "uk",
    "pol": "pl",
    "nld": "nl",
    "tur": "tr",
    "vie": "vi",
    "tha": "th",
    "ind": "id",
    "msa": "ms",
    "zsm": "ms",
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
}


def clean_text(s):
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def norm_lang3(code):
    c = clean_text(code).lower()
    return c


def to_lang2(code3):
    c3 = norm_lang3(code3)
    return LANG3_TO_2.get(c3, c3)


def parse_tab_lemmas(path):
    # returns iterator of (synset, lemma)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            synset = clean_text(parts[0])
            field = clean_text(parts[1]).lower()
            val = clean_text(parts[2])
            if not synset or not val:
                continue
            if not field.endswith(":lemma"):
                continue
            yield synset, val


def load_eng_lemmas(eng_tab):
    out = defaultdict(list)
    for synset, lemma in parse_tab_lemmas(eng_tab):
        l = clean_text(lemma)
        if not l:
            continue
        if l not in out[synset]:
            out[synset].append(l)
    return out


def merge_json_glossary(path, domain, src_lang, tgt_lang, new_entries):
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

    old_entries.update(new_entries)
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
    ap = argparse.ArgumentParser(description="Import OMW CLDR lemmas as large EN->X general glossaries.")
    ap.add_argument(
        "--repo-dir",
        default="datasets/terminology_sources/external/omw-data",
        help="Local omw-data repository path",
    )
    ap.add_argument(
        "--output-dir",
        default="ai_models/translation/glossaries",
        help="Glossary output directory",
    )
    ap.add_argument(
        "--domain",
        default="general",
        help="Domain for generated glossaries",
    )
    args = ap.parse_args()

    cldr_dir = os.path.join(args.repo_dir, "wns", "cldr")
    eng_tab = os.path.join(cldr_dir, "wn-cldr-eng.tab")
    if not os.path.isfile(eng_tab):
        raise SystemExit(f"English CLDR tab not found: {eng_tab}")

    eng_by_synset = load_eng_lemmas(eng_tab)
    if not eng_by_synset:
        raise SystemExit("No English lemmas loaded from CLDR.")

    os.makedirs(args.output_dir, exist_ok=True)
    updated = []
    total_pairs = 0
    files = sorted([f for f in os.listdir(cldr_dir) if f.startswith("wn-cldr-") and f.endswith(".tab")])
    for fname in files:
        lang3 = fname[len("wn-cldr-") : -len(".tab")]
        if lang3 == "eng":
            continue
        tgt_lang = to_lang2(lang3)
        if tgt_lang == "en":
            continue
        tab_path = os.path.join(cldr_dir, fname)
        entries = {}
        for synset, tgt_lemma in parse_tab_lemmas(tab_path):
            eng_terms = eng_by_synset.get(synset)
            if not eng_terms:
                continue
            tgt = clean_text(tgt_lemma)
            if not tgt:
                continue
            for src in eng_terms:
                s = clean_text(src).lower()
                if not s:
                    continue
                # Keep first mapping encountered for determinism and speed.
                if s not in entries:
                    entries[s] = tgt
        if not entries:
            continue

        out_path = os.path.join(args.output_dir, f"{args.domain}_en_{tgt_lang}.json")
        n = merge_json_glossary(out_path, args.domain, "en", tgt_lang, entries)
        total_pairs += n
        updated.append((out_path, n))

    print(f"[ok] target_files: {len(updated)}")
    print(f"[ok] cumulative_entries: {total_pairs}")
    for p, n in updated[:50]:
        print(f" - {p} ({n})")


if __name__ == "__main__":
    main()
