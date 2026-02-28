#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from translator import DocumentTranslator


TARGET_LANGS = ["es", "de", "it", "pt", "ar", "zh", "ja", "ko"]


def clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def fname_for(domain, src_lang, tgt_lang):
    safe_domain = domain.replace(".", "_").replace("/", "_")
    return f"{safe_domain}_{src_lang}_{tgt_lang}.json"


def main():
    ap = argparse.ArgumentParser(description="Generate en->X subdomain glossaries from en->fr seeds using local CT2.")
    ap.add_argument("--glossary-dir", default="ai_models/translation/glossaries")
    ap.add_argument("--include-science", action="store_true", help="Also process science.* seeds")
    args = ap.parse_args()

    gdir = args.glossary_dir
    if not os.path.isdir(gdir):
        raise SystemExit(f"Glossary dir not found: {gdir}")

    tr = DocumentTranslator()

    prefixes = [
        "economy_",
        "politics_",
        "legal_",
        "medicine_",
        "engineering_",
        "technology_",
        "education_",
        "history_",
        "geography_",
        "biology_",
    ]
    if args.include_science:
        prefixes.append("science_")
    seed_files = []
    for pfx in prefixes:
        seed_files.extend(glob.glob(os.path.join(gdir, f"{pfx}*_en_fr.json")))
    seed_files = sorted(set(seed_files))
    generated = 0
    touched_terms = 0

    for seed_path in seed_files:
        with open(seed_path, "r", encoding="utf-8") as f:
            seed = json.load(f)
        domain = clean(seed.get("domain", "general")).lower()
        if "." not in domain:
            continue
        if (not args.include_science) and domain.startswith("science."):
            continue
        entries = seed.get("entries", {})
        if not isinstance(entries, dict) or not entries:
            continue
        src_lang = clean(seed.get("source_lang", "en")).lower() or "en"
        if src_lang != "en":
            continue

        # subdomain from domain path; fallback to last segment.
        subdomain = domain.split(".", 1)[1] if "." in domain else domain

        for tgt in TARGET_LANGS:
            if tgt == "fr":
                continue
            out_name = fname_for(domain, "en", tgt)
            out_path = os.path.join(gdir, out_name)
            out_entries = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    oe = old.get("entries", {})
                    if isinstance(oe, dict):
                        out_entries.update({clean(k).lower(): clean(v) for k, v in oe.items() if clean(k) and clean(v)})
                except Exception:
                    pass

            for src_term in entries.keys():
                src = clean(src_term).lower()
                if not src:
                    continue
                if src in out_entries:
                    continue
                trm = tr._ct2_translate(src, target_lang=tgt)
                trm = tr._sanitize_translation(trm, src)
                trm = tr._restore_protected_tokens(src, trm)
                trm = tr._normalize_translation(trm, target_lang=tgt, original=src)
                trm = clean(trm)
                if not trm:
                    trm = src
                out_entries[src] = trm
                touched_terms += 1

            payload = {
                "domain": domain,
                "source_lang": "en",
                "target_lang": tgt,
                "entries": dict(sorted(out_entries.items(), key=lambda kv: kv[0])),
                "normalize": {},
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            generated += 1

    print(f"[ok] glossaries_generated_or_updated={generated}")
    print(f"[ok] translated_terms_added={touched_terms}")


if __name__ == "__main__":
    main()
