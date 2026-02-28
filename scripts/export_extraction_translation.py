#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter

import fitz

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_pdf_extractor import NativePDFExtractor
from translator import DocumentTranslator


def normalize_spaces(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def role_of_block(block):
    return (block.get("role") or "body").lower()


def phrase_texts(block):
    out = []
    for line in block.get("lines", []):
        for phrase in line.get("phrases", []):
            t = normalize_spaces(phrase.get("texte", ""))
            if t:
                out.append((line, phrase, t))
    return out


def sentence_units_from_block(block):
    seq = [t for _, _, t in phrase_texts(block)]
    units = []
    cur = ""
    for frag in seq:
        t = normalize_spaces(frag)
        if not t:
            continue
        if not cur:
            cur = t
        else:
            if cur.endswith("-"):
                cur = cur[:-1] + t
            else:
                cur = f"{cur} {t}"
        if re.search(r"[\.!\?]\s*$", cur):
            units.append(normalize_spaces(cur))
            cur = ""
    if cur:
        units.append(normalize_spaces(cur))
    return units


def classify_item(text, tr: DocumentTranslator, block_role: str):
    s = normalize_spaces(text)
    if not s:
        return "skip"
    if tr._is_protected_segment(s, block_role=block_role):
        return "symbole"
    if tr._looks_like_sentence(s):
        return "phrase"

    # If sentence split failed but we have >1 lexical token, keep as expression.
    lex = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", s)
    if len(lex) >= 2:
        return "expression"
    if len(lex) == 1:
        return "mot"

    # numbers, punctuations, symbols, greek, formulas
    return "symbole"


def split_to_residual_units(text, tr: DocumentTranslator, block_role: str):
    # For non-sentences, split into expression/mot/symbole to preserve hierarchy.
    parts = [p for p in tr._split_expressions(text) if p is not None and p != ""]
    result = []
    for p in parts:
        p_norm = normalize_spaces(p)
        if not p_norm:
            continue
        if tr._is_separator_token(p):
            continue
        kind = classify_item(p_norm, tr, block_role)
        if kind in {"phrase", "expression"}:
            result.append((kind if kind != "phrase" else "expression", p_norm))
            continue
        # word/symbol granular split
        wparts = [w for w in tr._split_words_with_separators(p_norm) if w is not None and w != ""]
        for w in wparts:
            w_norm = normalize_spaces(w)
            if not w_norm or tr._is_separator_token(w):
                continue
            k = classify_item(w_norm, tr, block_role)
            if k == "phrase":
                k = "expression"
            result.append((k, w_norm))
    return result


def translate_item(text, tr: DocumentTranslator, target_lang: str, context: str, role: str, domain: str, subdomain: str = ""):
    s = normalize_spaces(text)
    if not s:
        return s
    if tr._is_protected_segment(s, block_role=role):
        return s
    t = tr._translate_text_hierarchical(
        s,
        target_lang=target_lang,
        block_context=context,
        block_role=role,
        domain=domain,
        subdomain=subdomain,
    )
    t = tr._restore_protected_tokens(s, t)
    t = tr._normalize_translation(t, target_lang=target_lang, original=s, context_text=context)
    t = tr._apply_domain_glossary(t, source_text=s, target_lang=target_lang, domain=domain, subdomain=subdomain)
    return t


def _word_tokens(text):
    return re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", normalize_spaces(text))


def phrase_quality(source, translation, role, target_lang_code, tr: DocumentTranslator, is_protected=False):
    src = normalize_spaces(source)
    trn = normalize_spaces(translation)
    profile = tr.get_translation_profile(target_lang_code)
    qpen = profile.get("quality_penalties", {}) if isinstance(profile, dict) else {}
    qthr = profile.get("quality_thresholds", {}) if isinstance(profile, dict) else {}
    qrules = profile.get("quality_rules", {}) if isinstance(profile, dict) else {}
    pedit = profile.get("post_edit", {}) if isinstance(profile, dict) else {}

    def pen(name, default):
        return int(qpen.get(name, default))

    flags = []
    score = 100

    if not trn:
        return 0, ["empty_translation"], "critical"
    if trn == src and not is_protected:
        flags.append("unchanged_translation")
        score -= pen("unchanged_translation", 40)

    src_words = _word_tokens(src)
    tr_words = _word_tokens(trn)
    min_src_words = int(qrules.get("min_source_words_for_ratio_check", 5))
    short_ratio = float(qrules.get("short_ratio", 0.45))
    long_ratio = float(qrules.get("long_ratio", 2.7))
    if len(src_words) >= min_src_words and len(tr_words) < max(3, int(short_ratio * len(src_words))):
        flags.append("too_short")
        score -= pen("too_short", 20)
    if len(tr_words) > max(60, int(long_ratio * max(1, len(src_words)))):
        flags.append("too_long")
        score -= pen("too_long", 15)

    en_markers = tr._language_marker_counts(trn, "en")
    tgt_markers = tr._language_marker_counts(trn, target_lang_code)
    if target_lang_code != "en":
        if tgt_markers == 0 and en_markers >= 2:
            flags.append("english_leak")
            score -= pen("english_leak", 25)
        elif en_markers > tgt_markers + 3:
            flags.append("mixed_language")
            score -= pen("mixed_language", 12)

    if re.search(r"[!?]{2,}", trn):
        flags.append("punctuation_noise")
        score -= pen("punctuation_noise", 10)

    if target_lang_code == "fr":
        # Editorial fluency checks for French.
        literal_patterns = pedit.get("literal_patterns") or [
            r"\bce processus est appelé\b",
            r"\ble gradient détermine seulement\b",
            r"\bce peut être un pas\b",
            r"\bnous relançons le processus\b",
            r"\bnous choisissons cette voie\b",
            r"\bcela nous amène\b",
            r"\bvous finissez au point\b",
        ]
        literal_hits = sum(1 for p in literal_patterns if re.search(p, trn, flags=re.IGNORECASE))
        if literal_hits:
            flags.append("literal_style")
            score -= min(pen("literal_style_max", 18), pen("literal_style_per_hit", 6) * literal_hits)

        wcon = pedit.get("weak_connectors") or ["et ainsi de suite", "pour l'instant", "pas tout à fait"]
        weak_connectors = len(re.findall(r"\b(" + "|".join(re.escape(x) for x in wcon) + r")\b", trn, flags=re.IGNORECASE))
        if weak_connectors >= 2:
            flags.append("weak_style_connectors")
            score -= pen("weak_style_connectors", 8)

    # Source appears truncated -> review, but softer penalty.
    if src and not re.search(r"[\.!\?]\s*$", src) and len(src_words) >= 8:
        flags.append("source_fragment")
        score -= pen("source_fragment", 6)

    # Technical parenthesis mismatch can indicate broken fragment.
    if src.count("(") != trn.count("(") or src.count(")") != trn.count(")"):
        flags.append("parenthesis_mismatch")
        score -= pen("parenthesis_mismatch", 8)

    if role in {"header", "footer"} and len(src_words) <= 6:
        score = max(score, 70)

    score = max(0, min(100, score))
    critical_below = int(qthr.get("critical_below", 72))
    review_below = int(qthr.get("review_below", 88))
    if score < critical_below:
        status = "critical"
    elif score < review_below:
        status = "review"
    else:
        status = "ok"
    return score, flags, status


def summarize_span_style(span):
    st = span.get("style") or {}
    flags = st.get("flags") or {}
    return {
        "font": st.get("font"),
        "size": st.get("size"),
        "color": st.get("color"),
        "bold": bool(flags.get("bold", False)),
        "italic": bool(flags.get("italic", False)),
        "serif": bool(flags.get("serif", False)),
        "uppercase": bool(flags.get("uppercase", False)),
    }


def main():
    ap = argparse.ArgumentParser(description="Export extraction + hierarchical translation with context/domain.")
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--page", type=int, default=0, help="0-based page index")
    ap.add_argument("--target-lang", default="French", help="Target language")
    ap.add_argument("--out-prefix", default="", help="Output prefix (default: based on PDF name)")
    args = ap.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF introuvable: {pdf_path}")

    base = args.out_prefix or os.path.splitext(os.path.basename(pdf_path))[0]
    out_json = os.path.join("ocr_results", f"{base}_full_extraction_translation_with_context.json")
    out_txt = os.path.join("ocr_results", f"{base}_full_extraction_translation_with_context.txt")

    os.makedirs("ocr_results", exist_ok=True)

    doc = fitz.open(pdf_path)
    if args.page < 0 or args.page >= len(doc):
        raise SystemExit(f"Page invalide {args.page}; pages disponibles: 0..{len(doc)-1}")
    page = doc[args.page]

    # Use same raster scale as pipeline reference (150 DPI).
    dpi = 150
    sx = dpi / 72.0
    sy = dpi / 72.0

    extractor = NativePDFExtractor()
    native = extractor.extract_page(page, sx=sx, sy=sy)
    blocks = native.get("blocks", [])

    tr = DocumentTranslator()
    target_lang_code = tr._normalize_lang_code(args.target_lang)

    all_block_text = []
    for b in blocks:
        su = sentence_units_from_block(b)
        txt = normalize_spaces(" ".join(su if su else [t for _, _, t in phrase_texts(b)]))
        if txt:
            all_block_text.append(txt)
    global_context = normalize_spaces(" ".join(all_block_text))
    global_domain = tr._detect_domain(global_context)

    items = {"phrases": [], "expressions": [], "mots": [], "symboles": []}
    block_summaries = []
    domain_counter = Counter()
    phrase_quality_counter = Counter()

    for bi, b in enumerate(blocks):
        b_role = role_of_block(b)
        b_lines = phrase_texts(b)
        b_sentences = sentence_units_from_block(b)
        b_ctx = normalize_spaces(" ".join(b_sentences if b_sentences else [t for _, _, t in b_lines]))
        b_domain = tr._detect_domain(b_ctx)
        b_subdomain = tr._detect_subdomain(b_ctx, domain=b_domain)
        domain_counter[b_domain] += 1

        style_samples = []
        for _, phr, _ in b_lines:
            for sp in phr.get("spans", []):
                if len(style_samples) >= 3:
                    break
                style_samples.append(summarize_span_style(sp))
            if len(style_samples) >= 3:
                break

        block_summaries.append(
            {
                "block_id": b.get("id", f"b_{bi}"),
                "bbox": b.get("bbox"),
                "source": b.get("source", "native"),
                "role": b_role,
                "domain": b_domain,
                "subdomain": b_subdomain,
                "context": b_ctx[:320],
                "style_samples": style_samples,
                "line_count": len(b.get("lines", [])),
            }
        )

        src_items = b_sentences if b_sentences else [t for _, _, t in b_lines]
        paragraph_prev_translations = []
        for txt in src_items:
            kind = classify_item(txt, tr, b_role)
            if kind == "phrase":
                tr_txt = translate_item(txt, tr, args.target_lang, b_ctx, b_role, b_domain, b_subdomain)
                tr_txt = tr.post_edit_paragraph_sentence(
                    tr_txt,
                    target_lang=args.target_lang,
                    source_text=txt,
                    context_text=b_ctx,
                    previous_translations=paragraph_prev_translations,
                )
                score, flags, status = phrase_quality(
                    txt,
                    tr_txt,
                    role=b_role,
                    target_lang_code=target_lang_code,
                    tr=tr,
                    is_protected=tr._is_protected_segment(txt, block_role=b_role),
                )
                revised = False
                if status == "critical":
                    # Auto-revision pass: direct model translation without hierarchical splitting,
                    # then same normalization/post-edit pipeline.
                    alt = tr._ct2_translate(txt, target_lang=args.target_lang)
                    alt = tr._restore_protected_tokens(txt, alt)
                    alt = tr._normalize_translation(alt, target_lang=args.target_lang, original=txt, context_text=b_ctx)
                    alt = tr._apply_domain_glossary(
                        alt, source_text=txt, target_lang=args.target_lang, domain=b_domain, subdomain=b_subdomain
                    )
                    alt = tr.post_edit_paragraph_sentence(
                        alt,
                        target_lang=args.target_lang,
                        source_text=txt,
                        context_text=b_ctx,
                        previous_translations=paragraph_prev_translations,
                    )
                    alt_score, alt_flags, alt_status = phrase_quality(
                        txt,
                        alt,
                        role=b_role,
                        target_lang_code=target_lang_code,
                        tr=tr,
                        is_protected=tr._is_protected_segment(txt, block_role=b_role),
                    )
                    if alt_score >= score + 8:
                        tr_txt = alt
                        score, flags, status = alt_score, alt_flags, alt_status
                        revised = True

                paragraph_prev_translations.append(tr_txt)
                phrase_quality_counter[status] += 1
                items["phrases"].append(
                    {
                        "block_id": b.get("id"),
                        "domain": b_domain,
                        "subdomain": b_subdomain,
                        "context": b_ctx[:320],
                        "source": txt,
                        "translation": tr_txt,
                        "quality_score": score,
                        "quality_status": status,
                        "quality_flags": flags,
                        "auto_revised": revised,
                    }
                )
                continue

            residuals = split_to_residual_units(txt, tr, b_role)
            for rk, rv in residuals:
                if rk == "skip":
                    continue
                key = "symboles"
                if rk == "expression":
                    key = "expressions"
                elif rk == "mot":
                    key = "mots"
                elif rk == "symbole":
                    key = "symboles"
                items[key].append(
                    {
                        "block_id": b.get("id"),
                        "domain": b_domain,
                        "subdomain": b_subdomain,
                        "context": b_ctx[:320],
                        "source": rv,
                        "translation": translate_item(rv, tr, args.target_lang, b_ctx, b_role, b_domain, b_subdomain),
                    }
                )

    # Deduplicate by (category, block, source) preserving order.
    for k in list(items.keys()):
        seen = set()
        dedup = []
        for it in items[k]:
            sig = (it.get("block_id"), it.get("source"))
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(it)
        items[k] = dedup

    payload = {
        "document": {
            "pdf": pdf_path,
            "page_index": args.page,
            "dpi": dpi,
            "dimensions": {
                "width": int(page.rect.width * sx),
                "height": int(page.rect.height * sy),
            },
            "native_blocks": len(blocks),
            "ocr_ai_blocks": 0,
        },
        "analysis": {
            "global_context": global_context[:1200],
            "global_domain": global_domain,
            "block_domain_distribution": dict(domain_counter),
            "phrase_quality_distribution": dict(phrase_quality_counter),
        },
        "counts": {
            "blocks": len(blocks),
            "phrases": len(items["phrases"]),
            "expressions": len(items["expressions"]),
            "mots": len(items["mots"]),
            "symboles": len(items["symboles"]),
            "items_total": sum(len(v) for v in items.values()),
        },
        "blocks": block_summaries,
        "items": items,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"DOC: {payload['document']['dimensions']['width']}x{payload['document']['dimensions']['height']} | DPI: {dpi}")
    lines.append(f"FONTS: {len(blocks)} blocs natifs | OCR: 0 blocs IA")
    lines.append(
        "COUNTS blocks={blocks} items={items_total} phrase={phrases} expression={expressions} mot={mots} symbole={symboles}".format(**payload["counts"])  # noqa: E501
    )
    lines.append("")
    lines.append(f"GLOBAL_DOMAIN: {global_domain}")
    lines.append(f"GLOBAL_CONTEXT: {payload['analysis']['global_context']}")
    lines.append("")
    lines.append("--- BLOCS (attributs + contexte + domaine) ---")
    for b in block_summaries:
        lines.append(
            f"[{b['block_id']}] role={b['role']} source={b['source']} domain={b['domain']} subdomain={b.get('subdomain','')} bbox={b['bbox']} lines={b['line_count']}"
        )
        lines.append(f"  context: {b['context']}")
        for si, st in enumerate(b["style_samples"]):
            lines.append(
                f"  style#{si+1}: font={st['font']} size={st['size']} color={st['color']} bold={st['bold']} italic={st['italic']} serif={st['serif']}"
            )

    def add_items(title, key):
        lines.append("")
        lines.append(f"--- {title} ---")
        for idx, it in enumerate(items[key], start=1):
            if key == "phrases":
                q = f"score={it.get('quality_score')} status={it.get('quality_status')}"
                flags = ",".join(it.get("quality_flags") or [])
                rev = " revised=1" if it.get("auto_revised") else ""
                lines.append(f"[{idx}] ({it['domain']}/{it.get('subdomain','')}) {q}{rev} flags=[{flags}] {it['source']}")
            else:
                lines.append(f"[{idx}] ({it['domain']}/{it.get('subdomain','')}) {it['source']}")
            lines.append(f"    -> {it['translation']}")

    add_items("PHRASES", "phrases")
    add_items("EXPRESSIONS (hors phrases)", "expressions")
    add_items("MOTS (hors phrases/expressions)", "mots")
    add_items("LETTRES/CHIFFRES/SYMBOLES (restants)", "symboles")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[ok] JSON: {out_json}")
    print(f"[ok] TXT:  {out_txt}")
    print(f"[ok] domain={global_domain} counts={payload['counts']}")


if __name__ == "__main__":
    main()
