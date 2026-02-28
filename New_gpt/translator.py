import os
import re
import json
import unicodedata
from typing import Optional

try:
    import ctranslate2
except Exception:
    ctranslate2 = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

class DocumentTranslator:
    def __init__(self, backend: Optional[str] = None):
        print(f"Chargement du traducteur hiérarchique...")
        self.backend = (backend or os.getenv("TRANSLATOR_BACKEND", "ctranslate2")).strip().lower()
        self._cache = {}
        self._ct2_translator = None
        self._ct2_tokenizer = None
        self._fallback_ct2_translator = None
        self._fallback_ct2_tokenizer = None
        self._model_family = os.getenv("TRANSLATOR_MODEL_FAMILY", "auto").strip().lower()
        self._fallback_model_family = os.getenv("TRANSLATOR_FALLBACK_MODEL_FAMILY", "auto").strip().lower()
        self._strict_glossary = os.getenv("TRANSLATOR_STRICT_GLOSSARY", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._force_terms_in_sentences = os.getenv("TRANSLATOR_FORCE_TERMS_IN_SENTENCES", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._use_general_glossary = os.getenv("TRANSLATOR_USE_GENERAL_GLOSSARY", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._post_edit_enabled = os.getenv("TRANSLATOR_POST_EDIT", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._legacy_fr_post_edit = os.getenv("TRANSLATOR_FR_POST_EDIT", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._fr_strict_quality = os.getenv("TRANSLATOR_FR_STRICT_QUALITY", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._strict_gate = os.getenv("TRANSLATION_GATING_STRICT", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._profiles_path = os.getenv("TRANSLATION_PROFILES_PATH", "ai_models/translation/translation_profiles.json")
        self._profiles = self._load_translation_profiles()
        self._domain_glossaries = self._build_domain_glossaries()
        self._load_external_glossaries()
        if self.backend != "ctranslate2":
            raise RuntimeError(
                "Backend non supporté. "
                "Ce projet utilise uniquement CTranslate2 (M2M100) pour la traduction."
            )
        self._init_ct2_backend()

    def _init_ct2_backend(self):
        if ctranslate2 is None or AutoTokenizer is None:
            raise RuntimeError(
                "CTranslate2/Transformers indisponibles. "
                "Installe 'ctranslate2' et 'transformers' dans l'env actif."
            )
        model_dir = os.getenv("CT2_MODEL_DIR", "ai_models/translation/nllb_200_distilled_600m_ct2_int8")
        tokenizer_dir = os.getenv("CT2_TOKENIZER_DIR", "ai_models/translation/nllb_200_distilled_600m_tokenizer")
        if not os.path.isdir(model_dir):
            raise RuntimeError(
                f"Modèle CTranslate2 introuvable: {model_dir}. "
                "Prépare le modèle dans ai_models/translation/ (conversion int8)."
            )
        if not os.path.isdir(tokenizer_dir):
            raise RuntimeError(
                f"Tokenizer introuvable: {tokenizer_dir}. "
                "Télécharge/copie un tokenizer compatible dans ai_models/translation/."
            )
        inter_threads = int(os.getenv("CT2_INTER_THREADS", "1"))
        intra_threads = int(os.getenv("CT2_INTRA_THREADS", "4"))
        self._ct2_translator = ctranslate2.Translator(
            model_dir,
            device="cpu",
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
        self._ct2_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
        self._model_family = self._resolve_model_family(
            model_dir,
            tokenizer_dir,
            tokenizer=self._ct2_tokenizer,
            preferred=os.getenv("TRANSLATOR_MODEL_FAMILY", "auto"),
            allow_primary_env=True,
        )
        print(f"Traduction CT2 model_family: {self._model_family}")
        self._init_fallback_backend()
        self._cache = {}

    def _strip_invisible_chars(self, text):
        s = (text or "")
        s = re.sub(r"[\u00AD\u200B-\u200F\u2060\uFEFF]", "", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
        return s

    def _contains_invisible_chars(self, text):
        s = (text or "")
        return bool(re.search(r"[\u00AD\u200B-\u200F\u2060\uFEFF]", s))

    def _default_translation_profiles(self):
        return {
            "default": {
                "quality_thresholds": {"critical_below": 72, "review_below": 88},
                "quality_penalties": {
                    "unchanged_translation": 40,
                    "too_short": 20,
                    "too_long": 15,
                    "english_leak": 25,
                    "mixed_language": 12,
                    "punctuation_noise": 10,
                    "source_fragment": 6,
                    "parenthesis_mismatch": 8,
                    "weak_style_connectors": 8,
                    "literal_style_per_hit": 6,
                    "literal_style_max": 18,
                },
                "quality_rules": {"min_source_words_for_ratio_check": 5, "short_ratio": 0.45, "long_ratio": 2.7},
                "post_edit": {
                    "generic_cleanup": True,
                    "generic_replacements": [],
                    "literal_patterns": [],
                    "weak_connectors": [],
                },
            },
            "fr": {
                "post_edit": {
                    "generic_replacements": [
                        {"pattern": "\\bLaissez-nous utiliser\\b", "replace": "Utilisons"},
                        {"pattern": "\\bLaissez-les jeter un oeil à\\b", "replace": "Examinons"},
                        {"pattern": "\\bLaissez-les jeter un œil à\\b", "replace": "Examinons"},
                        {"pattern": "\\bMaintenant il ya\\b", "replace": "Maintenant, il y a"},
                        {"pattern": "\\bva continuer osciller\\b", "replace": "continuera à osciller"},
                    ],
                    "literal_patterns": [
                        "\\bce processus est appelé\\b",
                        "\\ble gradient détermine seulement\\b",
                        "\\bce peut être un pas\\b",
                        "\\bnous relançons le processus\\b",
                        "\\bnous choisissons cette voie\\b",
                        "\\bcela nous amène\\b",
                        "\\bvous finissez au point\\b",
                    ],
                    "weak_connectors": ["et ainsi de suite", "pour l'instant", "pas tout à fait"],
                }
            },
        }

    def _load_translation_profiles(self):
        profiles = self._default_translation_profiles()
        path = self._profiles_path
        if not path or not os.path.isfile(path):
            return profiles
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return profiles
            for k, v in payload.items():
                if not isinstance(v, dict):
                    continue
                base = profiles.get(k, {})
                merged = dict(base)
                for kk, vv in v.items():
                    if isinstance(vv, dict) and isinstance(merged.get(kk), dict):
                        m2 = dict(merged.get(kk, {}))
                        m2.update(vv)
                        merged[kk] = m2
                    else:
                        merged[kk] = vv
                profiles[k] = merged
            return profiles
        except Exception:
            return profiles

    def get_translation_profile(self, target_lang):
        code = self._normalize_lang_code(target_lang)
        base = dict(self._profiles.get("default", {}))
        lang = self._profiles.get(code, {})
        out = dict(base)
        for k, v in lang.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                d = dict(out[k])
                d.update(v)
                out[k] = d
            else:
                out[k] = v
        return out

    def _init_fallback_backend(self):
        # Auto-load multilingual fallback when primary model is pair-specific (e.g., Marian EN->FR).
        if self._model_family != "marian":
            return
        f_model_dir = os.getenv("CT2_FALLBACK_MODEL_DIR", "ai_models/translation/m2m100_418m_ct2_int8")
        f_tokenizer_dir = os.getenv("CT2_FALLBACK_TOKENIZER_DIR", "ai_models/translation/m2m100_418m_tokenizer")
        if not (os.path.isdir(f_model_dir) and os.path.isdir(f_tokenizer_dir)):
            return
        try:
            inter_threads = int(os.getenv("CT2_INTER_THREADS", "1"))
            intra_threads = int(os.getenv("CT2_INTRA_THREADS", "4"))
            self._fallback_ct2_translator = ctranslate2.Translator(
                f_model_dir,
                device="cpu",
                inter_threads=inter_threads,
                intra_threads=intra_threads,
            )
            self._fallback_ct2_tokenizer = AutoTokenizer.from_pretrained(f_tokenizer_dir, use_fast=False)
            self._fallback_model_family = self._resolve_model_family(
                f_model_dir,
                f_tokenizer_dir,
                tokenizer=self._fallback_ct2_tokenizer,
                preferred=os.getenv("TRANSLATOR_FALLBACK_MODEL_FAMILY", "auto"),
                allow_primary_env=False,
            )
            print(f"Traduction fallback model_family: {self._fallback_model_family}")
        except Exception:
            self._fallback_ct2_translator = None
            self._fallback_ct2_tokenizer = None

    def _resolve_model_family(self, model_dir, tokenizer_dir, tokenizer=None, preferred="auto", allow_primary_env=True):
        explicit = (preferred or "auto").strip().lower()
        if explicit == "auto" and allow_primary_env:
            explicit = (os.getenv("TRANSLATOR_MODEL_FAMILY", "auto") or "auto").strip().lower()
        if explicit and explicit != "auto":
            return explicit
        tok_obj = tokenizer if tokenizer is not None else self._ct2_tokenizer
        tok_name = tok_obj.__class__.__name__.lower() if tok_obj is not None else ""
        d = f"{model_dir} {tokenizer_dir}".lower()
        if "marian" in tok_name or "opus" in d:
            return "marian"
        if "nllb" in tok_name or "nllb" in d:
            return "nllb"
        if "m2m" in tok_name or "m2m100" in d:
            return "m2m100"
        return "m2m100"

    def translate_page(self, structure, target_lang="French"):
        blacklist = ["MANNING", "M A N N I N G", "O REILLY", "PACKT", "PEARSON"]
        tech_dict = {"Deep Learning": "Apprentissage profond", "Vision Systems": "Systèmes de vision"}
        tgt_code = self._normalize_lang_code(target_lang)

        for block in structure.get("blocks", []):
            block_role = block.get("role", "body")
            # Paragraph-level translation for narrative body blocks to preserve
            # sentence continuity across multiple lines.
            if self._should_translate_block_as_paragraph(block):
                self._translate_block_as_paragraph(block, target_lang)
                continue
            block_context = []
            phrases_to_translate = []
            previous_fr_phrases = []
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    src_text = self._normalize_spaces(phrase.get("texte", ""))
                    if src_text:
                        block_context.append(src_text)
                    phrases_to_translate.append(phrase)

            block_ctx_txt = " ".join(block_context)[:600]
            domain = self._detect_domain(block_ctx_txt)
            subdomain = self._detect_subdomain(block_ctx_txt, domain=domain)
            block["detected_domain"] = domain
            block["detected_subdomain"] = subdomain
            for phrase in phrases_to_translate:
                if phrase.get("render_mode") == "background_only":
                    orig_keep = self._normalize_spaces(phrase.get("texte", ""))
                    phrase["translated_text"] = orig_keep
                    phrase["texte"] = orig_keep
                    continue
                orig_phrase_text = self._normalize_spaces(phrase.get("texte", ""))
                if len(orig_phrase_text) < 2:
                    phrase["translated_text"] = orig_phrase_text
                    continue

                wc = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", orig_phrase_text))
                protected = self._is_protected_segment(orig_phrase_text, block_role=block_role)
                # In body text, do not over-protect full phrases: this causes EN leakage in translated PDFs.
                if protected and not (block_role == "body" and wc >= 5):
                    translated_phrase = orig_phrase_text
                elif orig_phrase_text in tech_dict:
                    translated_phrase = tech_dict[orig_phrase_text]
                elif orig_phrase_text.upper() in blacklist:
                    translated_phrase = orig_phrase_text
                else:
                    translated_phrase = self._translate_phrase_resilient(
                        orig_phrase_text,
                        target_lang=target_lang,
                        block_context=block_ctx_txt,
                        block_role=block_role,
                        domain=domain,
                        subdomain=subdomain,
                    )

                translated_phrase = self._normalize_spaces(translated_phrase)
                if tgt_code == "fr":
                    # Enforce standardized heading/label terms on phrase output.
                    for pat, repl in [
                        (r"\bTHE\s+DIRECTION\b", "LA DIRECTION"),
                        (r"\bTHE\s+STEP\s+SIZE\b", "LA TAILLE DU PAS"),
                        (r"\bGOAL\s+WEIGHT\b", "POIDS CIBLE"),
                    ]:
                        translated_phrase = re.sub(pat, repl, translated_phrase, flags=re.IGNORECASE)
                    if self._fr_strict_quality:
                        translated_phrase = self._strict_fr_phrase_pass(
                            translated_phrase,
                            source_text=orig_phrase_text,
                            context_text=block_ctx_txt,
                            previous_translations=previous_fr_phrases,
                        )
                phrase["detected_domain"] = domain
                phrase["detected_subdomain"] = subdomain
                phrase["texte_original"] = orig_phrase_text
                phrase["translated_text"] = translated_phrase
                # Backward compatibility: main field now carries translated phrase for downstream renderers.
                phrase["texte"] = translated_phrase
                if tgt_code == "fr" and translated_phrase:
                    previous_fr_phrases.append(translated_phrase)
                for span in phrase.get("spans", []):
                    span["texte_original"] = span.get("texte", "")
                    # Keep span text untouched to preserve OCR/native source record.
                    self._normalize_span_style(span, role=block_role)

            for line in block.get("lines", []):
                translated_line = self._normalize_spaces(" ".join(
                    (p.get("translated_text") or p.get("texte") or "").strip()
                    for p in line.get("phrases", [])
                ))
                line["translated_text"] = self._dedupe_sentence_runs(translated_line)

            block_translated = self._normalize_spaces(" ".join(
                (ln.get("translated_text") or "").strip()
                for ln in block.get("lines", [])
            ))
            if tgt_code == "fr":
                for pat, repl in [
                    (r"\bTHE\s+DIRECTION\b", "LA DIRECTION"),
                    (r"\bTHE\s+STEP\s+SIZE\b", "LA TAILLE DU PAS"),
                    (r"\bGOAL\s+WEIGHT\b", "POIDS CIBLE"),
                ]:
                    block_translated = re.sub(pat, repl, block_translated, flags=re.IGNORECASE)
            block["translated_text"] = self._dedupe_sentence_runs(block_translated)

        self._post_dedupe_translated_blocks(structure)
        return structure

    def _line_text_for_translation(self, line):
        txt = self._normalize_spaces((line.get("line_text") or "").strip())
        if txt:
            return txt
        parts = []
        for p in line.get("phrases", []):
            t = self._normalize_spaces((p.get("texte") or "").strip())
            if t:
                parts.append(t)
        return self._normalize_spaces(" ".join(parts))

    def _should_translate_block_as_paragraph(self, block):
        role = (block.get("role") or "body").lower()
        if role not in {"body", "figure_caption"}:
            return False
        lines = block.get("lines", []) or []
        if len(lines) < 2:
            return False
        if any((line.get("leading_marker") or "").strip() for line in lines):
            return False
        return True

    def _dehyphenate_line_stream(self, lines):
        out = []
        pending_markers = []
        for s in lines:
            t = self._strip_invisible_chars(self._normalize_spaces(s))
            if not t:
                continue
            if re.fullmatch(r"(?:\d+[.)]?|[•▪◦·\-\*])", t):
                pending_markers.append(t)
                continue
            if out and out[-1].endswith("-") and re.match(r"^[A-Za-zÀ-ÿ]", t):
                out[-1] = self._normalize_spaces(out[-1][:-1] + t)
            else:
                if pending_markers:
                    out.extend(pending_markers)
                    pending_markers = []
                out.append(t)
        if pending_markers:
            out.extend(pending_markers)
        return out

    def _redistribute_translated_to_lines(self, translated_text, source_lines, source_markers):
        # Legacy helper kept for compatibility only; translated paragraph reflow now
        # happens in the reconstructor from block["translated_text"].
        return ["" for _ in source_lines]

    def _translate_block_as_paragraph(self, block, target_lang):
        lines = block.get("lines", []) or []
        source_lines = [self._line_text_for_translation(ln) for ln in lines]
        dehyphenated = self._dehyphenate_line_stream(source_lines)
        src_para = self._strip_invisible_chars(self._normalize_spaces(" ".join(dehyphenated)))
        if not src_para:
            return
        domain = self._detect_domain(src_para[:600])
        subdomain = self._detect_subdomain(src_para[:600], domain=domain)
        translated_para = self._translate_phrase_resilient(
            src_para,
            target_lang=target_lang,
            block_context=src_para[:600],
            block_role=(block.get("role") or "body"),
            domain=domain,
            subdomain=subdomain,
        )
        translated_para = self._strip_invisible_chars(self._normalize_spaces(translated_para))
        translated_para = self._apply_domain_glossary(
            translated_para,
            source_text=src_para,
            target_lang=target_lang,
            domain=domain,
            subdomain=subdomain,
        )
        if self._normalize_lang_code(target_lang) == "fr" and self._fr_strict_quality:
            translated_para = self._strict_fr_phrase_pass(
                translated_para,
                source_text=src_para,
                context_text=src_para[:600],
                previous_translations=[],
            )
        translated_para = self._strip_invisible_chars(self._normalize_spaces(translated_para))
        translated_lines = ["" for _ in source_lines]
        for li, line in enumerate(lines):
            lt = translated_lines[li] if li < len(translated_lines) else ""
            line["translated_text"] = self._normalize_spaces(lt)
            phrases = line.get("phrases", []) or []
            for pi, phrase in enumerate(phrases):
                if phrase.get("render_mode") == "background_only":
                    continue
                phrase["translated_text"] = line["translated_text"] if pi == 0 else ""
                if pi == 0 and line["translated_text"]:
                    phrase["texte"] = line["translated_text"]
            for phrase in phrases:
                phrase["detected_domain"] = domain
                phrase["detected_subdomain"] = subdomain
                for span in phrase.get("spans", []):
                    span["texte_original"] = span.get("texte", "")
                    self._normalize_span_style(span, role=(block.get("role") or "body"))
        block["detected_domain"] = domain
        block["detected_subdomain"] = subdomain
        block["translated_text"] = self._normalize_spaces(translated_para)
        block["translation_compose_mode"] = "paragraph_flow"

    def _translate_text_hierarchical(self, text, target_lang="French", block_context="", block_role="body", domain="general", subdomain=""):
        src = self._normalize_spaces(text)
        if not src:
            return src
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'\-]*", src))

        # Avoid translating short/broken fragments (common in line-based PDF extraction),
        # but keep a focused connector translation for EN->FR.
        if (not self._looks_like_sentence(src)) and word_count <= 4:
            short_fragment = self._translate_short_fragment(src, target_lang=target_lang, block_role=block_role)
            if short_fragment:
                return short_fragment
            if (word_count >= 2) and (not self._is_protected_segment(src, block_role=block_role)):
                t_short = self._translate_snippet(src, target_lang=target_lang, block_context=block_context, level="short_fragment")
                t_short = self._restore_protected_tokens(src, t_short)
                t_short = self._normalize_translation(
                    t_short,
                    target_lang=target_lang,
                    original=src,
                    context_text=block_context,
                )
                if self._is_acceptable_translation(src, t_short):
                    return t_short
            return src

        # Exact terminology before model call.
        pre_exact = self._exact_glossary_match(
            src,
            target_lang=target_lang,
            domain=domain,
            subdomain=subdomain,
        )
        if pre_exact:
            return pre_exact

        # Strict terminology mode: keep glossary terms fixed inside larger sentences.
        use_forced_terms = self._strict_glossary and (self._force_terms_in_sentences or not self._looks_like_sentence(src))
        if use_forced_terms:
            forced = self._translate_with_forced_glossary_terms(
                src,
                target_lang=target_lang,
                block_context=block_context,
                domain=domain,
                subdomain=subdomain,
            )
            if forced and forced != src:
                forced = self._restore_protected_tokens(src, forced)
                forced = self._normalize_translation(
                    forced,
                    target_lang=target_lang,
                    original=src,
                    context_text=block_context,
                )
                forced = self._apply_domain_glossary(
                    forced,
                    source_text=src,
                    target_lang=target_lang,
                    domain=domain,
                    subdomain=subdomain,
                )
                if self._is_acceptable_translation(src, forced):
                    return forced

        # Level 1: full sentence/phrase translation.
        if self._looks_like_sentence(src):
            t1 = self._translate_snippet(src, target_lang=target_lang, block_context=block_context, level="sentence")
            t1 = self._restore_protected_tokens(src, t1)
            t1 = self._normalize_translation(
                t1,
                target_lang=target_lang,
                original=src,
                context_text=block_context,
            )
            t1 = self._apply_domain_glossary(
                t1,
                source_text=src,
                target_lang=target_lang,
                domain=domain,
                subdomain=subdomain,
            )
            if self._is_acceptable_translation(src, t1):
                return t1

        # Level 2: expression-based translation.
        expr_parts = self._split_expressions(src)
        if len(expr_parts) > 1:
            out = []
            for part in expr_parts:
                p = self._normalize_spaces(part)
                if not p:
                    out.append(part)
                    continue
                if self._is_protected_segment(p, block_role=block_role):
                    out.append(part)
                    continue
                if self._is_separator_token(part):
                    out.append(part)
                    continue
                tr = self._translate_snippet(p, target_lang=target_lang, block_context=block_context, level="expression")
                tr = self._restore_protected_tokens(p, tr)
                tr = self._apply_domain_glossary(
                    tr,
                    source_text=p,
                    target_lang=target_lang,
                    domain=domain,
                    subdomain=subdomain,
                )
                if not self._is_acceptable_translation(p, tr):
                    tr = p
                out.append(self._reinject_spacing(part, tr))
            expr_text = "".join(out)
            expr_text = self._normalize_translation(
                expr_text,
                target_lang=target_lang,
                original=src,
                context_text=block_context,
            )
            if self._is_acceptable_translation(src, expr_text):
                return expr_text

        # Level 3: word-level translation fallback.
        # Disabled for quality: word-by-word MT introduces severe semantic drift.
        return src

        # (kept for reference)
        word_parts = self._split_words_with_separators(src)
        out_words = []
        for part in word_parts:
            p = self._normalize_spaces(part)
            if not p:
                out_words.append(part)
                continue
            if self._is_separator_token(part):
                out_words.append(part)
                continue
            if self._is_protected_segment(p, block_role=block_role):
                out_words.append(part)
                continue
            # Tiny words/letters: keep unchanged to avoid noise.
            if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", p)) <= 2:
                out_words.append(part)
                continue
            tr = self._translate_snippet(p, target_lang=target_lang, block_context=block_context, level="word")
            tr = self._restore_protected_tokens(p, tr)
            tr = self._apply_domain_glossary(
                tr,
                source_text=p,
                target_lang=target_lang,
                domain=domain,
                subdomain=subdomain,
            )
            if not self._is_acceptable_translation(p, tr):
                tr = p
            out_words.append(self._reinject_spacing(part, tr))
        final = "".join(out_words)
        final = self._normalize_translation(final, target_lang=target_lang, original=src)
        if self._is_acceptable_translation(src, final):
            return final

        # Level 4: symbols/other => keep original.
        return src

    def _translate_short_fragment(self, text, target_lang="French", block_role="body"):
        s = self._normalize_spaces(text)
        if not s:
            return None
        src_lang = self._guess_source_lang(s)
        tgt_lang = self._normalize_lang_code(target_lang)
        if src_lang != "en" or tgt_lang != "fr":
            return None
        words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'\-]*", s)
        if not words or len(words) > 4:
            return None
        connector_map = {
            "for": "pour",
            "and": "et",
            "or": "ou",
            "of": "de",
            "to": "a",
            "in": "dans",
            "on": "sur",
            "with": "avec",
            "from": "de",
            "by": "par",
            "at": "a",
            "the": "le",
            "a": "un",
            "an": "un",
        }
        # Translate only pure connector fragments to avoid degrading technical terms.
        low_words = [w.lower() for w in words]
        if any(w not in connector_map for w in low_words):
            if self._is_protected_segment(s, block_role=block_role):
                return None
            return None
        tr = " ".join(connector_map[w] for w in low_words)
        if not tr:
            return None
        return self._reinject_spacing(s, tr)

    def _get_domain_priority_chain(self, domain="general", subdomain=""):
        d = self._normalize_spaces(domain).lower() or "general"
        sd = self._normalize_spaces(subdomain).lower()
        chain = []
        if d != "general" and sd:
            chain.append(f"{d}.{sd}")
            chain.append(f"{d}/{sd}")
            chain.append(sd)
        if d:
            chain.append(d)
        if self._use_general_glossary:
            chain.append("general")
        out = []
        seen = set()
        for x in chain:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _is_safe_glossary_key(self, key):
        k = self._normalize_spaces(key).lower()
        if not k:
            return False
        # Avoid toxic tiny/function tokens from very large generic lexicons.
        if len(k) < 4:
            return False
        if re.fullmatch(r"[a-z]", k):
            return False
        stop = {
            "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "by", "with",
            "is", "are", "be", "was", "were", "this", "that", "these", "those", "it", "as",
            "from", "into", "over", "under", "between", "about", "after", "before", "through",
            "very", "more", "most", "less", "least",
        }
        if k in stop:
            return False
        # Keep alpha/num technical tokens; reject mostly-symbolic noise.
        if len(re.findall(r"[a-z0-9]", k)) < max(3, int(0.6 * len(k))):
            return False
        return True

    def _get_domain_pair_map(self, domain, source_lang, target_lang, subdomain=""):
        pair_key = f"{source_lang}_{target_lang}"
        out = {}
        for dom in self._get_domain_priority_chain(domain=domain, subdomain=subdomain):
            d = self._domain_glossaries.get(dom, {})
            pairs = d.get("pairs", {})
            g = pairs.get(pair_key, {})
            if isinstance(g, dict):
                # Filter unsafe keys aggressively to avoid glossary poisoning.
                for k, v in g.items():
                    if self._is_safe_glossary_key(k):
                        out[k] = v
        return out

    def _translate_with_forced_glossary_terms(self, text, target_lang="French", block_context="", domain="general", subdomain=""):
        s = self._normalize_spaces(text)
        if not s:
            return s
        src_lang = self._guess_source_lang(s)
        tgt_lang = self._normalize_lang_code(target_lang)
        pair_map = self._get_domain_pair_map(domain, src_lang, tgt_lang, subdomain=subdomain)
        if not pair_map:
            return s

        # Match longest source terms first and split text into [non-term][term] chunks.
        terms = sorted({k for k in pair_map.keys() if k}, key=len, reverse=True)
        # Limit regex size for stability/performance.
        terms = terms[:2000]
        if not terms:
            return s
        pat = r"(?i)\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        rx = re.compile(pat)
        matches = list(rx.finditer(s))
        if not matches:
            return s

        out = []
        idx = 0
        changed = False
        for m in matches:
            a, b = m.span()
            if a > idx:
                chunk = s[idx:a]
                c = self._normalize_spaces(chunk)
                if c:
                    tr = self._translate_snippet(c, target_lang=target_lang, block_context=block_context, level="forced_chunk")
                    tr = self._sanitize_translation(tr, c)
                    out.append(self._reinject_spacing(chunk, tr))
                else:
                    out.append(chunk)
            src_term = self._normalize_spaces(m.group(0)).lower()
            tgt_term = pair_map.get(src_term)
            if tgt_term:
                out.append(tgt_term)
                changed = True
            else:
                out.append(m.group(0))
            idx = b
        if idx < len(s):
            tail = s[idx:]
            c = self._normalize_spaces(tail)
            if c:
                tr = self._translate_snippet(c, target_lang=target_lang, block_context=block_context, level="forced_chunk")
                tr = self._sanitize_translation(tr, c)
                out.append(self._reinject_spacing(tail, tr))
            else:
                out.append(tail)

        merged = self._normalize_spaces("".join(out))
        return merged if changed else s

    def _build_domain_glossaries(self):
        # Canonical normalized terms (source->target), plus output normalization variants.
        # Internal shape:
        # {
        #   "science": {
        #     "pairs": {"en_fr": {...}, "en_es": {...}},
        #     "normalize": {"fr": {...}, "es": {...}}
        #   }, ...
        # }
        return {
            "science": {
                "pairs": {
                    "en_fr": {
                        "gradient descent": "descente de gradient",
                        "learning rate": "taux d'apprentissage",
                        "neural network": "réseau de neurones",
                        "error": "erreur",
                        "hyperparameter": "hyperparamètre",
                        "optimization": "optimisation",
                        "oscillating": "oscillant",
                        "feedforward": "propagation avant",
                        "cnn architecture": "architecture des CNN",
                        "input layer": "couche d'entrée",
                        "convolutional layers": "couches convolutionnelles",
                        "feature extraction": "extraction de caractéristiques",
                        "fully connected layers": "couches entièrement connectées",
                        "classification": "classification",
                        "output prediction": "prédiction en sortie",
                        "feature maps": "cartes de caractéristiques",
                        "flattened": "aplati",
                        "layer depth": "profondeur de la couche",
                        "convolutional layer": "couche convolutionnelle",
                        "fully connected layer": "couche entièrement connectée",
                    },
                    "en_es": {
                        "input layer": "capa de entrada",
                        "convolutional layers": "capas convolucionales",
                        "feature extraction": "extracción de características",
                        "fully connected layers": "capas completamente conectadas",
                        "classification": "clasificación",
                        "output prediction": "predicción de salida",
                        "feature maps": "mapas de características",
                        "flattened": "aplanado",
                    },
                    "en_pt": {
                        "input layer": "camada de entrada",
                        "convolutional layers": "camadas convolucionais",
                        "feature extraction": "extração de características",
                        "fully connected layers": "camadas totalmente conectadas",
                        "classification": "classificação",
                        "output prediction": "predição de saída",
                        "feature maps": "mapas de características",
                        "flattened": "achatado",
                    },
                    "en_de": {
                        "input layer": "Eingabeschicht",
                        "convolutional layers": "Faltungsschichten",
                        "feature extraction": "Merkmalsextraktion",
                        "fully connected layers": "vollständig verbundene Schichten",
                        "classification": "Klassifikation",
                        "output prediction": "Ausgabevorhersage",
                        "feature maps": "Merkmalskarten",
                        "flattened": "abgeflacht",
                    },
                    "en_ar": {
                        "input layer": "طبقة الإدخال",
                        "convolutional layers": "الطبقات الالتفافية",
                        "feature extraction": "استخراج السمات",
                        "fully connected layers": "الطبقات المتصلة بالكامل",
                        "classification": "التصنيف",
                        "output prediction": "تنبؤ الخرج",
                        "feature maps": "خرائط السمات",
                        "flattened": "تسطيح",
                    },
                    "en_zh": {
                        "input layer": "输入层",
                        "convolutional layers": "卷积层",
                        "feature extraction": "特征提取",
                        "fully connected layers": "全连接层",
                        "classification": "分类",
                        "output prediction": "输出预测",
                        "feature maps": "特征图",
                        "flattened": "展平",
                    },
                    "en_ja": {
                        "input layer": "入力層",
                        "convolutional layers": "畳み込み層",
                        "feature extraction": "特徴抽出",
                        "fully connected layers": "全結合層",
                        "classification": "分類",
                        "output prediction": "出力予測",
                        "feature maps": "特徴マップ",
                        "flattened": "平坦化",
                    },
                },
                "normalize": {
                    "fr": {
                        "descent gradient": "descente de gradient",
                        "gradient descente": "descente de gradient",
                        "taux d’apprentissage": "taux d'apprentissage",
                        "réseau nerveux": "réseau de neurones",
                        "couches de convection": "couches convolutionnelles",
                        "couches de convolutions": "couches convolutionnelles",
                        "couches connectées localement": "couches convolutionnelles",
                        "cartes de fonctionnalités": "cartes de caractéristiques",
                        "caractéristiques cartes": "cartes de caractéristiques",
                        "fonctionnalités": "caractéristiques",
                        "prévision des produits": "prédiction en sortie",
                        "calque": "couche",
                        "aplatie vers un vecteur": "aplatie en un vecteur",
                        "featureextraction": "extraction de caractéristiques",
                    },
                },
            },
            "economy": {
                "pairs": {
                    "en_fr": {
                        "interest rate": "taux d'intérêt",
                        "gross domestic product": "produit intérieur brut",
                        "inflation": "inflation",
                    },
                },
                "normalize": {},
            },
            "politics": {
                "pairs": {
                    "en_fr": {
                        "foreign policy": "politique étrangère",
                        "rule of law": "état de droit",
                        "public policy": "politique publique",
                    },
                },
                "normalize": {},
            },
            "biology": {
                "pairs": {
                    "en_fr": {
                        "cell membrane": "membrane cellulaire",
                        "gene expression": "expression génique",
                        "immune response": "réponse immunitaire",
                    },
                },
                "normalize": {},
            },
            "general": {"pairs": {}, "normalize": {}},
        }

    def _merge_glossary_payload(self, payload):
        if not isinstance(payload, dict):
            return
        domain = str(payload.get("domain", "general")).strip().lower() or "general"
        source_lang = self._normalize_lang_code(payload.get("source_lang", "en"))
        target_lang = self._normalize_lang_code(payload.get("target_lang", "fr"))
        pair_key = f"{source_lang}_{target_lang}"

        if domain not in self._domain_glossaries:
            self._domain_glossaries[domain] = {"pairs": {}, "normalize": {}}
        if "pairs" not in self._domain_glossaries[domain]:
            self._domain_glossaries[domain]["pairs"] = {}
        if "normalize" not in self._domain_glossaries[domain]:
            self._domain_glossaries[domain]["normalize"] = {}

        pair_map = self._domain_glossaries[domain]["pairs"].setdefault(pair_key, {})
        norm_map = self._domain_glossaries[domain]["normalize"].setdefault(target_lang, {})

        entries = payload.get("entries", {})
        if isinstance(entries, dict):
            for k, v in entries.items():
                ks = self._normalize_spaces(str(k)).lower()
                vs = self._normalize_spaces(str(v))
                if ks and vs:
                    pair_map[ks] = vs
        elif isinstance(entries, list):
            for row in entries:
                if not isinstance(row, dict):
                    continue
                ks = self._normalize_spaces(str(row.get("source", ""))).lower()
                vs = self._normalize_spaces(str(row.get("target", "")))
                if ks and vs:
                    pair_map[ks] = vs

        normalize_map = payload.get("normalize", {})
        if isinstance(normalize_map, dict):
            for k, v in normalize_map.items():
                ks = self._normalize_spaces(str(k)).lower()
                vs = self._normalize_spaces(str(v))
                if ks and vs:
                    norm_map[ks] = vs

    def _load_external_glossaries(self):
        base_dir = os.getenv("TRANSLATOR_GLOSSARY_DIR", "ai_models/translation/glossaries")
        if not os.path.isdir(base_dir):
            return
        loaded = 0
        for root, _, files in os.walk(base_dir):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    self._merge_glossary_payload(payload)
                    loaded += 1
                except Exception:
                    continue
        if loaded:
            print(f"Glossaires externes chargés: {loaded}")

    def _detect_domain(self, context_text):
        s = (context_text or "").lower()
        lex = {
            "science": [
                "equation", "theorem", "integral", "derivative", "matrix", "vector", "physics",
                "molecule", "chemical", "astronomy", "galaxy", "orbit", "telescope",
                "neural", "learning rate", "optimization", "algorithm",
            ],
            "economy": [
                "inflation", "gdp", "interest rate", "fiscal", "monetary", "economy",
                "market", "bond", "equity", "exchange rate", "trade balance",
            ],
            "politics": [
                "election", "parliament", "government", "policy", "constitution",
                "diplomacy", "senate", "legislative", "executive", "public administration",
            ],
            "biology": [
                "cell", "protein", "gene", "dna", "rna", "enzyme", "organism", "immune",
                "genome", "microbiology", "ecology",
            ],
            "medicine": [
                "patient", "diagnosis", "therapy", "clinical", "pharmacology", "epidemiology",
                "oncology", "cardiology", "neurology", "hospital", "symptom",
            ],
            "engineering": [
                "mechanical", "electrical", "civil engineering", "control system", "signal processing",
                "manufacturing", "structural", "embedded", "robotics", "cad",
            ],
            "legal": [
                "court", "statute", "regulation", "contract", "criminal law", "civil law",
                "jurisdiction", "compliance", "litigation", "tax law",
            ],
            "technology": [
                "software", "hardware", "database", "cloud", "cybersecurity", "api", "protocol",
                "distributed system", "operating system", "container",
            ],
            "education": [
                "curriculum", "pedagogy", "assessment", "learning outcomes", "classroom",
                "didactics", "instructional design", "student performance",
            ],
            "history": [
                "historical period", "chronology", "empire", "archival", "historiography",
                "medieval", "antiquity", "industrial revolution",
            ],
            "geography": [
                "latitude", "longitude", "topography", "cartography", "climate", "river basin",
                "geology", "landform", "ecosystem", "geospatial",
            ],
        }
        scores = {k: 0 for k in lex.keys()}
        for d, kws in lex.items():
            for kw in kws:
                if kw in s:
                    scores[d] += 1
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "general"

    def _detect_subdomain(self, context_text, domain="general"):
        s = (context_text or "").lower()
        d = (domain or "").lower()
        lex = {
            "science": {
                "mathematics": [
                    "equation", "theorem", "lemma", "integral", "derivative", "matrix", "vector",
                    "probability", "statistics", "algebra", "calculus", "topology",
                ],
                "physics": [
                    "force", "energy", "velocity", "acceleration", "quantum", "relativity",
                    "mass", "momentum", "thermodynamics", "electromagnetic", "wave", "particle",
                ],
                "chemistry": [
                    "molecule", "molar", "stoichiometry", "reaction", "compound", "acid", "base",
                    "catalyst", "polymer", "organic chemistry", "inorganic", "ph", "atom",
                ],
                "astronomy": [
                    "galaxy", "planet", "star", "orbit", "cosmology", "telescope", "nebula",
                    "astrophysics", "solar system", "exoplanet", "supernova",
                ],
                "computer_science": [
                    "algorithm", "neural", "learning rate", "gradient descent", "dataset", "model",
                    "training", "inference", "backpropagation", "optimization", "network", "cpu",
                    "memory", "complexity", "compiler",
                ],
            },
            "economy": {
                "macroeconomics": ["inflation", "gdp", "fiscal policy", "monetary policy", "unemployment", "cpi"],
                "finance": ["equity", "bond", "portfolio", "derivative", "volatility", "asset pricing"],
                "banking": ["interest rate", "credit risk", "liquidity", "deposit", "loan", "capital adequacy"],
                "trade": ["export", "import", "tariff", "trade balance", "customs", "exchange rate"],
            },
            "politics": {
                "governance": ["governance", "public administration", "institutional", "accountability", "transparency"],
                "public_policy": ["public policy", "policy design", "implementation", "regulatory impact"],
                "diplomacy": ["foreign policy", "diplomacy", "treaty", "bilateral", "multilateral"],
                "elections": ["election", "electoral", "ballot", "voter", "campaign"],
                "law": ["constitutional", "legislative", "judiciary", "rule of law", "jurisdiction"],
            },
            "biology": {
                "genetics": ["gene", "genome", "mutation", "inheritance", "genetic expression"],
                "microbiology": ["bacteria", "virus", "microorganism", "culture medium", "pathogen"],
                "immunology": ["immune", "antibody", "antigen", "innate immunity", "adaptive immunity"],
                "ecology": ["ecosystem", "biodiversity", "habitat", "population dynamics", "food web"],
                "physiology": ["metabolism", "homeostasis", "organ system", "cell membrane", "enzyme"],
            },
            "medicine": {
                "cardiology": ["cardiac", "heart failure", "arrhythmia", "hypertension", "ecg"],
                "oncology": ["tumor", "cancer", "metastasis", "chemotherapy", "radiotherapy"],
                "neurology": ["neuron", "brain", "stroke", "epilepsy", "neurodegenerative"],
                "pharmacology": ["drug", "dosage", "pharmacokinetics", "adverse effect", "contraindication"],
                "epidemiology": ["incidence", "prevalence", "cohort", "outbreak", "public health"],
            },
            "engineering": {
                "mechanical": ["thermofluid", "mechanics", "kinematics", "dynamics", "stress", "strain"],
                "electrical": ["circuit", "voltage", "current", "resistance", "signal", "control"],
                "civil": ["structural", "geotechnical", "concrete", "beam", "foundation", "load"],
                "materials": ["alloy", "composite", "microstructure", "fatigue", "fracture"],
                "control_systems": ["feedback", "stability", "controller", "pid", "state space"],
            },
            "legal": {
                "civil_law": ["civil law", "tort", "liability", "damages", "obligation"],
                "criminal_law": ["criminal law", "offense", "prosecution", "penalty", "felony"],
                "international_law": ["international law", "treaty", "sovereignty", "jurisdiction", "convention"],
                "labor_law": ["employment", "collective bargaining", "labor code", "workplace", "union"],
                "tax_law": ["taxable income", "deduction", "vat", "withholding", "tax compliance"],
            },
            "technology": {
                "software": ["software architecture", "refactoring", "testing", "deployment", "dependency"],
                "data": ["database", "etl", "data warehouse", "query optimization", "schema"],
                "cloud": ["cloud", "container", "kubernetes", "autoscaling", "infrastructure as code"],
                "cybersecurity": ["encryption", "vulnerability", "threat model", "authentication", "authorization"],
            },
            "education": {
                "pedagogy": ["pedagogy", "didactics", "teaching strategy", "active learning"],
                "assessment": ["assessment", "rubric", "summative", "formative", "evaluation"],
                "curriculum": ["curriculum", "syllabus", "learning outcomes", "competency framework"],
            },
            "history": {
                "ancient_history": ["antiquity", "ancient empire", "classical period", "archaeological"],
                "medieval_history": ["medieval", "feudal", "kingdom", "chronicle", "manorial"],
                "modern_history": ["industrial revolution", "colonial", "nation-state", "modern era"],
            },
            "geography": {
                "physical_geography": ["landform", "hydrology", "geomorphology", "climate", "tectonic"],
                "human_geography": ["urbanization", "migration", "demography", "settlement", "economic geography"],
                "geospatial": ["gis", "cartography", "geospatial", "remote sensing", "geodesy"],
            },
        }
        if d not in lex:
            return ""
        scores = {k: 0 for k in lex[d].keys()}
        for sd, kws in lex[d].items():
            for kw in kws:
                if kw in s:
                    scores[sd] += 1
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else ""

    def _exact_glossary_match(self, text, target_lang="French", domain="general", subdomain=""):
        s = self._normalize_spaces(text)
        if not s:
            return None
        low = s.lower()
        tgt = self._normalize_lang_code(target_lang)
        src = self._guess_source_lang(s)
        pair_key = f"{src}_{tgt}"
        for dom in self._get_domain_priority_chain(domain=domain, subdomain=subdomain):
            d = self._domain_glossaries.get(dom, {})
            pairs = d.get("pairs", {})
            g = pairs.get(pair_key, {})
            if low in g:
                return g[low]
        return None

    def _apply_domain_glossary(self, translated, source_text="", target_lang="French", domain="general", subdomain=""):
        if not translated:
            return translated
        # Do not touch hard-protected source segments.
        if self._is_protected_segment(source_text):
            return translated
        sentence_like = self._looks_like_sentence(source_text)
        out = translated
        tgt_lang = self._normalize_lang_code(target_lang)
        src_lang = self._guess_source_lang(source_text)
        pair_key = f"{src_lang}_{tgt_lang}"
        for dom in self._get_domain_priority_chain(domain=domain, subdomain=subdomain):
            g = self._domain_glossaries.get(dom, {})
            pairs = g.get("pairs", {})
            norms = g.get("normalize", {})
            pair_map = pairs.get(pair_key, {})
            norm_map = norms.get(tgt_lang, {})
            # Replace known source technical chunks still present after translation.
            for src, tgt in sorted(pair_map.items(), key=lambda kv: len(kv[0]), reverse=True):
                if sentence_like and (" " not in src):
                    # In full sentences, single-word forced replacements degrade fluency.
                    continue
                out = re.sub(rf"(?i)\\b{re.escape(src)}\\b", tgt, out)
            # Normalize common bad variants in target output.
            for bad, good in sorted(norm_map.items(), key=lambda kv: len(kv[0]), reverse=True):
                if sentence_like and (" " not in bad):
                    continue
                out = re.sub(rf"(?i)\\b{re.escape(bad)}\\b", good, out)
        return out

    def _source_leak_score(self, text, target_lang, source_lang):
        s = self._normalize_spaces(text)
        if not s:
            return 0.0
        tgt = self._normalize_lang_code(target_lang)
        src = self._normalize_lang_code(source_lang)
        if not src or src == tgt:
            return 0.0
        src_hits = float(self._language_marker_counts(s, src))
        tgt_hits = float(self._language_marker_counts(s, tgt))
        return (src_hits + 1.0) / (tgt_hits + 1.0)

    def _translation_gate_ok(self, text, target_lang, source_lang="en"):
        s = self._normalize_spaces(text)
        if not s:
            return False
        if self._contains_invisible_chars(s):
            return False
        if not self._strict_gate:
            return True
        tgt = self._normalize_lang_code(target_lang)
        if tgt == "fr":
            low = s.lower()
            if re.search(r"\b(the|we|you|suppose|look at|remember|passes through|high-level|architecture)\b", low):
                return False
            en_words = len(re.findall(r"\b(the|and|with|for|from|this|that|are|you|your|will|layers|feature|network|looks|suppose|building|classify|passes|through|detect|patterns|extract)\b", s, flags=re.IGNORECASE))
            if en_words > 1:
                return False
        leak = self._source_leak_score(s, target_lang=tgt, source_lang=source_lang)
        return leak <= 1.15

    def _translate_snippet(self, snippet, target_lang="French", block_context="", level="sentence"):
        s = self._normalize_spaces(snippet)
        if not s:
            return s
        key = ("v2", target_lang.lower(), level, s, block_context[:180])
        if key in self._cache:
            return self._cache[key]
        raw = self._ct2_translate(s, target_lang=target_lang)
        cleaned = self._sanitize_translation(raw, s)
        self._cache[key] = cleaned
        return cleaned

    def _normalize_lang_code(self, lang):
        # map user-facing names/codes to m2m100 language codes.
        l = (lang or "French").strip().lower()
        mapping = {
            "french": "fr",
            "fr": "fr",
            "english": "en",
            "en": "en",
            "spanish": "es",
            "es": "es",
            "german": "de",
            "de": "de",
            "italian": "it",
            "it": "it",
            "portuguese": "pt",
            "pt": "pt",
            "russian": "ru",
            "ru": "ru",
            "arabic": "ar",
            "ar": "ar",
            "chinese": "zh",
            "zh": "zh",
            "zh-cn": "zh",
            "japanese": "ja",
            "ja": "ja",
            "korean": "ko",
            "ko": "ko",
            "hindi": "hi",
            "hi": "hi",
            "vietnamese": "vi",
            "vi": "vi",
            "thai": "th",
            "th": "th",
            "indonesian": "id",
            "id": "id",
            "turkish": "tr",
            "tr": "tr",
            "dutch": "nl",
            "nl": "nl",
            "polish": "pl",
            "pl": "pl",
            "ukrainian": "uk",
            "uk": "uk",
        }
        return mapping.get(l, l if re.fullmatch(r"[a-z]{2,3}", l) else "fr")

    def _to_nllb_lang_code(self, lang):
        code = self._normalize_lang_code(lang)
        mapping = {
            "en": "eng_Latn",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "ar": "arb_Arab",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "hi": "hin_Deva",
            "vi": "vie_Latn",
            "th": "tha_Thai",
            "id": "ind_Latn",
            "tr": "tur_Latn",
            "nl": "nld_Latn",
            "pl": "pol_Latn",
            "uk": "ukr_Cyrl",
        }
        return mapping.get(code)

    def _is_known_token(self, tokenizer, token):
        if not token:
            return False
        try:
            tok_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tok_id, (list, tuple)):
                tok_id = tok_id[0] if tok_id else None
            unk_id = getattr(tokenizer, "unk_token_id", None)
            return tok_id is not None and tok_id != -1 and tok_id != unk_id
        except Exception:
            return False

    def _nllb_target_prefix(self, tokenizer, tgt_nllb):
        if not tgt_nllb:
            return None
        candidates = []
        lang_to_token = getattr(tokenizer, "lang_code_to_token", None)
        if isinstance(lang_to_token, dict):
            tok = lang_to_token.get(tgt_nllb)
            if isinstance(tok, str) and tok:
                candidates.append(tok)
        candidates.extend([f"__{tgt_nllb}__", tgt_nllb])
        for tok in candidates:
            if self._is_known_token(tokenizer, tok):
                return [[tok]]
        return [[candidates[-1]]]

    def _language_markers(self, lang_code):
        lc = (lang_code or "").lower()
        markers = {
            "en": {"the", "and", "of", "for", "with", "from", "to", "in", "is", "are", "this", "that", "will", "when"},
            "fr": {"le", "la", "les", "de", "des", "du", "et", "est", "sont", "pour", "avec", "dans", "ce", "cette", "qui", "que"},
            "es": {"el", "la", "los", "las", "de", "del", "y", "en", "para", "con", "que", "por", "es", "son"},
            "de": {"der", "die", "das", "und", "mit", "für", "von", "zu", "in", "ist", "sind", "den", "dem"},
            "it": {"il", "lo", "la", "gli", "le", "di", "e", "con", "per", "in", "che", "è", "sono"},
            "pt": {"o", "a", "os", "as", "de", "do", "da", "e", "com", "para", "em", "que", "é", "são"},
            "ru": {"и", "в", "на", "с", "для", "что", "это", "к", "из", "по"},
            "ar": {"و", "في", "من", "على", "مع", "إلى", "أن", "هذا", "هذه"},
        }
        return markers.get(lc, set())

    def _language_marker_counts(self, text, lang_code):
        s = self._normalize_spaces(text)
        m = self._language_markers(lang_code)
        if not s or not m:
            return 0
        if lang_code in {"zh", "ja", "ko"}:
            # CJK scripts do not rely on whitespace function words in the same way.
            return 1 if s else 0
        c = 0
        for tok in m:
            c += len(re.findall(rf"\b{re.escape(tok)}\b", s, flags=re.IGNORECASE))
        return c

    def _guess_source_lang(self, text):
        s = text or ""
        # Basic script heuristic for multilingual support.
        if re.search(r"[\u4e00-\u9fff]", s):
            return "zh"
        if re.search(r"[\u3040-\u30ff]", s):
            return "ja"
        if re.search(r"[\uac00-\ud7af]", s):
            return "ko"
        if re.search(r"[\u0600-\u06ff]", s):
            return "ar"
        if re.search(r"[\u0400-\u04FF]", s):
            return "ru"
        # default latin script: English-like source unless configured otherwise.
        return self._normalize_lang_code(os.getenv("TRANSLATOR_DEFAULT_SOURCE_LANG", "en"))

    def _ct2_translate_with_backend(self, translator, tokenizer, model_family, text, target_lang="French"):
        if not translator or not tokenizer:
            return None
        tgt = self._normalize_lang_code(target_lang)
        src = self._guess_source_lang(text)
        try:
            if model_family == "marian":
                # Marian setup here is EN->FR only.
                if src != "en" or tgt != "fr":
                    return None
                encoded = tokenizer(text, return_attention_mask=False)
                input_ids = encoded.get("input_ids", [])
                if not input_ids:
                    return None
                source_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                results = translator.translate_batch(
                    [source_tokens],
                    max_batch_size=1,
                    beam_size=int(os.getenv("CT2_BEAM_SIZE", "4")),
                    repetition_penalty=1.05,
                )
                if not results:
                    return None
                out_tokens = results[0].hypotheses[0]
                out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
                if not out_ids:
                    return None
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                return self._normalize_spaces(out_text)

            if model_family == "nllb":
                src_nllb = self._to_nllb_lang_code(src)
                tgt_nllb = self._to_nllb_lang_code(tgt)
                if not src_nllb or not tgt_nllb:
                    return None
                try:
                    tokenizer.src_lang = src_nllb
                except Exception:
                    pass
                encoded = tokenizer(text, return_attention_mask=False)
                input_ids = encoded.get("input_ids", [])
                if not input_ids:
                    return None
                source_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                target_prefix = self._nllb_target_prefix(tokenizer, tgt_nllb)
                results = translator.translate_batch(
                    [source_tokens],
                    target_prefix=target_prefix,
                    max_batch_size=1,
                    beam_size=int(os.getenv("CT2_BEAM_SIZE", "4")),
                    repetition_penalty=1.05,
                )
                if not results:
                    return None
                out_tokens = results[0].hypotheses[0]
                out_tokens = [
                    t for t in out_tokens
                    if not re.fullmatch(r"__.+__", t)
                    and not re.fullmatch(r"[a-z]{3}_[A-Za-z]{4}", t)
                ]
                out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
                if not out_ids:
                    return None
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                return self._normalize_spaces(out_text)

            # Default path: M2M100-like multilingual models.
            try:
                tokenizer.src_lang = src
            except Exception:
                pass
            encoded = tokenizer(text, return_attention_mask=False)
            input_ids = encoded.get("input_ids", [])
            if not input_ids:
                return None
            source_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            target_prefix = [[f"__{tgt}__"]]
            results = translator.translate_batch(
                [source_tokens],
                target_prefix=target_prefix,
                max_batch_size=1,
                beam_size=int(os.getenv("CT2_BEAM_SIZE", "4")),
                repetition_penalty=1.05,
            )
            if not results:
                return None
            out_tokens = results[0].hypotheses[0]
            # Remove language tag token if present.
            out_tokens = [t for t in out_tokens if not re.fullmatch(r"__.+__", t)]
            out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            if not out_ids:
                return None
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            return self._normalize_spaces(out_text)
        except Exception:
            return None

    def _ct2_translate(self, text, target_lang="French"):
        if not self._ct2_translator or not self._ct2_tokenizer:
            return text
        primary = self._ct2_translate_with_backend(
            self._ct2_translator,
            self._ct2_tokenizer,
            self._model_family,
            text,
            target_lang=target_lang,
        )
        if primary is not None:
            return primary
        fallback = self._ct2_translate_with_backend(
            self._fallback_ct2_translator,
            self._fallback_ct2_tokenizer,
            self._fallback_model_family,
            text,
            target_lang=target_lang,
        )
        return fallback if fallback is not None else text

    def _looks_like_sentence(self, text):
        s = self._normalize_spaces(text)
        words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'\-]*", s)
        alpha_words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*", s)
        if len(words) < 4:
            return False
        if len(alpha_words) < 3:
            return False
        if re.search(r"[\.!\?:;]$", s):
            return True
        if len(words) >= 5:
            return True
        return False

    def _is_separator_token(self, token):
        return bool(re.fullmatch(r"[\s,\.;:\(\)\[\]\{\}\-–—/]+", token or ""))

    def _split_expressions(self, text):
        # Keep separators to preserve original shape/spacing.
        return re.split(r"(\s*[,:;]\s*|\s+\-\s+|\s+–\s+|\s+—\s+|\s*\(\s*|\s*\)\s*)", text)

    def _split_words_with_separators(self, text):
        return re.split(r"(\s+|[,\.;:\(\)\[\]\{\}\-–—/])", text)

    def _sanitize_source_for_translation(self, text):
        s = self._normalize_spaces(text)
        if not s:
            return s
        # Common OCR artifacts that hurt MT quality.
        s = s.replace("·", "'").replace("’", "'")
        s = re.sub(r"([A-Za-zÀ-ÿ])-\s+([A-Za-zÀ-ÿ])", r"\1\2", s)
        s = re.sub(r"\s+\.\.\.\s*", " ... ", s)
        return self._normalize_spaces(s)

    def _strip_leading_bullets(self, text):
        s = self._normalize_spaces(text)
        m = re.match(r"^\s*([•▪◦·\-\*]+)\s*", s)
        if not m:
            return s, ""
        bullet = m.group(1).strip()
        rest = s[m.end():]
        return self._normalize_spaces(rest), bullet

    def _translation_leak_score(self, text, target_lang):
        s = self._normalize_spaces(text)
        if not s:
            return 1e9
        tgt = self._normalize_lang_code(target_lang)
        if tgt in {"", "en"}:
            return 0.0
        en = float(self._language_marker_counts(s, "en"))
        tg = float(self._language_marker_counts(s, tgt))
        words = max(1.0, len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", s)))
        return (en * 1.4 - tg * 0.9) / words

    def _split_for_direct_translation(self, text, max_chars=260):
        s = self._normalize_spaces(text)
        if not s:
            return []
        parts = re.split(r"(?<=[\.\!\?\:\;])\s+", s)
        out = []
        for p in parts:
            p = self._normalize_spaces(p)
            if not p:
                continue
            if len(p) <= max_chars:
                out.append(p)
                continue
            subs = re.split(r"(?<=,)\s+", p)
            cur = ""
            for t in subs:
                t = self._normalize_spaces(t)
                if not t:
                    continue
                cand = t if not cur else f"{cur} {t}"
                if len(cand) <= max_chars:
                    cur = cand
                else:
                    if cur:
                        out.append(cur)
                    cur = t
            if cur:
                out.append(cur)
        return out

    def _direct_ct2_translate_chunks(self, text, target_lang):
        src = self._normalize_spaces(text)
        if not src:
            return src
        chunks = self._split_for_direct_translation(src)
        if not chunks:
            return src
        out = []
        for ch in chunks:
            if self._is_protected_segment(ch, block_role="body"):
                out.append(ch)
                continue
            try:
                t = self._ct2_translate(ch, target_lang=target_lang)
            except Exception:
                t = ch
            out.append(self._normalize_spaces(t) or ch)
        return self._normalize_spaces(" ".join(out))

    def _translate_phrase_resilient(self, src_text, target_lang, block_context, block_role, domain, subdomain):
        src, bullet = self._strip_leading_bullets(src_text)
        src = self._sanitize_source_for_translation(src)
        if not src:
            return src_text
        wc = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", src))

        # Professional strict path: for long body phrases, prefer direct CT2 segmentation first.
        direct_first = ""
        if self._normalize_lang_code(target_lang) == "fr" and block_role == "body" and wc >= 6:
            direct_first = self._direct_ct2_translate_chunks(src, target_lang=target_lang)
            direct_first = self._normalize_spaces(direct_first)

        translated = self._translate_text_hierarchical(
            src,
            target_lang=target_lang,
            block_context=block_context,
            block_role=block_role,
            domain=domain,
            subdomain=subdomain,
        )
        translated = self._restore_protected_tokens(src, translated)
        translated = self._normalize_translation(
            translated,
            target_lang=target_lang,
            original=src,
            context_text=block_context,
        )
        translated = self._apply_domain_glossary(
            translated,
            source_text=src,
            target_lang=target_lang,
            domain=domain,
            subdomain=subdomain,
        )
        translated = self._normalize_spaces(translated)

        if direct_first:
            if self._translation_leak_score(direct_first, target_lang) <= self._translation_leak_score(translated, target_lang) + 0.01:
                translated = direct_first

        # If unchanged or language leak too high, force a direct CT2 pass.
        if self._normalize_lang_code(target_lang) != "en":
            unchanged = translated.lower() == src.lower()
            leak = self._translation_leak_score(translated, target_lang)
            leak_src = self._translation_leak_score(src, target_lang)
            if unchanged or leak >= (leak_src - 0.01):
                alt = self._direct_ct2_translate_chunks(src, target_lang=target_lang)
                alt = self._restore_protected_tokens(src, alt)
                alt = self._normalize_translation(
                    alt,
                    target_lang=target_lang,
                    original=src,
                    context_text="",
                )
                alt = self._apply_domain_glossary(
                    alt,
                    source_text=src,
                    target_lang=target_lang,
                    domain=domain,
                    subdomain=subdomain,
                )
                alt = self._normalize_spaces(alt)
                if alt and (alt.lower() != src.lower()):
                    if self._translation_leak_score(alt, target_lang) + 0.015 < leak or unchanged:
                        translated = alt
            # FR strict cleanup pass: reject mixed EN/FR residues when possible.
            if self._normalize_lang_code(target_lang) == "fr":
                en_words = len(re.findall(r"\b(the|and|with|for|from|this|that|are|you|your|will|layers|feature|network)\b", translated, flags=re.IGNORECASE))
                if en_words >= 2:
                    alt2 = self._direct_ct2_translate_chunks(src, target_lang=target_lang)
                    alt2 = self._normalize_spaces(alt2)
                    if alt2 and len(re.findall(r"\b(the|and|with|for|from|this|that|are|you|your|will|layers|feature|network)\b", alt2, flags=re.IGNORECASE)) < en_words:
                        translated = alt2

        if bullet:
            translated = f"{bullet} {translated}".strip()
        return self._normalize_spaces(translated)

    def _reinject_spacing(self, original_chunk, translated_chunk):
        if original_chunk == self._normalize_spaces(original_chunk):
            return translated_chunk
        m1 = re.match(r"^\s+", original_chunk)
        m2 = re.search(r"\s+$", original_chunk)
        left = m1.group(0) if m1 else ""
        right = m2.group(0) if m2 else ""
        return f"{left}{translated_chunk}{right}"

    def _is_acceptable_translation(self, original, translated):
        o = self._normalize_spaces(original)
        t = self._normalize_spaces(translated)
        if not t:
            return False
        if len(t) > max(500, len(o) * 6):
            return False
        # Keep hard-protected source as-is.
        if self._is_protected_segment(o):
            return t == o
        # Reject meta/descriptive artifacts.
        if re.search(r"\b(traduction|phrase|mot|assistant|système|system)\b", t, flags=re.IGNORECASE):
            return False
        # Avoid severe shrink/expansion artifacts.
        lo = max(1, len(o))
        lt = len(t)
        if lt < int(0.35 * lo) or lt > int(3.0 * lo):
            return False
        return True

    def _is_protected_segment(self, text, block_role="body"):
        s = self._normalize_spaces(text)
        if not s:
            return True
        role = (block_role or "body").lower()
        if role in {"diagram_label", "diagram_text_label", "figure_label", "equation_inline", "equation_block"}:
            return True
        # Headers/footers in technical docs often contain references/section markers.
        if role in {"header", "footer"} and len(s) <= 80:
            return True

        # Preserve tiny labels/tokens (diagram points, axis markers, variable marks).
        if len(s) <= 2:
            return True
        if re.fullmatch(r"[A-Z]{1,3}", s):
            return True

        # URLs/emails/file refs: never translate.
        if re.search(r"(https?://|www\.|[\w\.-]+@[\w\.-]+\.\w+|doi:\s*|arxiv:)", s, flags=re.IGNORECASE):
            return True

        # Bibliographic patterns.
        if re.search(r"(et al\.|vol\.|no\.|pp\.|doi|isbn|issn)", s, flags=re.IGNORECASE):
            return True
        if re.search(r"\[[0-9,\-\s]+\]", s):
            return True
        if re.search(r"\([A-Z][A-Za-z\-]+,\s*(19|20)\d{2}\)", s):
            return True

        # Math / physics / chemistry / symbolic expressions.
        if re.search(r"[=<>±×÷∑∫∞≈≠≤≥√∆∂µλΩα-ωΑ-Ω]", s):
            return True
        if re.search(r"\b[a-zA-Z]\s*/\s*[a-zA-Z]\b", s):
            return True
        if re.search(r"\b[dD][A-Za-z]\s*/\s*d[A-Za-z]\b", s):
            return True
        if re.search(r"\b(?:[A-Z][a-z]?\d*){2,}\b", s):
            return True  # chemistry-like formulas (H2SO4, NaCl, CH3COOH...)
        if re.search(r"\b[A-Za-z]+\^\d+\b|\b\d+\s*[x\*]\s*10\^\-?\d+\b", s):
            return True

        # Mostly acronym/abbreviation segment.
        toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", s)
        if toks:
            acr = [t for t in toks if re.fullmatch(r"[A-Z]{2,8}", t)]
            if len(acr) >= max(2, int(0.5 * len(toks))):
                return True
        return False

    def _restore_protected_tokens(self, original, translated):
        o = self._normalize_spaces(original)
        t = self._normalize_spaces(translated)
        if not o or not t:
            return o or t

        # Preserve technical abbreviations/acronyms as-is from original.
        for tok in sorted(set(re.findall(r"\b[A-Z]{2,8}\b", o)), key=len, reverse=True):
            if tok not in t:
                # If model lowercased token, restore canonical uppercase token.
                t = re.sub(rf"\b{re.escape(tok.lower())}\b", tok, t, flags=re.IGNORECASE)

        # Preserve common symbolic/math tokens when present in original.
        protected_tokens = set(re.findall(r"[\[\]\(\)\{\}=<>±×÷∑∫∞≈≠≤≥√∆∂µλΩα-ωΑ-Ω]|[A-Za-z]\d+|d[A-Za-z]|[A-Za-z]/[A-Za-z]", o))
        if protected_tokens:
            # If translation looks too transformed for symbolic content, keep original.
            if len(protected_tokens) >= 2 and self._is_protected_segment(o):
                return o
        return t

    def _sanitize_translation(self, text, original):
        t = (text or "").strip()
        if not t:
            return original

        # Remove common instruction leakage / template markers.
        leak_patterns = [
            r"<\|im_start\|>.*",
            r"<\|im_end\|>.*",
            r"^\s*(system|assistant|user)\s*[:\-].*$",
            r"^\s*(rule|r[èe]gle)\s*[:\-].*$",
            r"^\s*(output constraints|instruction)\s*[:\-].*$",
            r"\b(propose|provide)\s+only\b.*",
        ]
        for pat in leak_patterns:
            t = re.sub(pat, "", t, flags=re.IGNORECASE | re.MULTILINE).strip()

        # Keep only first clean paragraph.
        t = re.split(r"\n{2,}", t)[0].strip()
        t = re.sub(r"\s+", " ", t).strip()

        # Reject obviously corrupted outputs.
        if len(t) < 2:
            return original
        if len(t) > max(400, len(original) * 6):
            return original
        if re.search(r"(rule|r[èe]gle|assistant|system|im_start|im_end)", t, flags=re.IGNORECASE):
            return original

        return t

    def _normalize_spaces(self, text):
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _dedupe_sentence_runs(self, text):
        s = self._normalize_spaces(text)
        parts = [p.strip() for p in re.split(r"(?<=[\.\!\?;:])\s+", s) if p.strip()]
        if not parts:
            return s
        out = []
        for p in parts:
            key = re.sub(r"\W+", "", p).lower()
            if out and re.sub(r"\W+", "", out[-1]).lower() == key:
                continue
            out.append(p)
        return " ".join(out)

    def _normalize_translation(self, text, target_lang="French", original="", context_text=""):
        tgt_code = self._normalize_lang_code(target_lang)
        s = unicodedata.normalize("NFC", self._strip_invisible_chars(text or ""))
        s = self._normalize_spaces(s)
        fixes = {
            "c-ur": "coeur",
            "c-urs": "coeurs",
            "n-ud": "noeud",
            "n-uds": "noeuds",
            "d-": "d'",
            "l-": "l'",
        }
        for k, v in fixes.items():
            s = s.replace(k, v)
        if tgt_code == "fr":
            editorial = [
                (r"\bLaissez-nous utiliser\b", "Utilisons"),
                (r"\bfonctionnalités\b", "caractéristiques"),
                (r"\bcouches de convection\b", "couches convolutionnelles"),
                (r"\bcouches de convolutions\b", "couches convolutionnelles"),
                (r"\bcartes de fonctionnalités\b", "cartes de caractéristiques"),
                (r"\bcaractéristiques cartes\b", "cartes de caractéristiques"),
                (r"\bPrévision des produits\b", "Prédiction en sortie"),
                (r"\bprévision des produits\b", "prédiction en sortie"),
                (r"\baplatie vers un vecteur\b", "aplatie en un vecteur"),
                (r"\baplatir vers un vecteur\b", "aplatir en un vecteur"),
                (r"\bfeatureextraction\b", "extraction de caractéristiques"),
                (r"\bLaissez-les jeter un oeil à\b", "Examinons"),
                (r"\bLaissez-les jeter un œil à\b", "Examinons"),
                (r"\bMaintenant il ya\b", "Maintenant, il y a"),
                (r"\bva continuer osciller\b", "continuera à osciller"),
                (r"\bbaisse des gradients\b", "descente de gradient"),
                (r"\bdescente des gradients\b", "descente de gradient"),
                (r"\bcalculer l['’]avance et l['’]erreur\b", "calculer la propagation avant et l'erreur"),
                (r"\bL['’]établissement d['’]un très grand taux d['’]apprentissage fait osciller l['’]erreur et ne descend jamais\b",
                 "Le réglage d'un taux d'apprentissage très élevé fait osciller l'erreur et l'empêche de descendre"),
            ]
            for pat, repl in editorial:
                s = re.sub(pat, repl, s, flags=re.IGNORECASE)
            # Force standardized translations for frequent technical headings/labels.
            forced_terms = [
                (r"\bTHE\s+DIRECTION\b", "LA DIRECTION"),
                (r"\bTHE\s+STEP\s+SIZE\b", "LA TAILLE DU PAS"),
                (r"\bGOAL\s+WEIGHT\b", "POIDS CIBLE"),
                (r"\bgoal\s+weight\b", "poids cible"),
            ]
            for pat, repl in forced_terms:
                s = re.sub(pat, repl, s, flags=re.IGNORECASE)
            if re.search(r"\bfeedforward\b", original or "", flags=re.IGNORECASE):
                s = re.sub(r"\bavance\b", "propagation avant", s, flags=re.IGNORECASE)
        if self._post_edit_enabled:
            s = self._post_edit_language(s, target_lang=tgt_code, original=original, context_text=context_text)
        s = re.sub(r"\s+([,;:\.\!\?])", r"\1", s)
        s = re.sub(r"([!?])\1+", r"\1", s)
        s = self._dedupe_sentence_runs(s)
        if tgt_code != "en":
            en_markers = self._language_marker_counts(s, "en")
            tgt_markers = self._language_marker_counts(s, tgt_code)
            if en_markers >= 3 and en_markers > max(2, tgt_markers * 2) and original:
                # Avoid orphan English sentence when target is not English.
                return original
        return s

    def _post_edit_language(self, text, target_lang="fr", original="", context_text=""):
        code = self._normalize_lang_code(target_lang)
        profile = self.get_translation_profile(code)
        pedit = profile.get("post_edit", {}) if isinstance(profile, dict) else {}
        if code == "fr":
            if not self._legacy_fr_post_edit:
                return self._normalize_spaces(text)
            return self._post_edit_french(text, original=original, context_text=context_text, profile=pedit)
        return self._post_edit_generic(text, target_lang=code, profile=pedit)

    def _post_edit_generic(self, text, target_lang="en", profile=None):
        s = self._normalize_spaces(text)
        if not s:
            return s
        profile = profile or {}
        for row in profile.get("generic_replacements", []) if isinstance(profile, dict) else []:
            pat = row.get("pattern")
            repl = row.get("replace", "")
            if pat:
                s = re.sub(pat, repl, s, flags=re.IGNORECASE)
        # Generic cleanup valid for most languages.
        s = re.sub(r"[ \t]+([,;:\.\!\?])", r"\1", s)
        s = re.sub(r"([!?])\1+", r"\1", s)
        return self._normalize_spaces(s)

    def _post_edit_french(self, text, original="", context_text="", profile=None):
        s = self._normalize_spaces(text)
        if not s:
            return s
        ctx = self._normalize_spaces(context_text).lower()
        src = self._normalize_spaces(original).lower()
        profile = profile or {}

        # Targeted fluency fixes observed on long technical fragments.
        replacements = [
            (r"\bLes gens disent que c['’]est\b", "Supposons que ce soit"),
            (r"\bMaintenant,\s*il y a une chose qui reste\b", "Il reste une chose"),
            (r"\bQuelle taille devrait être la taille de l['’]étape\??\b", "Quelle doit être la taille de l'étape ?"),
            (r"\bUtilisons de grands taux d['’]apprentissage et compléter\b", "Utilisons de grands taux d'apprentissage et complétons"),
            (r"\bNous parlerons plus tard sur le réglage\b", "Nous parlerons plus tard du réglage"),
            (r"\ble réseau va éventuellement\b", "le réseau finira par"),
            (r"\bL['’]erreur va continuer à osciller\b", "L'erreur continuera à osciller"),
            (r"\bIl pourrait être un pas de 1 pied\b", "Ce peut être un pas d'un pied"),
            (r"\bun saut de 100 pieds\b", "un saut de cent pieds"),
            (r"\bnous redémarrons le processus\b", "nous relançons le processus"),
            (r"\bdu taux d['’]apprentissage et comment déterminer si l['’]erreur est oscillante\b",
             "du taux d'apprentissage et de la façon de déterminer si l'erreur oscille"),
            (r"\bfonctionnalités\b", "caractéristiques"),
            (r"\bcartes de fonctionnalités\b", "cartes de caractéristiques"),
            (r"\bcaractéristiques cartes\b", "cartes de caractéristiques"),
            (r"\bPrévision des produits\b", "Prédiction en sortie"),
            (r"\bprévision des produits\b", "prédiction en sortie"),
            (r"\bcouches connectées localement\b", "couches convolutionnelles"),
            (r"\bcouches de convolution\b", "couches convolutionnelles"),
            (r"\bcouches de convolutions\b", "couches convolutionnelles"),
            (r"\bcalque\b", "couche"),
            (r"\bfeatureextraction\b", "extraction de caractéristiques"),
        ]
        for row in profile.get("generic_replacements", []) if isinstance(profile, dict) else []:
            pat = row.get("pattern")
            repl = row.get("replace", "")
            if pat:
                replacements.append((pat, repl))
        for pat, repl in replacements:
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)

        # Contextual consistency for technical paragraphs.
        if "gradient descent" in ctx or "descente de gradient" in ctx or "gradient descent" in src:
            s = re.sub(r"\bdescente la plus profonde\b", "plus forte descente", s, flags=re.IGNORECASE)
        if "learning rate" in ctx or "taux d'apprentissage" in ctx or "learning rate" in src:
            s = re.sub(r"\bvitesse d['’]apprentissage\b", "taux d'apprentissage", s, flags=re.IGNORECASE)
        if "error mountain" in ctx or "mountain" in ctx:
            s = re.sub(r"\bmontagne de l['’]erreur\b", "montagne d'erreur", s, flags=re.IGNORECASE)

        # Minimal punctuation typography for French.
        s = re.sub(r"\s*\?\s*", " ? ", s)
        s = re.sub(r"\s*!\s*", " ! ", s)
        s = re.sub(r"\?{2,}", "?", s)
        s = re.sub(r"!{2,}", "!", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    def _strict_fr_phrase_pass(self, text, source_text="", context_text="", previous_translations=None):
        s = self._normalize_spaces(text)
        if not s:
            return s
        if self._is_protected_segment(source_text):
            return s
        # Keep very short fragments stable (already handled by short-fragment logic).
        if len(re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'\-]*", s)) < 3:
            return s
        s = self.post_edit_paragraph_sentence(
            s,
            target_lang="fr",
            source_text=source_text,
            context_text=context_text,
            previous_translations=previous_translations,
        )
        # Extra strict cleanup for frequent MT artifacts in technical docs.
        strict_fixes = [
            (r"\bapprentissage en profondeur\b", "apprentissage profond"),
            (r"\bsystèmes de vision\b", "systèmes de vision par ordinateur"),
        ]
        for pat, repl in strict_fixes:
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)
        return self._normalize_spaces(s)

    def post_edit_paragraph_sentence(self, text, target_lang="French", source_text="", context_text="", previous_translations=None):
        s = self._normalize_spaces(text)
        if not s:
            return s
        prev = " ".join(self._normalize_spaces(x) for x in (previous_translations or []) if x)
        merged_ctx = self._normalize_spaces(f"{context_text} {prev}")
        tgt_code = self._normalize_lang_code(target_lang)
        s = self._post_edit_language(s, target_lang=tgt_code, original=source_text, context_text=merged_ctx)

        # Intra-paragraph consistency: reuse dominant preferred terms.
        if tgt_code == "fr":
            low_prev = prev.lower()
            if "taux d'apprentissage" in low_prev:
                s = re.sub(r"\b(vitesse|rythme)\s+d['’]apprentissage\b", "taux d'apprentissage", s, flags=re.IGNORECASE)
            if "descente de gradient" in low_prev:
                s = re.sub(r"\bgradient descent\b", "descente de gradient", s, flags=re.IGNORECASE)
            if "propagation avant" in low_prev:
                s = re.sub(r"\bfeedforward\b", "propagation avant", s, flags=re.IGNORECASE)
        return self._normalize_spaces(s)

    # Backward compatibility
    def post_edit_french_paragraph_sentence(self, text, source_text="", context_text="", previous_translations=None):
        return self.post_edit_paragraph_sentence(
            text,
            target_lang="fr",
            source_text=source_text,
            context_text=context_text,
            previous_translations=previous_translations,
        )

    def _post_dedupe_translated_blocks(self, structure):
        blocks = structure.get("blocks", [])
        kept = []
        for b in blocks:
            txt = self._normalize_spaces(b.get("translated_text") or "")
            bb = b.get("bbox", [0, 0, 0, 0])
            if len(bb) != 4:
                kept.append(b)
                continue
            bx0, by0, bx1, by1 = [float(v) for v in bb]
            area = max(1.0, (bx1 - bx0) * (by1 - by0))
            is_dup = False
            for kb in kept:
                ktxt = self._normalize_spaces(kb.get("translated_text") or "")
                kbb = kb.get("bbox", [0, 0, 0, 0])
                if len(kbb) != 4:
                    continue
                kx0, ky0, kx1, ky1 = [float(v) for v in kbb]
                ix0, iy0 = max(bx0, kx0), max(by0, ky0)
                ix1, iy1 = min(bx1, kx1), min(by1, ky1)
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if inter <= 0:
                    continue
                karea = max(1.0, (kx1 - kx0) * (ky1 - ky0))
                ov = inter / max(1.0, min(area, karea))
                if ov < 0.55:
                    continue
                if txt and ktxt and (txt == ktxt or txt in ktxt or ktxt in txt):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(b)
        structure["blocks"] = kept

    def _normalize_span_style(self, span, role="body"):
        st = span.get("style")
        if not isinstance(st, dict):
            return
        c = (st.get("color") or "#000000").lstrip("#")
        if len(c) != 6:
            return
        try:
            r = int(c[0:2], 16) / 255.0
            g = int(c[2:4], 16) / 255.0
            b = int(c[4:6], 16) / 255.0
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if role == "body" and lum > 0.82:
                st["color"] = "#101010"
        except Exception:
            return
