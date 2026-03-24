import csv
import json
import os
import re
from typing import Dict, List, Optional


class TerminologyManager:
    def __init__(self, terminology_table_path: Optional[str] = None):
        self.terminology_table_path = terminology_table_path or os.getenv(
            "TRANSLATOR_TERMINOLOGY_TABLE",
            "ai_models/translation/glossaries/terminology_master.csv",
        )
        self._entries: List[Dict[str, object]] = []
        self._load_terminology_table()

    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _normalize_bool(self, value: object) -> bool:
        return self._normalize_spaces("" if value is None else str(value)).lower() in {"1", "true", "yes", "on"}

    def _load_terminology_table(self):
        path = self.terminology_table_path
        if not path or not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                entry = {}
                for key, value in row.items():
                    entry[(key or "").strip()] = self._normalize_spaces("" if value is None else str(value))
                if not entry.get("term_id"):
                    continue
                entry["domain"] = (entry.get("domain") or "general").lower()
                entry["subdomain"] = (entry.get("subdomain") or "").lower()
                entry["doc_role"] = (entry.get("doc_role") or "all").lower()
                entry["match_type"] = (entry.get("match_type") or "phrase").lower()
                entry["priority"] = int(entry.get("priority") or "100")
                entry["locked"] = self._normalize_bool(entry.get("locked"))
                entry["forbid_literal_translation"] = self._normalize_bool(entry.get("forbid_literal_translation"))
                try:
                    entry["aliases"] = json.loads(entry.get("aliases_json") or "[]")
                except Exception:
                    entry["aliases"] = []
                self._entries.append(entry)

    def resolve_terms(self, source_lang: str, target_lang: str, domain: str = "general", subdomain: str = "", doc_role: str = "all") -> List[Dict[str, object]]:
        src = (source_lang or "").lower()
        tgt = (target_lang or "").lower()
        dom = (domain or "general").lower()
        sub = (subdomain or "").lower()
        role = (doc_role or "all").lower()
        out = []
        for entry in self._entries:
            if entry.get("domain") not in {"general", dom}:
                continue
            if entry.get("subdomain") and entry.get("subdomain") != sub:
                continue
            if entry.get("doc_role") not in {"all", role}:
                continue
            source_text = self._normalize_spaces(entry.get(src) or "")
            target_text = self._normalize_spaces(entry.get(tgt) or "")
            if not source_text or not target_text:
                continue
            resolved = dict(entry)
            resolved["source_text"] = source_text
            resolved["target_text"] = target_text
            resolved["source_lang"] = src
            resolved["target_lang"] = tgt
            resolved["aliases"] = [self._normalize_spaces(x) for x in entry.get("aliases") or [] if self._normalize_spaces(x)]
            out.append(resolved)
        out.sort(key=lambda item: (-int(item.get("priority") or 0), -len(str(item.get("source_text") or ""))))
        return out

    def infer_context(self, text: str, source_lang: str, doc_role: str = "all") -> Dict[str, object]:
        normalized = self._normalize_spaces(text).lower()
        src = (source_lang or "").lower()
        role = (doc_role or "all").lower()
        if not normalized:
            return {"domain": "general", "subdomain": "", "confidence": 0.0}
        scores: Dict[str, int] = {}
        sub_scores: Dict[str, int] = {}
        for entry in self._entries:
            if entry.get("doc_role") not in {"all", role}:
                continue
            source_text = self._normalize_spaces(entry.get(src) or "").lower()
            aliases = [self._normalize_spaces(x).lower() for x in entry.get("aliases") or []]
            candidates = [source_text] + aliases
            if not any(candidate and re.search(rf"(?i)\b{re.escape(candidate)}\b", normalized) for candidate in candidates):
                continue
            domain = str(entry.get("domain") or "general").lower()
            subdomain = str(entry.get("subdomain") or "").lower()
            weight = max(1, int(entry.get("priority") or 100) // 50)
            scores[domain] = scores.get(domain, 0) + weight
            if subdomain:
                key = f"{domain}:{subdomain}"
                sub_scores[key] = sub_scores.get(key, 0) + weight
        if not scores:
            return {"domain": "general", "subdomain": "", "confidence": 0.0}
        best_domain = max(scores, key=scores.get)
        domain_total = sum(scores.values())
        best_subdomain = ""
        if sub_scores:
            filtered = {k: v for k, v in sub_scores.items() if k.startswith(f"{best_domain}:")}
            if filtered:
                best_key = max(filtered, key=filtered.get)
                best_subdomain = best_key.split(":", 1)[1]
        return {
            "domain": best_domain,
            "subdomain": best_subdomain,
            "confidence": round(scores[best_domain] / max(1, domain_total), 4),
        }

    def exact_match(self, text: str, source_lang: str, target_lang: str, domain: str = "general", subdomain: str = "", doc_role: str = "all") -> Optional[Dict[str, object]]:
        normalized = self._normalize_spaces(text).lower()
        if not normalized:
            return None
        for entry in self.resolve_terms(source_lang, target_lang, domain=domain, subdomain=subdomain, doc_role=doc_role):
            candidates = [str(entry.get("source_text") or "").lower()] + [str(x).lower() for x in entry.get("aliases") or []]
            if normalized in candidates:
                return entry
        return None

    def apply_output_terms(self, text: str, source_text: str, source_lang: str, target_lang: str, domain: str = "general", subdomain: str = "", doc_role: str = "all") -> str:
        out = text or ""
        source_lc = self._normalize_spaces(source_text).lower()
        for entry in self.resolve_terms(source_lang, target_lang, domain=domain, subdomain=subdomain, doc_role=doc_role):
            candidates = [str(entry.get("source_text") or "").lower()] + [str(x).lower() for x in entry.get("aliases") or []]
            if not any(candidate and re.search(rf"(?i)\b{re.escape(candidate)}\b", source_lc) for candidate in candidates):
                continue
            target_value = str(entry.get("target_text") or "")
            for candidate in candidates:
                if candidate:
                    out = re.sub(rf"(?i)\b{re.escape(candidate)}\b", target_value, out)
        return out

    def validate_reserved_terms(self, source_text: str, translated_text: str, source_lang: str, target_lang: str, domain: str = "general", subdomain: str = "", doc_role: str = "all") -> Dict[str, object]:
        findings = []
        source_lc = self._normalize_spaces(source_text).lower()
        translated_lc = self._normalize_spaces(translated_text).lower()
        relevant = 0
        for entry in self.resolve_terms(source_lang, target_lang, domain=domain, subdomain=subdomain, doc_role=doc_role):
            candidates = [str(entry.get("source_text") or "").lower()] + [str(x).lower() for x in entry.get("aliases") or []]
            if not any(candidate and re.search(rf"(?i)\b{re.escape(candidate)}\b", source_lc) for candidate in candidates):
                continue
            relevant += 1
            target_value = str(entry.get("target_text") or "").lower()
            if target_value and not re.search(rf"(?i)\b{re.escape(target_value)}\b", translated_lc):
                findings.append({
                    "term_id": entry.get("term_id"),
                    "source_text": entry.get("source_text"),
                    "target_text": entry.get("target_text"),
                    "status": "missing_target_term",
                })
        return {"ok": len(findings) == 0, "findings": findings, "relevant_term_count": relevant}
