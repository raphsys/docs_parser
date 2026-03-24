class TranslationValidator:
    def evaluate(self, source_text, translated_text, terminology_report, source_leak_score):
        findings = []
        if not translated_text:
            findings.append("empty_translation")
        if source_text and translated_text and source_text.strip().lower() == translated_text.strip().lower():
            findings.append("unchanged_translation")
        if terminology_report and not terminology_report.get("ok", True):
            findings.append("reserved_term_mismatch")
        if source_leak_score > 1.15:
            findings.append("source_language_leak")
        return {"ok": len(findings) == 0, "findings": findings}
