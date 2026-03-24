import re
from typing import Dict, List, Tuple


class StyleToneClassifier:
    def __init__(self):
        self._styles = {
            "academique": ["therefore", "moreover", "however", "hypothesis", "methodology", "results", "discussion"],
            "professionnel": ["implementation", "process", "stakeholder", "deliverable", "compliance", "workflow"],
            "journalistique": ["reported", "according to", "witness", "breaking", "interview", "newsroom"],
            "reporter": ["scene", "field", "on the ground", "observed", "reported from"],
            "pedagogique": ["for example", "let us", "suppose", "consider", "remember", "in other words"],
            "technique": ["architecture", "algorithm", "system", "interface", "input", "output", "layer"],
            "scientifique": ["experiment", "analysis", "sample", "observation", "measurement", "evidence"],
            "administratif": ["request", "approval", "procedure", "applicant", "authority", "file"],
            "juridique": ["court", "statute", "regulation", "liability", "jurisdiction", "compliance"],
            "marketing": ["solution", "benefit", "value", "brand", "customer", "offer"],
            "conversationnel": ["hi", "hello", "thanks", "please", "let's", "you"],
            "narratif": ["story", "character", "suddenly", "then", "afterward", "memory"],
        }
        self._tones = {
            "formel": ["therefore", "furthermore", "pursuant", "respectfully", "accordingly"],
            "neutre": ["the", "and", "with", "for", "from"],
            "serieux": ["risk", "critical", "failure", "safety", "important", "warning"],
            "amical": ["hello", "thanks", "glad", "welcome", "please"],
            "didactique": ["for example", "consider", "suppose", "let us", "remember"],
            "analytique": ["analysis", "evidence", "measure", "assess", "compare"],
            "persuasif": ["best", "proven", "powerful", "effective", "benefit"],
            "grave": ["death", "crisis", "severe", "catastrophic", "emergency"],
            "enthousiaste": ["great", "excellent", "exciting", "remarkable", "impressive"],
            "humoristique": ["joke", "funny", "laugh", "ironic", "comical"],
            "derision": ["mock", "ridiculous", "absurd", "sarcastic", "ironic"],
        }

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    def _score(self, text: str, keywords: List[str]) -> int:
        return sum(1 for keyword in keywords if keyword in text)

    def _detect(self, text: str, lexicon: Dict[str, List[str]], fallback: str) -> Tuple[str, float]:
        normalized = self._normalize(text)
        if not normalized:
            return fallback, 0.0
        scores = {key: self._score(normalized, keywords) for key, keywords in lexicon.items()}
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        if scores[best] <= 0:
            return fallback, 0.0
        return best, round(scores[best] / max(1, total), 4)

    def classify(self, text: str, block_role: str = "body", domain: str = "general") -> Dict[str, object]:
        combined = " ".join(x for x in [domain, block_role, text] if x)
        style, style_confidence = self._detect(combined, self._styles, "professionnel")
        tone, tone_confidence = self._detect(combined, self._tones, "neutre")
        if (block_role or "").lower() in {"title", "section_heading", "figure_caption"} and style == "conversationnel":
            style = "professionnel"
        return {
            "style": style,
            "style_confidence": style_confidence,
            "tone": tone,
            "tone_confidence": tone_confidence,
        }
