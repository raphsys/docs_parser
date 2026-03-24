import re
from typing import Dict, List, Tuple


class ContextClassifier:
    def __init__(self):
        self._domain_lexicon = {
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
        self._subdomain_lexicon = {
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
            "technology": {
                "software": ["software architecture", "refactoring", "testing", "deployment", "dependency"],
                "data": ["database", "etl", "data warehouse", "query optimization", "schema"],
                "cloud": ["cloud", "container", "kubernetes", "autoscaling", "infrastructure as code"],
                "cybersecurity": ["encryption", "vulnerability", "threat model", "authentication", "authorization"],
            },
        }

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    def _score(self, text: str, keywords: List[str]) -> int:
        return sum(1 for keyword in keywords if keyword in text)

    def detect_domain(self, text: str) -> Tuple[str, float]:
        normalized = self._normalize_text(text)
        if not normalized:
            return "general", 0.0
        scores: Dict[str, int] = {
            domain: self._score(normalized, keywords)
            for domain, keywords in self._domain_lexicon.items()
        }
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        if scores[best] <= 0:
            return "general", 0.0
        return best, round(scores[best] / max(1, total), 4)

    def detect_subdomain(self, text: str, domain: str) -> Tuple[str, float]:
        normalized = self._normalize_text(text)
        subdomains = self._subdomain_lexicon.get((domain or "").lower(), {})
        if not normalized or not subdomains:
            return "", 0.0
        scores: Dict[str, int] = {
            subdomain: self._score(normalized, keywords)
            for subdomain, keywords in subdomains.items()
        }
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        if scores[best] <= 0:
            return "", 0.0
        return best, round(scores[best] / max(1, total), 4)

    def classify(self, text: str, page_text: str = "", document_text: str = "") -> Dict[str, object]:
        combined = " ".join(x for x in [document_text, page_text, text] if x).strip()
        domain, domain_confidence = self.detect_domain(combined)
        subdomain, subdomain_confidence = self.detect_subdomain(combined, domain)
        return {
            "domain": domain,
            "domain_confidence": domain_confidence,
            "subdomain": subdomain,
            "subdomain_confidence": subdomain_confidence,
        }
